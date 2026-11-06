"""Cherrypick interesting steered records across one or more experiments.

Walks ``records.jsonl`` files under one or more experiment directories,
applies a filter (default: strength_mean_coh<=1, persona>=2,
|effect.combined|>=2, configurable), and emits the matching records plus
a small summary table broken down by experiment / cell / strength.

Filter design:
    - The default thresholds reflect what's interesting for paper /
      report cherrypicks: coherent at the strength level, in persona,
      with a clearly-non-trivial steering effect.
    - The strength_mean_coh filter is on the strength-mean rather than
      per-record coh because incoherence is a strength-level property
      whereas judge noise is per-question (same reasoning that drives
      the dispatcher's skip-or-judge filter).  Records skipped from
      RP/effect judging via skip_due_to_strength_mean_coh would not
      pass anyway, but filtering on strength_mean_coh first means we
      drop them without examining their (null) persona/effect fields.
    - --filter takes a Python expression evaluated against each record;
      see EVAL_NAMESPACE_DOC below for what's bound.

Sort order: descending by |effect.combined|, then by question_idx, so
the most-extreme effects float to the top of --print-N output.

Usage::

    uv run python -m assistant_axis.cherrypick \\
        --experiment_dirs /workspace/outputs/qwen-3-32b/steering/* \\
        --print-N 20 \\
        --output cherrypicks.jsonl

    uv run python -m assistant_axis.cherrypick \\
        --experiment_dirs /workspace/outputs/qwen-3-32b/steering/smoke_test_v1 \\
        --filter "abs(record['judges']['effect']['combined']) >= 2.5" \\
        --print-N 5
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .atomic_io import read_jsonl_with_retry, atomic_write_text

logger = logging.getLogger(__name__)


EVAL_NAMESPACE_DOC = """\
Each record passes through a Python expression with these names bound:

  record:       the full record dict (judges, strength, response, ...)
  effect:       record['judges'].get('effect', {}) (may be None or {})
  persona:      record['judges'].get('persona', {}) (may be None or {})
  coh:          record['judges'].get('coherence', {}).get('score')
                (per-record coherence; usually you want strength_mean_coh)
  smc:          record['judges'].get('strength_mean_coh')
                (the canonical 'is this strength incoherent?' value)
  combined:     effect.get('combined') if effect is a dict, else None
  abs_eff:      abs(combined) if combined is not None, else 0.0
  strength:     record['strength']
  sign:         record['sign']
  experiment:   experiment dir name (basename)
  cell:         cell dir name (e.g. 's3_l25_+1')

Expressions must evaluate to a truthy value for the record to pass.
"""


def _safe_get(d: Optional[Dict], key: str, default: Any = None) -> Any:
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


def _extract_combined(record: Dict[str, Any]) -> Optional[float]:
    eff = _safe_get(record.get("judges"), "effect")
    if not isinstance(eff, dict):
        return None
    combined = eff.get("combined")
    if combined is None:
        return None
    try:
        return float(combined)
    except (TypeError, ValueError):
        return None


def default_filter(
    record: Dict[str, Any],
    *,
    max_strength_mean_coh: float = 1.0,
    min_persona: int = 2,
    min_abs_effect: float = 2.0,
) -> bool:
    """The cherrypick default: coherent strength, in persona, strong effect.

    Returns False (drops the record) for any of:
      - strength_mean_coh missing or > threshold
      - persona score missing or < threshold
      - effect.combined missing or |effect| < threshold
    """
    judges = record.get("judges") or {}
    smc = judges.get("strength_mean_coh")
    if smc is None or float(smc) > max_strength_mean_coh:
        return False

    persona = judges.get("persona")
    persona_score = _safe_get(persona, "score")
    if not isinstance(persona_score, int) or persona_score < min_persona:
        return False

    combined = _extract_combined(record)
    if combined is None or abs(combined) < min_abs_effect:
        return False

    return True


def make_expr_filter(expression: str) -> Callable[[Dict[str, Any]], bool]:
    """Compile a user-supplied Python expression into a filter function.

    Bindings are documented in :data:`EVAL_NAMESPACE_DOC`.  The
    expression is compiled once and reused per record; we trust the
    caller's expression (this CLI is local-only, no untrusted input).
    """
    code = compile(expression, "<cherrypick-filter>", "eval")

    def _check(record: Dict[str, Any]) -> bool:
        judges = record.get("judges") or {}
        effect = judges.get("effect") if isinstance(judges.get("effect"), dict) else {}
        persona = judges.get("persona") if isinstance(judges.get("persona"), dict) else {}
        coherence = judges.get("coherence") if isinstance(judges.get("coherence"), dict) else {}
        combined = _extract_combined(record)
        ns = {
            "record": record,
            "judges": judges,
            "effect": effect or {},
            "persona": persona or {},
            "coherence": coherence or {},
            "coh": coherence.get("score") if isinstance(coherence, dict) else None,
            "smc": judges.get("strength_mean_coh"),
            "combined": combined,
            "abs_eff": abs(combined) if combined is not None else 0.0,
            "strength": record.get("strength"),
            "sign": record.get("sign"),
            "experiment": record.get("_experiment"),
            "cell": record.get("_cell"),
            "abs": abs, "min": min, "max": max,
        }
        try:
            return bool(eval(code, {"__builtins__": {}}, ns))  # noqa: S307
        except Exception as e:  # noqa: BLE001
            logger.warning(f"filter expression raised on record: {e}")
            return False

    return _check


def iter_records(
    experiment_dirs: Iterable[Path],
) -> Iterable[Dict[str, Any]]:
    """Yield every record from every cell under each experiment dir.

    Each record is annotated with ``_experiment`` (basename of the
    experiment dir) and ``_cell`` (basename of the cell subdir) so
    downstream filters and sorts can group cleanly.
    """
    for exp_dir in experiment_dirs:
        exp_dir = Path(exp_dir)
        if not exp_dir.is_dir():
            logger.warning(f"experiment dir not found: {exp_dir}")
            continue
        # Each cell subdir owns one records.jsonl.  Baseline records live
        # in baselines/records.jsonl and are usually uninteresting for
        # cherrypicking (strength=0, no steering effect to filter on).
        for cell_dir in sorted(exp_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            if cell_dir.name == "baselines":
                continue
            records_file = cell_dir / "records.jsonl"
            if not records_file.exists():
                continue
            for r in read_jsonl_with_retry(records_file, logger_obj=logger):
                r["_experiment"] = exp_dir.name
                r["_cell"] = cell_dir.name
                yield r


def summarise(records: List[Dict[str, Any]]) -> str:
    """Print a small summary table of matched records."""
    by_exp_cell_strength: Counter = Counter()
    by_exp: Counter = Counter()
    for r in records:
        exp = r.get("_experiment", "?")
        cell = r.get("_cell", "?")
        strength = r.get("strength")
        by_exp_cell_strength[(exp, cell, strength)] += 1
        by_exp[exp] += 1

    lines: List[str] = []
    lines.append(f"matched {len(records)} record(s) across {len(by_exp)} experiment(s)")
    if not by_exp:
        return "\n".join(lines)
    lines.append("")
    lines.append(f"  {'experiment':<32}  {'cell':<14}  {'strength':>10}  count")
    for (exp, cell, strength), n in sorted(by_exp_cell_strength.items()):
        s_str = f"{strength:.4g}" if isinstance(strength, (int, float)) else str(strength)
        lines.append(f"  {exp:<32}  {cell:<14}  {s_str:>10}  {n:>5d}")
    return "\n".join(lines)


def print_top_n(records: List[Dict[str, Any]], n: int) -> None:
    """Pretty-print the top N records by absolute effect."""
    if n <= 0 or not records:
        return
    print()
    print(f"=== top {min(n, len(records))} by |effect.combined| ===")
    for i, r in enumerate(records[:n], 1):
        combined = _extract_combined(r)
        smc = (r.get("judges") or {}).get("strength_mean_coh")
        persona_score = _safe_get(_safe_get(r.get("judges"), "persona"), "score")
        print()
        print(f"--- {i}/{n}  exp={r.get('_experiment')}  cell={r.get('_cell')}  "
              f"strength={r.get('strength')}  sign={r.get('sign')}  q_idx={r.get('question_idx')}")
        print(f"    effect.combined={combined!r}  strength_mean_coh={smc!r}  "
              f"persona={persona_score!r}")
        question = r.get("question", "")
        response = r.get("response", "")
        print(f"    Q: {question[:200]}")
        print(f"    A: {response[:400]}{'...' if len(response) > 400 else ''}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EVAL_NAMESPACE_DOC,
    )
    p.add_argument("--experiment_dirs", nargs="+", required=True, type=Path,
                   help="One or more experiment dirs (each containing cell "
                        "subdirs with records.jsonl).")
    p.add_argument("--filter", default=None,
                   help="Python expression evaluated per-record; if "
                        "supplied, replaces the default thresholds.  See "
                        "the EVAL_NAMESPACE in --help epilog.")
    p.add_argument("--max-strength-mean-coh", type=float, default=1.0,
                   help="Default filter: max strength_mean_coh allowed.")
    p.add_argument("--min-persona", type=int, default=2,
                   help="Default filter: min persona score (0-3).")
    p.add_argument("--min-abs-effect", type=float, default=2.0,
                   help="Default filter: min |effect.combined|.")
    p.add_argument("--output", type=Path, default=None,
                   help="If set, write matched records (annotated with "
                        "_experiment/_cell) as JSONL to this path.")
    p.add_argument("--print-N", type=int, default=10, dest="print_n",
                   help="Pretty-print the top N records by |effect.combined| "
                        "(default 10; pass 0 to disable).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if args.filter:
        check = make_expr_filter(args.filter)
        logger.info(f"using custom filter: {args.filter!r}")
    else:
        def check(r: Dict[str, Any]) -> bool:
            return default_filter(
                r,
                max_strength_mean_coh=args.max_strength_mean_coh,
                min_persona=args.min_persona,
                min_abs_effect=args.min_abs_effect,
            )
        logger.info(
            f"using default filter: strength_mean_coh<={args.max_strength_mean_coh} "
            f"AND persona>={args.min_persona} AND "
            f"|effect.combined|>={args.min_abs_effect}"
        )

    matched: List[Dict[str, Any]] = []
    total = 0
    for r in iter_records(args.experiment_dirs):
        total += 1
        if check(r):
            matched.append(r)

    matched.sort(
        key=lambda r: (-(abs(_extract_combined(r) or 0.0)),
                       int(r.get("question_idx", 0))),
    )

    print(summarise(matched))
    print_top_n(matched, args.print_n)

    if args.output is not None:
        lines = [json.dumps(r, ensure_ascii=False) for r in matched]
        atomic_write_text("\n".join(lines) + ("\n" if lines else ""),
                          args.output)
        logger.info(f"wrote {len(matched)} matched record(s) to {args.output}")

    logger.info(f"scanned {total} record(s); matched {len(matched)}")


if __name__ == "__main__":
    main()
