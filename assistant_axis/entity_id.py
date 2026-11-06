"""Disambiguated entity identifiers for trait/role mixing contexts.

The project's corpus has 11 names that appear in *both* the trait and
role lists -- ``ascetic``, ``contrarian``, ``cosmopolitan``,
``generalist``, ``pacifist``, ``parent``, ``patient``,
``perfectionist``, ``romantic``, ``specialist``, ``stoic`` (checked
against the files by ``tests/test_collision_regression.py``).  Bare
names are therefore *not* a unique identifier in any context that
mixes kinds (dict keys, set members,
JSON cache keys, sorted name lists for rho computation, ...).
Historically several sites silently dropped one side of every
collision via the dict-overwrite ``merged[name] = ...`` pattern,
biasing rho calculations by ~1.5 %.

This module provides the canonical disambiguator: pipe-suffixed
single-letter kind tags.

    >>> entity_id("patient", "roles")
    'patient|R'
    >>> entity_id("patient", "T")
    'patient|T'
    >>> parse_entity_id("patient|R")
    EntityId(name='patient', kind='roles')
    >>> display_label("patient|R")
    'patient'

Use ``entity_id(name, kind)`` whenever a data structure could
plausibly contain entries from both kinds.  Use ``display_label(eid)``
on the *display* side to strip the suffix; encode kind visually via
the helpers in :mod:`assistant_axis.plot_palette` (``kind_color``,
``kind_marker``, ...).

Pure-kind contexts (e.g. inside a ``runpod_workspace/.../roles/``
directory, or a per-cohort ``haiku_responses_traits_*/`` cache) keep
bare names; the kind is implicit in the path.

See ``AGENT_NOTES.md`` (section "Trait/role name collisions") for
the full convention and pitfalls.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

__all__ = [
    "EntityId",
    "KIND_R",
    "KIND_T",
    "KIND_LONG",
    "entity_id",
    "parse_entity_id",
    "is_entity_id",
    "display_label",
    "kind_short",
    "kind_long",
    "ID_SEPARATOR",
    "normalize_to_file_name",
    "display_form_name",
    "corpus_display_name",
    "clear_corpus_display_cache",
    "default_data_dir",
    "resolve_renamed_stem",
    "ROLE_DISPLAY_OVERRIDES",
    "STANDARD_SUFFIX_FORMS",
    "JUDGE_LABEL_FORM",
    "judge_form_of_label",
    "judge_form_of_negative_label",
    "judge_label",
    "judge_negative_label",
]


# Canonical short tags used inside the disambiguated id.
KIND_R: str = "R"  # role
KIND_T: str = "T"  # trait

# Canonical long-form kind tokens used elsewhere in the project
# (matching ``--pair_type {traits,roles}`` and the corpus directory
# names ``runpod_workspace/.../{roles,traits}/``).
_KIND_ROLES = "roles"
_KIND_TRAITS = "traits"

#: Maps the short tag back to the long-form kind token used elsewhere
#: in the project.  Use :func:`kind_long` rather than indexing this
#: dict directly when you have a short tag in hand.
KIND_LONG: dict[str, str] = {KIND_R: _KIND_ROLES, KIND_T: _KIND_TRAITS}

# Accepted spellings for each kind on input (normalised before
# resolving to the canonical short tag).  Lower-case keys; callers
# may pass any case.
_KIND_ALIASES_TO_SHORT: dict[str, str] = {
    "r": KIND_R,
    "role": KIND_R,
    "roles": KIND_R,
    "t": KIND_T,
    "trait": KIND_T,
    "traits": KIND_T,
}

#: The separator character between name and short-kind tag.
ID_SEPARATOR: str = "|"


class EntityId(NamedTuple):
    """The parsed pieces of a disambiguated entity id.

    ``name`` is the bare entity name (e.g. ``"patient"``); ``kind`` is
    the canonical long-form kind token (``"roles"`` or ``"traits"``)
    matching the rest of the project's vocabulary (corpus
    subdirectories, ``--pair_type`` CLI argument, etc.).
    """
    name: str
    kind: str  # "roles" or "traits"


def kind_short(kind: str) -> str:
    """Normalise a kind token to its canonical short tag (``"R"`` or
    ``"T"``).

    Accepts: ``"R"``, ``"T"``, ``"role"``, ``"roles"``, ``"trait"``,
    ``"traits"`` (any case).  Raises :class:`ValueError` on anything
    else.
    """
    if not isinstance(kind, str):
        raise ValueError(
            f"entity_id: kind must be a string, got {type(kind).__name__} "
            f"({kind!r})"
        )
    norm = kind.strip().lower()
    if norm not in _KIND_ALIASES_TO_SHORT:
        raise ValueError(
            f"entity_id: unknown kind {kind!r}; accepted: "
            f"{sorted(_KIND_ALIASES_TO_SHORT)}"
        )
    return _KIND_ALIASES_TO_SHORT[norm]


def kind_long(kind: str) -> str:
    """Normalise a kind token to its canonical long-form name
    (``"roles"`` or ``"traits"``).

    Accepts the same set of spellings as :func:`kind_short`.
    """
    return KIND_LONG[kind_short(kind)]


def entity_id(name: str, kind: str) -> str:
    """Return the canonical disambiguated id for ``(name, kind)``.

    The format is ``"<name>|R"`` for roles, ``"<name>|T"`` for traits.
    The name itself must not contain the pipe separator.

    Args:
        name: Bare entity name (e.g. ``"patient"``).
        kind: Any accepted kind token (``"R"``, ``"T"``, ``"role"``,
            ``"roles"``, ``"trait"``, ``"traits"``; case-insensitive).

    Raises:
        ValueError: If ``name`` is empty, contains :data:`ID_SEPARATOR`,
            or if ``kind`` isn't recognised.

    >>> entity_id("patient", "roles")
    'patient|R'
    >>> entity_id("patient", "T")
    'patient|T'
    >>> entity_id("patient", "Roles")
    'patient|R'
    """
    if not isinstance(name, str):
        raise ValueError(
            f"entity_id: name must be a string, got {type(name).__name__} "
            f"({name!r})"
        )
    if not name:
        raise ValueError("entity_id: name must not be empty")
    if ID_SEPARATOR in name:
        raise ValueError(
            f"entity_id: name {name!r} contains the reserved separator "
            f"{ID_SEPARATOR!r}; this would make the id ambiguous on parse."
        )
    return f"{name}{ID_SEPARATOR}{kind_short(kind)}"


def parse_entity_id(eid: str) -> EntityId:
    """Parse a disambiguated entity id into its (name, kind) pieces.

    Args:
        eid: The disambiguated id (e.g. ``"patient|R"``).

    Returns:
        :class:`EntityId` with ``kind`` resolved to its long-form
        name (``"roles"`` or ``"traits"``).

    Raises:
        ValueError: If ``eid`` is not a string, lacks the separator,
            has more than one separator, has an empty name, or has an
            unknown short tag.

    >>> parse_entity_id("patient|R")
    EntityId(name='patient', kind='roles')
    >>> parse_entity_id("perfectionist|T")
    EntityId(name='perfectionist', kind='traits')
    """
    if not isinstance(eid, str):
        raise ValueError(
            f"parse_entity_id: expected str, got {type(eid).__name__} "
            f"({eid!r})"
        )
    parts = eid.split(ID_SEPARATOR)
    if len(parts) != 2:
        raise ValueError(
            f"parse_entity_id: {eid!r} is not a disambiguated id; "
            f"expected exactly one {ID_SEPARATOR!r} separator, found "
            f"{len(parts) - 1}."
        )
    name, short = parts
    if not name:
        raise ValueError(
            f"parse_entity_id: {eid!r} has an empty name component."
        )
    if short not in KIND_LONG:
        raise ValueError(
            f"parse_entity_id: {eid!r} has unknown kind tag {short!r}; "
            f"accepted: {sorted(KIND_LONG)}."
        )
    return EntityId(name=name, kind=KIND_LONG[short])


def is_entity_id(s: object) -> bool:
    """Return ``True`` iff ``s`` parses as a disambiguated entity id.

    Lets call sites probe a string without try/except scaffolding.
    A bare name like ``"patient"`` returns ``False``; ``"patient|R"``
    returns ``True``.
    """
    if not isinstance(s, str):
        return False
    try:
        parse_entity_id(s)
    except ValueError:
        return False
    return True


def display_label(eid_or_name: str) -> str:
    """Return the bare display name from either a disambiguated id or
    a bare name.

    Plots, axis labels, legends, console output etc. should call this
    on the dict key so the visible label is always the bare name --
    kind is encoded visually via colour/marker (see
    :mod:`assistant_axis.plot_palette`).

    Bare names are passed through unchanged, so this is safe to call
    on ids of unknown provenance.

    >>> display_label("patient|R")
    'patient'
    >>> display_label("teacher")  # already bare
    'teacher'
    """
    if is_entity_id(eid_or_name):
        return parse_entity_id(eid_or_name).name
    return eid_or_name


# Lowercase Latin letters with no NFKD decomposition, for ``normalize_to_file_name``.
_ASCII_FOLD = str.maketrans({
    "\u00df": "ss",  # ß
    "\u00f8": "o",   # ø
    "\u00e6": "ae",  # æ
    "\u0153": "oe",  # œ
    "\u0142": "l",   # ł
    "\u0111": "d",   # đ
    "\u00f0": "d",   # ð
    "\u00fe": "th",  # þ
})


def normalize_to_file_name(name: str) -> str:
    """Normalise a possibly display-form entity name to file-name form.

    Project convention is **file-name form everywhere except display
    sites** (see ``AGENT_NOTES.md`` ``File-name vs display-name
    convention``).  External inputs (user-supplied JSONs, CSVs, CLI
    args copy-pasted from a plot label) sometimes leak in the
    display form -- spaces, hyphens, capitals, apostrophes -- which
    silently misses on every multi-word entity in the corpus
    (~3 % of names: ``aligned_artificial_intelligence``,
    ``systems_thinker``, ``devils_advocate``, ...).

    This helper performs the minimal canonical conversion so
    boundary code can normalise once and treat the result like any
    other file-name key.  The transform is:

    1. Lowercase.
    2. Strip surrounding whitespace.
    3. Drop apostrophes (``devil's advocate`` -> ``devils advocate``)
       so the output matches the file-name form
       ``devils_advocate`` rather than the typo ``devil_s_advocate``.
    4. Collapse runs of whitespace and hyphens (including en / em
       dashes) into single underscores.
    5. Fold diacritics to ASCII (NFKD, drop combining marks; ``ß`` ->
       ``ss``), so ``Gemeinschaft (Tönnies)`` -> ``gemeinschaft_tonnies``
       and stems stay ASCII on every filesystem (macOS stores names
       NFD, Linux NFC; an accented stem would become two files across
       rsync).  The label keeps its diacritics; ``corpus_display_name``
       reads it back from the JSON.
    6. Drop parentheses and brackets, so the standard-derived label
       ``open (Big Five)`` -> ``open_big_five`` (Sep 2026 convention: the
       source is a parenthesised, capitalised suffix in the label and
       absent from the stem).

    The function is **idempotent**: file-name input passes through
    unchanged.

    Detection (did anything change?) is just
    ``normalize_to_file_name(s) != s``; callers should log a warning
    on conversion so the user is aware their input was massaged.

    Examples
    --------
    >>> normalize_to_file_name("aligned_artificial_intelligence")
    'aligned_artificial_intelligence'
    >>> normalize_to_file_name("aligned artificial intelligence")
    'aligned_artificial_intelligence'
    >>> normalize_to_file_name("Aligned Artificial Intelligence")
    'aligned_artificial_intelligence'
    >>> normalize_to_file_name("devil's advocate")
    'devils_advocate'
    >>> normalize_to_file_name("systems-thinker")
    'systems_thinker'
    >>> normalize_to_file_name("  multi   word   ")
    'multi_word'
    >>> normalize_to_file_name("Gemeinschaft (Tönnies)")
    'gemeinschaft_tonnies'
    >>> normalize_to_file_name("open (Big Five)")
    'open_big_five'
    >>> normalize_to_file_name("Inglehart\u2013Welzel")
    'inglehart_welzel'
    """
    import re
    import unicodedata
    s = name.strip().lower()
    s = s.replace("'", "").replace("\u2019", "")  # ASCII + curly apostrophe
    s = s.replace("\u2013", "-").replace("\u2014", "-")  # en / em dash -> hyphen
    # 6. Drop parentheses and brackets: the standard-derived label
    #    convention writes the source as a parenthesised suffix,
    #    ``open (Big Five)`` -> ``open_big_five``.
    for ch in "()[]":
        s = s.replace(ch, " ")
    s = s.strip()
    # 5. Fold to ASCII: NFKD splits base letter + combining mark, drop the
    #    marks, then map the few Latin letters that have no decomposition.
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.translate(_ASCII_FOLD)
    s = re.sub(r"[\s\-]+", "_", s)
    return s


def display_form_name(name: str) -> str:
    """Convert a file-name-form entity name to display form for
    rendering to humans **and LLMs**.

    Display sites are anywhere we render an entity name to a human
    or an LLM: plot labels, axis annotations, legends, console
    output, *and the body of LLM rubrics/prompts* (LLMs read e.g.
    ``aligned artificial intelligence`` more naturally than
    ``aligned_artificial_intelligence``, and judging accuracy
    measurably depends on how clearly the entity is described).
    See ``AGENT_NOTES.md`` ``File-name vs display-name convention``.

    The transform is the **minimal** one that round-trips through
    :func:`normalize_to_file_name` for plain underscored names:
    just ``_`` → space.  Capitalisation is preserved (the display
    side is where humans/LLMs decide on case).  Apostrophes are
    not re-introduced (the file-name form is lossy in that
    direction; we accept ``devils advocate`` as the displayed
    form, which is fine for both readers and LLMs).

    The function is **idempotent**: display-form input
    (``"aligned artificial intelligence"``, ``"systems-thinker"``,
    or any name without underscores) passes through unchanged.

    **Stable by design.**  This transform is deliberately mechanical
    (no corpus lookup), and its output must not change: it was the
    judge-prompt form up to rubric v3, :func:`judge_label` falls back to
    it for a name the corpus does not know (so such prompts stay
    byte-identical), and other code keys on it.  Since 2026-10-09 (W19,
    Roger) judge prompts no longer call it directly: they use
    :func:`judge_label` / :func:`judge_negative_label`.  For
    human-facing text -- plot labels, legends, console output -- use
    :func:`corpus_display_name`, which consults the trait/role JSONs
    and returns the stored display form (``systems-thinker``,
    ``devil's advocate``, ``traditional (Inglehart-Welzel)``).

    Disambiguated ids (``patient|R``) are passed through untouched
    so callers don't have to special-case them; downstream code
    typically applies :func:`display_label` to strip the suffix
    and *then* :func:`display_form_name` for the underscore swap.

    Examples
    --------
    >>> display_form_name("aligned_artificial_intelligence")
    'aligned artificial intelligence'
    >>> display_form_name("systems_thinker")
    'systems thinker'
    >>> display_form_name("devils_advocate")
    'devils advocate'
    >>> display_form_name("patient")
    'patient'
    >>> display_form_name("aligned artificial intelligence")
    'aligned artificial intelligence'
    >>> display_form_name("patient|R")
    'patient|R'
    """
    if "|" in name and name.rsplit("|", 1)[1] in ("R", "T"):
        return name
    return name.replace("_", " ")


# ---------------------------------------------------------------------------
# Corpus-backed display names (plots, legends, console) -- Sep 2026
# ---------------------------------------------------------------------------

#: Role stems whose display form cannot be recovered mechanically from the
#: file name.  Mirrors ``_ROLE_NAME_OVERRIDES`` in
#: ``data_analysis/regenerate_role_instructions.py`` (which must stay in
#: sync -- see ``test_entity_id.py::TestCorpusDisplayName``).
ROLE_DISPLAY_OVERRIDES: dict[str, str] = {
    "devils_advocate": "devil's advocate",
    # Two conceptions of an aligned AI (Roger, 2026-09-28).  Hyphenated,
    # not "aligned AI (virtuous)": the parenthesised form is reserved for
    # entries drawn from a named external set.
    "instrumentally_aligned_ai": "instrumentally-aligned AI",
    "virtue_aligned_ai": "virtue-aligned AI",
    # English hyphenates these; the mechanical form read "ex convict" in
    # prompts (2026-10-10, under the judge display form policy)
    "ex_convict": "ex-convict",
    "hunter_gatherer": "hunter-gatherer",
}


def default_data_dir() -> Path:
    """Locate the corpus ``data/`` directory.

    ``$ASSISTANT_AXIS_DATA_DIR`` wins when set (useful on RunPod or in
    tests); otherwise the repo-relative ``<repo>/data`` next to this
    package.
    """
    env = os.environ.get("ASSISTANT_AXIS_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "data"


@lru_cache(maxsize=8)
def _corpus_display_map(data_dir: str) -> dict[str, dict[str, str]]:
    """Build ``{"traits": {stem: display}, "roles": {stem: display}}``
    from the instruction JSONs under ``data_dir``.  Cached per
    directory; call :func:`clear_corpus_display_cache` after editing
    trait/role files in a long-lived process.

    Traits: the stored ``positive_label`` is the canonical display
    form (it carries hyphens, spaces, and any capitalised standard-name
    suffix that the file stem cannot).  Roles: an override if one
    exists, else the mechanical ``_`` -> space transform.  Missing
    directories or unreadable files are skipped, so the helper
    degrades to :func:`display_form_name` rather than raising.
    """
    root = Path(data_dir)
    out: dict[str, dict[str, str]] = {_KIND_TRAITS: {}, _KIND_ROLES: {}}
    tdir = root / "traits" / "instructions"
    if tdir.is_dir():
        for fp in sorted(tdir.glob("*.json")):
            try:
                blob = json.loads(fp.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            lbl = blob.get("positive_label") if isinstance(blob, dict) else None
            out[_KIND_TRAITS][fp.stem] = (
                str(lbl) if lbl else fp.stem.replace("_", " ")
            )
    rdir = root / "roles" / "instructions"
    if rdir.is_dir():
        for fp in sorted(rdir.glob("*.json")):
            out[_KIND_ROLES][fp.stem] = ROLE_DISPLAY_OVERRIDES.get(
                fp.stem, fp.stem.replace("_", " ")
            )
    return out


@lru_cache(maxsize=8)
def _corpus_negative_label_map(data_dir: str) -> dict[str, tuple[str, str]]:
    """``{trait stem: (negative_label, positive_label)}`` as stored in the
    trait JSONs under ``data_dir`` (only traits carry labels).  Cached per
    directory and cleared by :func:`clear_corpus_display_cache`."""
    out: dict[str, tuple[str, str]] = {}
    tdir = Path(data_dir) / _KIND_TRAITS / "instructions"
    if not tdir.is_dir():
        return out
    for fp in sorted(tdir.glob("*.json")):
        try:
            blob = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(blob, dict):
            continue
        neg = blob.get("negative_label")
        if isinstance(neg, str) and neg:
            pos = blob.get("positive_label")
            out[fp.stem] = (neg, str(pos) if pos else fp.stem.replace("_", " "))
    return out


def clear_corpus_display_cache() -> None:
    """Drop the cached stem -> display tables (after editing corpus JSONs)."""
    _corpus_display_map.cache_clear()
    _corpus_rename_map.cache_clear()
    _corpus_negative_label_map.cache_clear()


# ---------------------------------------------------------------------------
# Renamed entities: old stem -> current stem (Sep 2026)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _corpus_rename_map(data_dir: str) -> dict[str, dict[str, str]]:
    """Build ``{"traits": {old_stem: stem}, "roles": {old_stem: stem}}``
    from the ``renamed_from`` field of the instruction JSONs under
    ``data_dir`` (a dict with a ``stem`` key, a bare stem, or a list of
    either).  ``stem`` is the file that carries the field, so it always
    exists; an entity renamed twice needs the list form to keep its
    first stem.  Cached per directory and cleared by
    :func:`clear_corpus_display_cache`.
    """
    root = Path(data_dir)
    out: dict[str, dict[str, str]] = {_KIND_TRAITS: {}, _KIND_ROLES: {}}
    for kind in (_KIND_TRAITS, _KIND_ROLES):
        idir = root / kind / "instructions"
        if not idir.is_dir():
            continue
        for fp in sorted(idir.glob("*.json")):
            try:
                blob = json.loads(fp.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            renamed = blob.get("renamed_from") if isinstance(blob, dict) else None
            if not renamed:
                continue
            for item in renamed if isinstance(renamed, list) else [renamed]:
                old = item.get("stem") if isinstance(item, dict) else item
                if isinstance(old, str) and old and old != fp.stem:
                    out[kind][old] = fp.stem
    return out


def resolve_renamed_stem(
    stem: str,
    kind: str | None,
    *,
    data_dir: Path | str | None = None,
) -> str:
    """Return the stem the corpus uses **now** for an entity that an
    older file (a ``pair_list_*_v1.json``, a steering config) names by
    the stem it had when that file was written.

    A stem that still has an instruction file is returned unchanged,
    and so is one the corpus has no record of: the caller's own
    missing-file error is more useful than a guess.  Otherwise the
    result is the file whose ``renamed_from`` records the stem.
    ``kind=None`` is for a caller that does not know the kind (a
    steering config's pole names): a stem with a file of either kind is
    kept, and otherwise the trait renames are consulted before the role
    renames, the order in which such callers look the files up.

    This maps names for **reading the corpus** (descriptions,
    instructions).  Extracted vectors, response files and judge caches
    stay under the stem they were produced with; do not use the result
    to look those up, because a renamed entity's text may have changed
    with its name.
    """
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    kinds = (_KIND_TRAITS, _KIND_ROLES) if kind is None else (kind_long(kind),)
    if any((root / k / "instructions" / f"{stem}.json").exists() for k in kinds):
        return stem
    renames = _corpus_rename_map(str(root))
    for k in kinds:
        if stem in renames[k]:
            return renames[k][stem]
    return stem


def corpus_display_name(
    name_or_id: str,
    kind: str | None = None,
    *,
    data_dir: Path | str | None = None,
) -> str:
    """Return the corpus-recorded display form of an entity for
    **human-facing** text: plot labels, legends, axis annotations,
    console output.

    Resolution order:

    1. A disambiguated id (``"systems_thinker|T"``) supplies both the
       name and the kind.
    2. With a known kind, the stem is looked up in that kind's table
       (trait ``positive_label`` / role override or ``_`` -> space).
    3. With no kind, both tables are consulted.  The eleven
       collision names display identically in both, so they resolve
       cleanly; if a future collision ever displayed differently the
       helper falls back to the mechanical form rather than guess.
    4. Anything not in the corpus falls back to
       :func:`display_form_name`.

    The result always round-trips: ``normalize_to_file_name(
    corpus_display_name(stem)) == stem`` for every corpus entity
    (apostrophes and capitals are dropped by the normaliser, which is
    exactly the lossy direction the convention allows).

    Do **not** use the result as a dict key, cache key, or rho
    intersection operand, and do not splice it into an LLM prompt:
    prompts take :func:`judge_label`, which differs from this helper
    only in rewriting a standard's suffix (``careless (HEXACO)`` ->
    ``careless (from HEXACO)``).

    >>> corpus_display_name("systems_thinker")
    'systems-thinker'
    >>> corpus_display_name("systems_thinker|T")
    'systems-thinker'
    >>> corpus_display_name("devils_advocate", "roles")
    "devil's advocate"
    >>> corpus_display_name("aligned_artificial_intelligence")
    'aligned artificial intelligence'
    >>> corpus_display_name("obama_administration_health_team")  # not in corpus
    'obama administration health team'
    """
    if is_entity_id(name_or_id):
        parsed = parse_entity_id(name_or_id)
        name, kind = parsed.name, parsed.kind
    else:
        name = name_or_id
    if not name:
        return name
    root = str(Path(data_dir) if data_dir is not None else default_data_dir())
    table = _corpus_display_map(root)
    if kind is not None:
        hit = table[kind_long(kind)].get(name)
        return hit if hit is not None else display_form_name(name)
    t_hit = table[_KIND_TRAITS].get(name)
    r_hit = table[_KIND_ROLES].get(name)
    if t_hit is not None and r_hit is not None:
        return t_hit if t_hit == r_hit else display_form_name(name)
    if t_hit is not None:
        return t_hit
    if r_hit is not None:
        return r_hit
    return display_form_name(name)


# ---------------------------------------------------------------------------
# Judge display form: what every LLM prompt shows (W19, Roger 2026-10-09)
# ---------------------------------------------------------------------------
#
# Roger's rule (2026-10-09): anything that sends a label or a negative_label
# to an LLM judge (and, by his scope decision the same day, to the instruction
# generators and every other prompt) uses the most legible form: the display
# form (hyphens, spaces, capitals, diacritics, apostrophes kept), and, for a
# trait from an official instrument, the long form with "from":
# ``careless (HEXACO)`` -> ``careless (from HEXACO)``.  The stored label
# (``positive_label`` in the JSON) is unchanged; the rewrite happens only at
# the prompt.  Policy text: AGENT_NOTES § "Judge prompts show the judge
# display form".

#: How a standard's name reads after "from" in a judge prompt, keyed by the
#: suffix exactly as the labels write it (``<pole> (<Standard>)``).  The
#: default, for a suffix that is not listed, is the suffix itself
#: (``(from VALS)``); the defaults in use are listed anyway so that the table
#: is the whole record.  Sets that read better with the article or another
#: form are spelled out (chosen by the main agent at Roger's request,
#: 2026-10-09: "from Holland" reads as the country, "from Hall" as a
#: building, and in response mode the label is all that scopes the pole).
#: To add a standard: add its suffix here with the form it should take
#: (or leave it to the default), and extend
#: ``test_entity_id.py::TestJudgeLabel`` if it needs a pinned example.
STANDARD_SUFFIX_FORMS: dict[str, str] = {
    # default form ("from X"), listed explicitly
    "HEXACO": "HEXACO",
    "VALS": "VALS",
    "DISC": "DISC",
    "VARK": "VARK",
    "Tönnies": "Tönnies",
    # with the article
    "Big Five": "the Big Five",
    "MBTI": "the MBTI",
    "Enneagram": "the Enneagram",
    "BFAS": "the BFAS",
    "IPIP-NEO": "the IPIP-NEO",
    "Tarot": "the Tarot",
    "Dark Tetrad": "the Dark Tetrad",
    "Light Triad": "the Light Triad",
    "Inglehart-Welzel": "the Inglehart-Welzel map",
    # a person's name: name the framework, not the person
    "Holland": "Holland's RIASEC",
    "Bartle": "Bartle's player types",
    "Baumrind": "Baumrind's parenting styles",
    "Hall": "Edward Hall",
    "Kohlberg": "Kohlberg's stages",
    "Allport": "Allport's religious orientation",
    "Gelfand": "Gelfand's tight and loose cultures",
}

#: Recorded by callers that persist which label form a prompt used (the
#: instruction generators' ``generator`` field, the antonym check's result).
JUDGE_LABEL_FORM = "judge-display-v1"

# ``<head> (<suffix>)`` at the end of a label; the suffix has no parentheses.
_TRAILING_SUFFIX = None  # compiled lazily (keeps the module's import cheap)


def _suffix_match(label: str):
    global _TRAILING_SUFFIX
    if _TRAILING_SUFFIX is None:
        import re
        _TRAILING_SUFFIX = re.compile(r"^(?P<head>.*?\S)\s*\((?P<suffix>[^()]*)\)\s*$", re.S)
    return _TRAILING_SUFFIX.match(label)


def judge_form_of_label(label: str) -> str:
    """Apply the standard-suffix rewrite to a label string, and nothing else.

    A label ending in a parenthesised suffix that starts with a capital
    letter (a named standard, ``careless (HEXACO)``) has the suffix
    rewritten to ``(from <form>)``, where ``<form>`` is
    :data:`STANDARD_SUFFIX_FORMS`'s entry for it or, for an unlisted
    suffix, the suffix itself.  Anything else is returned unchanged: a
    lower-case suffix (``(tentative, PC10)``), a label with no suffix, the
    empty string.  The function is idempotent (``(from HEXACO)`` starts
    with a lower-case letter).

    This is the string-level half of :func:`judge_label`; use it directly
    for a label that is not (yet) a corpus file, such as a staged copy or a
    trait-gap candidate.

    >>> judge_form_of_label("careless (HEXACO)")
    'careless (from HEXACO)'
    >>> judge_form_of_label("Intellect (BFAS)")
    'Intellect (from the BFAS)'
    >>> judge_form_of_label("artistic (Holland)")
    "artistic (from Holland's RIASEC)"
    >>> judge_form_of_label("everyday sadism (Bourdieu)")
    'everyday sadism (from Bourdieu)'
    >>> judge_form_of_label("humorous/serious (tentative, PC10)")
    'humorous/serious (tentative, PC10)'
    >>> judge_form_of_label("systems-thinker")
    'systems-thinker'
    """
    import unicodedata
    if not label:
        return label
    m = _suffix_match(label)
    if m is None:
        return label
    suffix = m.group("suffix").strip()
    if not suffix or not suffix[0].isupper():
        return label
    key = unicodedata.normalize("NFC", suffix)
    form = STANDARD_SUFFIX_FORMS.get(key, suffix)
    return f"{m.group('head')} (from {form})"


def judge_label(
    name_or_id: str,
    kind: str | None = None,
    *,
    data_dir: Path | str | None = None,
) -> str:
    """The judge display form of an entity: what an LLM prompt shows.

    The name is resolved as :func:`corpus_display_name` resolves it (a
    trait's stored ``positive_label``; a role's
    :data:`ROLE_DISPLAY_OVERRIDES` entry, else ``_`` -> space; a
    disambiguated id supplies the kind; no kind consults both tables), and
    then :func:`judge_form_of_label` rewrites a standard's suffix.  A role
    with a known kind takes its override even when ``data_dir`` holds no
    file for it (roles store no label).  A name the corpus does not know
    falls back to :func:`display_form_name` (mechanical ``_`` -> space)
    before the suffix rewrite, so a free-text label such as
    ``careless (HEXACO)`` is rewritten too, and an unknown stem
    (``careless_hexaco`` with no file, a renamed pole's old stem) renders
    exactly as it did under the mechanical form.

    Capitals, hyphens, diacritics and apostrophes come from the stored
    label; the result never contains an underscore for a corpus entity.
    Never use it as a key: it is a prompt string.

    >>> judge_label("careless_hexaco")
    'careless (from HEXACO)'
    >>> judge_label("systems_thinker", "traits")
    'systems-thinker'
    >>> judge_label("devils_advocate|R")
    "devil's advocate"
    >>> judge_label("obama_administration_health_team")
    'obama administration health team'
    """
    if is_entity_id(name_or_id):
        parsed = parse_entity_id(name_or_id)
        name, kind = parsed.name, parsed.kind
    else:
        name = name_or_id
    if not name:
        return name
    if kind is not None and kind_long(kind) == _KIND_ROLES:
        root = str(Path(data_dir) if data_dir is not None else default_data_dir())
        hit = _corpus_display_map(root)[_KIND_ROLES].get(name)
        base = hit if hit is not None else ROLE_DISPLAY_OVERRIDES.get(name, display_form_name(name))
    else:
        base = corpus_display_name(name, kind, data_dir=data_dir)
    return judge_form_of_label(base)


def judge_form_of_negative_label(
    negative_label: str,
    positive_label: str | None = None,
    *,
    data_dir: Path | str | None = None,
) -> str:
    """The judge display form of a trait's stored ``negative_label``,
    given the strings (for a file that may not be in the corpus directory:
    a staged copy, an old commit's version).

    1. If the label names a corpus trait (by :func:`normalize_to_file_name`),
       that trait's :func:`judge_label`, so the partner reads exactly as it
       does when it is the entity itself.
    2. If it is the ``non-X`` placeholder of this trait
       (``"non-" + positive_label``, standards included:
       ``non-careless (HEXACO)``), ``"non-"`` plus the judge form of the
       trait's own label: ``non-careless (from HEXACO)``.
    3. Otherwise the stored string, with the suffix rewrite of
       :func:`judge_form_of_label` (a no-op unless it ends in a capitalised
       suffix, as a standard partner not yet seeded would).

    >>> judge_form_of_negative_label("non-careless (HEXACO)", "careless (HEXACO)")
    'non-careless (from HEXACO)'
    >>> judge_form_of_negative_label("conscientious (HEXACO)", "careless (HEXACO)")
    'conscientious (from HEXACO)'
    """
    if not negative_label:
        return negative_label
    root = str(Path(data_dir) if data_dir is not None else default_data_dir())
    stem = normalize_to_file_name(negative_label)
    if stem in _corpus_display_map(root)[_KIND_TRAITS]:
        return judge_label(stem, _KIND_TRAITS, data_dir=root)
    if positive_label and negative_label == f"non-{positive_label}":
        return "non-" + judge_form_of_label(positive_label)
    return judge_form_of_label(negative_label)


def judge_negative_label(
    trait_stem: str,
    *,
    data_dir: Path | str | None = None,
) -> str:
    """The judge display form of a corpus trait's ``negative_label``: the
    stored label of ``trait_stem`` resolved by
    :func:`judge_form_of_negative_label`.  Accepts ``stem`` or
    ``stem|T``.  Raises :class:`KeyError` for a stem with no trait file
    or no ``negative_label`` (roles have none).

    >>> judge_negative_label("careless_hexaco")
    'conscientious (from HEXACO)'
    """
    if is_entity_id(trait_stem):
        parsed = parse_entity_id(trait_stem)
        if parsed.kind != _KIND_TRAITS:
            raise KeyError(f"{trait_stem!r}: only traits have a negative_label")
        trait_stem = parsed.name
    root = str(Path(data_dir) if data_dir is not None else default_data_dir())
    stored = _corpus_negative_label_map(root).get(trait_stem)
    if stored is None:
        raise KeyError(f"{trait_stem!r}: no trait file with a negative_label under {root}")
    negative_label, positive_label = stored
    return judge_form_of_negative_label(negative_label, positive_label, data_dir=root)
