#!/usr/bin/env bash
# Run infer_axis_description.py for each (PC × style) cell that's missing
# spec.json.  Prepares input_scores.json from post_shear_projection.json
# (just adds type info), then invokes the describer.
#
# Pre-req: launch_judge_runs.py with the new PCs has already populated
# axis_postshear.pt and post_shear_projection.json in each cell.

set -euo pipefail

cd "$(dirname "$0")/.."
SWEEP_DIR="roger/pc_axis_describer_sweep"
DATA_DIR="runpod_workspace/qwen/qwen-3-32b Roger 8slot"

PCS="${1:-3 6 12}"
STYLES="${2:-glossary inline}"

for pc in $PCS; do
    pc3=$(printf '%03d' "$pc")
    for style in $STYLES; do
        cell="$SWEEP_DIR/pc${pc3}_${style}"
        if [ -f "$cell/spec.json" ]; then
            echo "[skip] $cell/spec.json already exists"
            continue
        fi
        if [ ! -f "$cell/post_shear_projection.json" ]; then
            echo "[error] $cell/post_shear_projection.json missing — run launch_judge_runs.py first"
            exit 1
        fi
        echo "[$cell] preparing input_scores.json..."
        # Convert {entity_id: score} → [{name, type, score}, ...].
        # post_shear_projection.json keys are disambiguated entity_ids
        # (e.g. "patient|R" / "patient|T") since trait/role disambiguation
        # landed in May 2026; parse the kind from the suffix.
        uv run python -c "
import json, sys
from assistant_axis.entity_id import parse_entity_id, kind_short
proj = json.load(open('$cell/post_shear_projection.json'))
out = []
bad = []
for k, s in proj.items():
    try:
        eid = parse_entity_id(k)
    except ValueError:
        bad.append(k)
        continue
    out.append({'name': eid.name, 'type': kind_short(eid.kind),
                'score': float(s)})
if bad:
    print(f'WARN: {len(bad)} non-entity_id keys in post_shear_projection.json: '
          f'{bad[:5]} (regenerate via launch_judge_runs.py --setup_only)',
          file=sys.stderr)
json.dump(out, open('$cell/input_scores.json', 'w'), indent=2)
print(f'wrote {len(out)} entries')
"
        echo "[$cell] running infer_axis_description.py (Opus, ~30-60s)..."
        uv run python -m results_analysis.infer_axis_description \
            --input "$cell/input_scores.json" \
            --output "$cell/spec.json" \
            --style "$style" \
            2>&1 | tail -8
        echo
    done
done

echo "All described.  Now run:"
echo "  uv run python -m results_analysis.pc_round_trip.launch_judge_runs --pcs 3,6,12 --skip_setup"
