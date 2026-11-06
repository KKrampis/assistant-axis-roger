# The Assistant Axis

**Situating and Stabilizing the Default Persona of Language Models**

<p align="center">
  <img src="img/assistant_axis.png" width="800" alt="Persona drift trajectory showing activation projections along the Assistant Axis over a conversation">
</p>

<p align="center"><em>(Left) Vectors corresponding to character archetypes are computed by measuring model activations on responses when the model is system-prompted to act as that character. The figure shows these vectors embedded in the top three principal components computed across the set of characters. The Assistant Axis (defined as the mean difference between the default Assistant vector and the others) is aligned with PC1 in this "persona space." This occurs across different models; results from Llama 3.3 70B are pictured here. Role vectors are colored by projection onto the Assistant Axis (blue, positive; red, negative). (Right) In a conversation between Llama 3.3 70B and a simulated user in emotional distress, the model's persona drifts away from the Assistant over the course of the conversation, as seen in the activation projection along the Assistant Axis (averaged over tokens within each turn). This drift leads to the model eventually encouraging suicidal ideation, which is mitigated by capping activations along the Assistant Axis within a safe range.</em></p>

## Overview

Large language models default to a "helpful Assistant" persona cultivated during post-training. However, this persona can *drift* during conversations—particularly in emotionally charged or meta-reflective contexts—leading to harmful or bizarre behavior.

The **Assistant Axis** is a direction in activation space that captures how "Assistant-like" a model's current persona is. It can be used to:

- **Monitor** persona drift in real-time by projecting activations onto the axis
- **Steer** model behavior toward or away from the Assistant persona
- **Mitigate** persona-based jailbreaks through activation capping

This repository provides tools for computing, analyzing, and steering with the Assistant Axis. It also contains full transcripts from conversations mentioned in the paper.

See the full [paper here](https://arxiv.org/abs/2601.10387). A demo for chatting with activation capped Llama 3.3 70B is available on [Neuronpedia](https://neuronpedia.org/assistant-axis).

Pre-computed axes and persona vectors for Gemma 2 27B, Qwen 3 32B, and Llama 3.3 70B are available on [HuggingFace](https://huggingface.co/datasets/lu-christina/assistant-axis-vectors). Qwen 3 32B and Llama 3.3 70B also have activation capping steering settings available.

## Installation

```bash
git clone https://github.com/safety-research/assistant-axis.git
cd assistant-axis

# Install with uv (recommended)
uv sync
```

## Understanding the Axis

The Assistant Axis is computed as:

```
axis = mean(default_activations) - mean(role_activations)
```

Where:
- `default_activations`: Activations from neutral system prompts ("You are an AI Assistant")
- `role_activations`: Activations from responses fully embodying character roles (score=3 from judge)

The axis points **toward default Assistant behavior**:
- **Higher projections**: More Assistant-like (transparent, grounded, flexible)
- **Lower projection**: Drifting away from the Assistant (enigmatic, subversive, dramatic)

## Notebooks

Interactive notebooks for analysis and experimentation. See [`notebooks/README.md`](notebooks/README.md) for details.

- **PCA analysis** of role vectors and variance explained
- **Assistant Axis visualization** with cosine similarity to roles
- **Steering and activation capping** on arbitrary prompts
- **Transcript projection** to visualize persona trajectories

## Computing the Axis

To compute the axis for a new model, run the 5-step pipeline:

1. **Generate** model responses for 275 character roles
2. **Extract** mean response activations
3. **Score** role adherence with an LLM judge
4. **Compute** per-role vectors from high-scoring responses
5. **Aggregate** into the final axis

See [`pipeline/README.md`](pipeline/README.md) for detailed instructions.

## Trait Instructions

Trait data lives in `data/traits/instructions/<trait>.json`. Each file contains `positive_label`, `negative_label`, `description`, `instruction` (pos/neg pairs), `questions`, and `eval_prompt`.

### Adding new traits

Create a stub JSON with just the metadata fields:

```json
{
  "positive_label": "merciful",
  "negative_label": "non-merciful",
  "description": "This means showing compassion and clemency in the face of wrongdoing..."
}
```

Then run the regeneration script to generate `instruction`, `questions`, and `eval_prompt` via Claude:

```bash
uv run python data_analysis/regenerate_trait_instructions.py --traits merciful
```

### Regenerating existing traits

```bash
# Regenerate specific traits (skips if already has expected counts)
uv run python data_analysis/regenerate_trait_instructions.py --traits arrogant stoic

# Force overwrite existing instructions and questions
uv run python data_analysis/regenerate_trait_instructions.py --traits stoic --force

# Regenerate only instructions (keep existing questions)
uv run python data_analysis/regenerate_trait_instructions.py --traits stoic --instructions-only --force

# Preview what would happen without making changes
uv run python data_analysis/regenerate_trait_instructions.py --all --dry-run
```

Requires `ANTHROPIC_API_KEY` in environment or `.env`. Each trait costs 1 API call (combined prompt for instructions + questions + eval_prompt).

## Transcripts

Example conversations from the paper are available in [`transcripts/`](transcripts/README.md):

- **Case studies** showing persona drift and activation capping mitigation (jailbreaks, delusion reinforcement, self-harm scenarios)
- **Example conversations** from simulated multi-turn conversations across domains (coding, writing, therapy, philosophy)

## Quick Start

### Load a pre-computed axis

```python
from huggingface_hub import hf_hub_download
from assistant_axis import load_model, load_axis

# Load model
model, tokenizer = load_model("google/gemma-2-27b-it")

# Download pre-computed axis
axis_path = hf_hub_download(
    repo_id="lu-christina/assistant-axis-vectors",
    filename="gemma-2-27b/assistant_axis.pt",
    repo_type="dataset"
)
axis = load_axis(axis_path)
```

### Steer model outputs

Steering vectors live in **raw model-activation space** — the native
frame the model operates in.  Pass `axis[layer]` through
`ActivationSteering` as-is; do **not** whiten or shear it first.
(The whitening / soft-shear regimes that show up elsewhere in this
repo are analysis-side conveniences that sharpen judge-projection ρ;
they don't pass through to the steering side.  See
[`AGENT_NOTES.md` § Whitening / soft-shear defaults](AGENT_NOTES.md)
for the analysis defaults and the scope-of-applicability note.)

```python
from assistant_axis import ActivationSteering, generate_response

# Positive coefficient = more Assistant-like
# Negative coefficient = pushing away from the Assistant
with ActivationSteering(
    model,
    steering_vectors=[axis[22]],
    coefficients=[1.0],
    layer_indices=[22]
):
    response = generate_response(model, tokenizer, conversation)
```

### Monitor persona drift

```python
from assistant_axis import extract_response_activations, project

# Extract activations from a conversation
activations = extract_response_activations(model, tokenizer, [conversation])

# Project onto axis (higher = more assistant-like)
projection = project(activations[0], axis, layer=22)
print(f"Projection: {projection:.4f}")
```

### Mitigate persona drift with activation capping

Activation capping is a more targeted intervention that prevents activations from exceeding a threshold along specific directions. Pre-computed capping configs are available for Qwen 3 32B and Llama 3.3 70B.

```python
from huggingface_hub import hf_hub_download
from assistant_axis import get_config, load_capping_config, build_capping_steerer

# Get model config (includes recommended capping experiment)
config = get_config("Qwen/Qwen3-32B")

# Download and load capping config
capping_config_path = hf_hub_download(
    repo_id="lu-christina/assistant-axis-vectors",
    filename=config["capping_config"],  # "qwen-3-32b/capping_config.pt"
    repo_type="dataset"
)
capping_config = load_capping_config(capping_config_path)

# Apply capping during generation
with build_capping_steerer(model, capping_config, config["capping_experiment"]):
    response = model.generate(...)
```

## API Reference

### Models

```python
from assistant_axis import load_model, get_config, MODEL_CONFIGS

model, tokenizer = load_model("google/gemma-2-27b-it")
config = get_config("google/gemma-2-27b-it")  # {"target_layer": 22, ...}
```

### Axis

```python
from assistant_axis import compute_axis, load_axis, save_axis, project

axis = compute_axis(role_activations, default_activations)
projection = project(activations, axis, layer=22)
```

### Steering

```python
from assistant_axis import ActivationSteering

with ActivationSteering(
    model,
    steering_vectors=[axis[22]],
    coefficients=[1.0],       # Positive = more assistant-like
    layer_indices=[22],
    intervention_type="addition"
):
    output = model.generate(...)
```

### Activation Capping

```python
from assistant_axis import load_capping_config, build_capping_steerer

# Load pre-computed capping config
capping_config = load_capping_config("path/to/capping_config.pt")

# Build steerer from a specific experiment
# Experiments define which layers to cap and threshold values
with build_capping_steerer(model, capping_config, "layers_46:54-p0.25"):
    output = model.generate(...)

# List available experiments
for exp in capping_config['experiments']:
    print(exp['id'])
```

### PCA

```python
from assistant_axis import compute_pca, plot_variance_explained

result, variance, n_comp, pca, scaler = compute_pca(activations, layer=22)
fig = plot_variance_explained(variance)
```

### Plot provenance metadata

Every plot generated in this repo embeds a small PNG-text-chunk
provenance block via [`assistant_axis.png_metadata`](assistant_axis/plot_metadata.py).
The block records `Title`, `Author`, `Software` (canonical UNIX
command, repo-relative, prefixed with `uv run python ...`),
`Creation Time`, and the git short SHA — so when you rediscover an
old plot in a slide deck or notebook and ask "how was this made?",
the answer is in the file:

```bash
exiftool roger/canonical_angles_layer_sweep.png | grep -E 'Title|Author|Software|Creation Time|Source'
# Title:           Goal vs Non-goal canonical angles by transformer layer
# Author:          Roger Dearnaley
# Software:        uv run python -m results_analysis.canonical_angles.plots.layer_sweep --output roger/canonical_angles_layer_sweep.png
# Creation Time:   2026-04-27 17:46:59 +0100
# Source:          git 3ed6291+dirty
```

(or in Python: `Image.open(p).info`).

```python
from assistant_axis import png_metadata
fig.savefig(out_path, dpi=150, bbox_inches="tight",
            metadata=png_metadata(title="My plot title"))
```

For ad-hoc exploratory plots (one-off `/tmp/foo.py` scripts that
won't end up tracked), embed the script source too so the plot stays
reproducible without the chat transcript:

```python
from pathlib import Path
fig.savefig(out_path, metadata=png_metadata(
    title="My plot title",
    source_text=Path(__file__).read_text(),
))
```

This adds `Source Code` and `Source Code SHA256` chunks (hundreds of
bytes to a few KB; negligible PNG bloat).  Recovery is one line:

```python
from PIL import Image
print(Image.open("plot.png").info["Source Code"])
```

Embedded source isn't a complete archeological record (pip-package
versions live in `uv.lock`; co-imported repo files live at the git
SHA), but it's a strong "where did this come from?" record that
combined with the SHA gets you almost all the way back. See
[`AGENT_NOTES.md`](AGENT_NOTES.md) for the full convention and
[`assistant_axis/plot_metadata.py`](assistant_axis/plot_metadata.py)
for the helper.

### Data provenance

The plot-provenance block above answers "how was this made?". A
parallel layered system answers "are this plot's *inputs* still
current?" — i.e., has any upstream dataset, judge cache, or
intermediate JSON drifted since the artifact was rendered? Layers:

- **Dataset manifests** — each dataset under `runpod_workspace/`
  carries a `MANIFEST.json` with subtree-granular `summary_sha256`
  fingerprints. Regenerate after dataset changes:

  ```bash
  uv run python tools/regenerate_dataset_manifest.py \
      --dataset 'runpod_workspace/qwen/qwen-3-32b Roger 8slot'
  ```

- **Raw vs derived layout** — pipeline outputs live under
  `combinations/vectors/`; everything *post-pipeline* (marginals,
  aggregates, axes, legacy centroids) lives under `vectors/derived/`.
  See [`audits/post_pipeline_derived_layout.md`](audits/post_pipeline_derived_layout.md)
  for the locked layout.

- **Writer pattern** — analysis scripts that emit JSON wrap the
  payload in a `_provenance` envelope via
  [`assistant_axis.json_metadata`](assistant_axis/plot_metadata.py)
  and pass the same `inputs=[InputSpec, ...]` list to
  `png_metadata(...)` for any plot output.

- **Reader pattern** — consumers expose `--cache-policy {strict,
  warn, rebuild, off}` and load upstream caches via
  `load_validated_json(...)`, which re-derives current fingerprints
  and reports drift.

- **Audit tools** — answer "what's stale?" repo-wide:

  ```bash
  uv run python tools/audit_pngs.py   --status stale
  uv run python tools/audit_caches.py --status stale
  ```

  Cache audit also propagates staleness *transitively* through
  inter-cache file dependencies.

Full reference (including `InputSpec` kinds, status taxonomy, and
the migrated-script roster) lives in
[`AGENT_NOTES.md` § End-to-End Data Provenance](AGENT_NOTES.md).

## Agent collaboration guide

[`AGENT_NOTES.md`](AGENT_NOTES.md) is the canonical working guide for AI
agents on this repo (Cursor and Claude Code alike).  `CLAUDE.md` and
everything under `.claude/` are generated from it by
[`tools/sync_agent_notes.py`](tools/sync_agent_notes.py); after editing
`AGENT_NOTES.md`, run `uv run python tools/sync_agent_notes.py` (a Claude
Code session-start hook also runs it).

## Models from the Paper

| Model | Target Layer | Best Activation Capping Setting |
|-------|-------------|------------------------|
| `google/gemma-2-27b-it` | 22 | - |
| `Qwen/Qwen3-32B` | 32 | `layers_46:54-p0.25` |
| `meta-llama/Llama-3.3-70B-Instruct` | 40 | `layers_56:72-p0.25` |

Other models will auto-infer configuration based on architecture. We recommend turning reasoning off.

## Citation

```bibtex
@misc{lu2026assistant,
      title={The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models}, 
      author={Christina Lu and Jack Gallagher and Jonathan Michala and Kyle Fish and Jack Lindsey},
      year={2026},
      eprint={2601.10387},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.10387}, 
}
```

## License

MIT
