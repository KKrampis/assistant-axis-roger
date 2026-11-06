"""
Activation steering utilities for transformer models.

This module provides a context manager for intervening on model activations
during inference, supporting addition, ablation, mean ablation, capping, and
replacement operations, with optional multi-layer coefficient scaling.

Example:
    from assistant_axis import load_model, load_axis, ActivationSteering

    model, tokenizer = load_model("google/gemma-2-27b-it")
    axis = load_axis("outputs/gemma-2-27b/axis.pt")

    with ActivationSteering(model, steering_vectors=[axis[22]],
                           coefficients=[1.0], layer_indices=[22]):
        output = model.generate(...)
"""

import torch
from typing import Dict, Optional, Sequence, Tuple, Union, Iterable, List


class ActivationSteering:
    """
    Context manager for activation steering supporting:
    - Multiple feature directions simultaneously
    - Both addition and ablation interventions
    - Multiple layers
    - Per-direction coefficients

    For ablation: projects out the direction, then adds back with coefficient
    For addition: standard activation steering (add coeff * direction)
    """

    _POSSIBLE_LAYER_ATTRS: Iterable[str] = (
        "transformer.h",          # GPT-2/Neo, Bloom, etc.
        "encoder.layer",          # BERT/RoBERTa
        "model.layers",           # Llama/Mistral/Gemma 2/Qwen
        "language_model.layers",  # Gemma 3 (vision-language models)
        "gpt_neox.layers",        # GPT-NeoX
        "block",                  # Flan-T5
    )

    def __init__(
        self,
        model: torch.nn.Module,
        steering_vectors: Union[torch.Tensor, List[torch.Tensor], List[Sequence[float]]],
        *,
        coefficients: Union[float, List[float]] = 1.0,
        layer_indices: Union[int, List[int]] = -1,
        intervention_type: str = "addition",
        positions: str = "all",
        mean_activations: Union[torch.Tensor, List[torch.Tensor], List[Sequence[float]], None] = None,
        cap_thresholds: Union[float, List[float], None] = None,
        input_ids: Optional[torch.Tensor] = None,
        header_token_ids: Optional[List[int]] = None,
        vector_position_types: Optional[List[Union[str, int]]] = None,
        end_of_turn_id: Optional[int] = None,
        turn_headers: Optional[Dict[str, Tuple[List[int], List[int]]]] = None,
        multi_layer: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Args:
            model: The transformer model to steer
            steering_vectors: Either a single vector or list of vectors to use for steering
            coefficients: Either a single coefficient or list of coefficients (one per vector)
            layer_indices: Either a single layer index or list of layer indices to intervene at
            intervention_type: "addition" (standard steering), "ablation" (project out then add back),
                             "mean_ablation", "capping", or "replacement" (blend original activations
                             toward the provided vector: result = (1-coeff)*original + coeff*vector)
            positions: "all" (steer all positions), "last" (steer only last position),
                      "header_matched" (position-aware: each vector targets either generated text or specific header tokens),
                      "prefill_only" (steer all positions during prefill to bake into KV-cache, skip decode),
                      "system_only" (steer only system prompt content during prefill, skip decode),
                      "user_only" (steer only user prompt content during prefill, skip decode),
                      or "system_user_only" (steer system + user content during prefill, skip decode).
                      Region modes require input_ids and turn_headers to locate turn
                      boundaries via forward scanning. Content spans from the first token
                      after a role's header sequence to the last token before the next
                      role's header sequence.
            mean_activations: For mean_ablation only - replacement activations to add after projection
            cap_thresholds: For capping only - threshold values to cap projected activations at
            input_ids: For header_matched and region modes - tokenized prompt (1D or 2D tensor)
            header_token_ids: For header_matched mode only - token IDs of the assistant header
                sequence (from axis metadata)
            vector_position_types: For header_matched only - parallel to steering_vectors, each either
                "body" (apply to assistant response tokens) or an int token ID from header_token_ids
                (apply only to positions where that token appears as part of a matched header sequence)
            end_of_turn_id: For header_matched only - token ID marking end of a turn (e.g. <|im_end|>).
                Used to delimit assistant response body in multi-turn prompts.
                Falls back to header_token_ids[0] if not provided.
            turn_headers: For region modes only - dict mapping role names to
                (start_token_sequence, end_token_sequence) pairs. Required keys depend on
                the mode: system_only needs "system" and "user", user_only needs "user" and
                "assistant", system_user_only needs "system" and "assistant". Sequences are
                matched by forward scanning from the start of the token stream; only the
                first occurrence of each turn is used.
            multi_layer: Coefficient scaling for multi-layer steering. None = no scaling (default).
                "incremental": first layer at full coefficient; subsequent layers scaled by
                (norm - prev_norm) / norm, skipped if norm decreased. Distributes the total
                perturbation across layers proportionally to norm growth.
                "average": divide each coefficient by the number of layers in its group.
                In header_matched mode, groups are per-position-type; otherwise all vectors
                form one group.
            debug: Whether to print debugging information

        Note: For 1:1 mapping, steering_vectors, coefficients, and layer_indices must all have same length.
              steering_vectors[i] will be applied at layer_indices[i] with coefficients[i].
              If layer_indices has fewer elements than vectors, it will be broadcast to match.
        """
        self.model = model
        self.intervention_type = intervention_type.lower()
        self.positions = positions.lower()
        self.multi_layer = multi_layer.lower() if isinstance(multi_layer, str) else multi_layer
        self.debug = debug
        self._handles = []

        if self.multi_layer is not None and self.multi_layer not in {"incremental", "average"}:
            raise ValueError("multi_layer must be None, 'incremental', or 'average'")

        if self.intervention_type not in {"addition", "ablation", "mean_ablation", "capping", "replacement"}:
            raise ValueError("intervention_type must be 'addition', 'ablation', 'mean_ablation', 'capping', or 'replacement'")

        _region_modes = {"system_only", "user_only", "system_user_only"}
        if self.positions not in {"all", "last", "header_matched", "prefill_only"} | _region_modes:
            raise ValueError("positions must be 'all', 'last', 'header_matched', 'prefill_only', "
                             "'system_only', 'user_only', or 'system_user_only'")

        if self.positions == "header_matched":
            if input_ids is None:
                raise ValueError("input_ids required for positions='header_matched'")
            if header_token_ids is None:
                raise ValueError("header_token_ids required for positions='header_matched'")
            if vector_position_types is None:
                raise ValueError("vector_position_types required for positions='header_matched'")

        if self.positions in _region_modes:
            if input_ids is None:
                raise ValueError(f"input_ids required for positions='{self.positions}'")
            if turn_headers is None:
                raise ValueError(f"turn_headers required for positions='{self.positions}'")

        if self.intervention_type == "mean_ablation":
            if self.positions not in {"all", "header_matched"}:
                raise ValueError("mean_ablation only supports positions='all' or 'header_matched'")
            if mean_activations is None:
                raise ValueError("mean_activations is required for mean_ablation")

        self.steering_vectors = self._normalize_vectors(steering_vectors)
        self.coefficients = self._normalize_coefficients(coefficients)
        self.layer_indices = self._normalize_layers(layer_indices)
        self.mean_activations = self._normalize_mean_activations(mean_activations) if mean_activations is not None else None
        self.cap_thresholds = None

        if self.intervention_type == "capping":
            if cap_thresholds is None:
                raise ValueError("cap_thresholds is required when intervention_type='capping'")
            self.cap_thresholds = (
                [float(cap_thresholds)] if isinstance(cap_thresholds, (int, float))
                else [float(t) for t in cap_thresholds]
            )
            if len(self.cap_thresholds) != len(self.steering_vectors):
                raise ValueError(
                    f"Number of cap_thresholds ({len(self.cap_thresholds)}) must match number of vectors ({len(self.steering_vectors)})"
                )

        if self.intervention_type != "mean_ablation" and len(self.coefficients) != len(self.steering_vectors):
            raise ValueError(f"Number of coefficients ({len(self.coefficients)}) must match number of vectors ({len(self.steering_vectors)})")

        if self.mean_activations is not None and len(self.mean_activations) != len(self.steering_vectors):
            raise ValueError(f"Number of mean_activations ({len(self.mean_activations)}) must match number of vectors ({len(self.steering_vectors)})")

        if len(self.layer_indices) == 1 and len(self.steering_vectors) > 1:
            self.layer_indices = self.layer_indices * len(self.steering_vectors)
        elif len(self.layer_indices) != len(self.steering_vectors):
            raise ValueError(f"Number of layer_indices ({len(self.layer_indices)}) must match number of vectors ({len(self.steering_vectors)}) or be 1 (for broadcasting)")

        self.vectors_by_layer = {}
        for i, (vector, coeff, layer_idx) in enumerate(zip(self.steering_vectors, self.coefficients, self.layer_indices)):
            if layer_idx not in self.vectors_by_layer:
                self.vectors_by_layer[layer_idx] = []
            mean_act = self.mean_activations[i] if self.mean_activations is not None else None
            tau = self.cap_thresholds[i] if self.cap_thresholds is not None else None
            self.vectors_by_layer[layer_idx].append((vector, coeff, i, mean_act, tau))

        # Header-matched position setup (must precede multi_layer scaling
        # so vector_position_types is available for grouping)
        self.vector_position_types = None
        self._header_positions = {}
        self._layer_call_counts = {}

        if self.positions == "header_matched":
            self.vector_position_types = list(vector_position_types)
            if len(self.vector_position_types) != len(self.steering_vectors):
                raise ValueError(
                    f"vector_position_types length ({len(self.vector_position_types)}) "
                    f"must match steering_vectors ({len(self.steering_vectors)})")
            self.header_token_ids = list(header_token_ids)
            self.end_of_turn_id = end_of_turn_id
            header_id_set = set(self.header_token_ids)
            for i, pt in enumerate(self.vector_position_types):
                if pt != "body" and pt not in header_id_set:
                    raise ValueError(
                        f"vector_position_types[{i}] = {pt!r}: must be 'body' or "
                        f"a token ID from header_token_ids {self.header_token_ids}")
            self.set_input_ids(input_ids)

        if self.positions in _region_modes:
            if isinstance(input_ids, torch.Tensor):
                ids = input_ids.squeeze().tolist()
            else:
                ids = list(input_ids)
            self._prompt_length = len(ids)

            self._region_positions = self._compute_region_positions(
                ids, turn_headers, self.positions, self.debug)

        if self.multi_layer is not None:
            self._apply_multi_layer_scaling()

        if self.debug:
            print(f"[ActivationSteering] Initialized with:")
            print(f"  - {len(self.steering_vectors)} steering vectors")
            print(f"  - {len(set(self.layer_indices))} unique layers: {sorted(set(self.layer_indices))}")
            print(f"  - Intervention: {self.intervention_type}")
            if self.positions == "header_matched":
                print(f"  - Position types: {self.vector_position_types}")
                for tid, pos in self._header_positions.items():
                    print(f"  - Header token {tid}: positions {pos}")
                print(f"  - Body positions (previous responses): "
                      f"{len(self._prompt_body_positions)} tokens"
                      + (f" {self._prompt_body_positions}" if len(self._prompt_body_positions) < 30 else ""))
                if self.end_of_turn_id is not None:
                    print(f"  - End-of-turn token: {self.end_of_turn_id}")
            elif self.positions in ("system_only", "user_only", "system_user_only"):
                print(f"  - Region positions ({self.positions}): "
                      f"{len(self._region_positions)} of {self._prompt_length} tokens"
                      + (f" {self._region_positions}" if len(self._region_positions) < 30 else
                         f" [{self._region_positions[0]}..{self._region_positions[-1]}]"))

    def _normalize_vectors(self, steering_vectors):
        """Convert steering vectors to a list of tensors on the correct device/dtype."""
        p = next(self.model.parameters())

        if torch.is_tensor(steering_vectors):
            if steering_vectors.ndim == 1:
                vectors = [steering_vectors]
            elif steering_vectors.ndim == 2:
                vectors = [steering_vectors[i] for i in range(steering_vectors.shape[0])]
            else:
                raise ValueError("steering_vectors tensor must be 1D or 2D")
        else:
            vectors = steering_vectors

        result = []
        hidden_size = getattr(self.model.config, "hidden_size", None)

        for i, vec in enumerate(vectors):
            tensor_vec = torch.as_tensor(vec, dtype=p.dtype, device=p.device)
            if tensor_vec.ndim != 1:
                raise ValueError(f"Steering vector {i} must be 1-D, got shape {tensor_vec.shape}")
            if hidden_size and tensor_vec.numel() != hidden_size:
                raise ValueError(f"Vector {i} length {tensor_vec.numel()} != model hidden_size {hidden_size}")
            result.append(tensor_vec)

        return result

    def _normalize_coefficients(self, coefficients):
        """Convert coefficients to a list of floats."""
        if isinstance(coefficients, (int, float)):
            return [float(coefficients)]
        else:
            return [float(c) for c in coefficients]

    def _normalize_layers(self, layer_indices):
        """Convert layer indices to a list of ints."""
        if isinstance(layer_indices, int):
            return [layer_indices]
        else:
            return list(layer_indices)

    def _normalize_mean_activations(self, mean_activations):
        """Convert mean activations to a list of tensors on the correct device/dtype."""
        p = next(self.model.parameters())

        if torch.is_tensor(mean_activations):
            if mean_activations.ndim == 1:
                vectors = [mean_activations]
            elif mean_activations.ndim == 2:
                vectors = [mean_activations[i] for i in range(mean_activations.shape[0])]
            else:
                raise ValueError("mean_activations tensor must be 1D or 2D")
        else:
            vectors = mean_activations

        result = []
        hidden_size = getattr(self.model.config, "hidden_size", None)

        for i, vec in enumerate(vectors):
            tensor_vec = torch.as_tensor(vec, dtype=p.dtype, device=p.device)
            if tensor_vec.ndim != 1:
                raise ValueError(f"Mean activation {i} must be 1-D, got shape {tensor_vec.shape}")
            if hidden_size and tensor_vec.numel() != hidden_size:
                raise ValueError(f"Mean activation {i} length {tensor_vec.numel()} != model hidden_size {hidden_size}")
            result.append(tensor_vec)

        return result

    @staticmethod
    def _find_seq(ids: List[int], seq: List[int], start: int = 0) -> int:
        """Find first occurrence of token subsequence. Returns index or -1."""
        n = len(seq)
        for i in range(start, len(ids) - n + 1):
            if ids[i:i + n] == seq:
                return i
        return -1

    @staticmethod
    def _compute_region_positions(
        ids: List[int],
        turn_headers: Dict[str, Tuple[List[int], List[int]]],
        mode: str,
        debug: bool = False,
    ) -> List[int]:
        """Compute steered positions for region modes using forward-only scanning.

        Each turn_headers entry maps a role name ("system", "user", "assistant")
        to (start_seq, end_seq) where start_seq is the full header token sequence
        preceding content and end_seq is the end-of-turn token sequence after content.

        Returns a sorted list of token positions to steer.
        """
        import warnings
        find = ActivationSteering._find_seq

        if mode == "system_only":
            sys_start_seq, sys_end_seq = turn_headers["system"]
            usr_start_seq, _usr_end_seq = turn_headers["user"]

            sys_hdr = find(ids, sys_start_seq, 0)
            if sys_hdr < 0:
                warnings.warn("system_only: system header not found")
                return []
            content_start = sys_hdr + len(sys_start_seq)

            sys_end = find(ids, sys_end_seq, content_start)
            if sys_end < 0:
                warnings.warn("system_only: system end-of-turn not found")
                return []
            usr_hdr = find(ids, usr_start_seq, sys_end)
            if usr_hdr < 0:
                warnings.warn("system_only: user header not found after system turn")
                return []

            positions = list(range(content_start, usr_hdr))
            if debug:
                print(f"[region] system_only: content [{content_start}, {usr_hdr})")
            return positions

        elif mode == "user_only":
            usr_start_seq, usr_end_seq = turn_headers["user"]
            asst_start_seq, _asst_end_seq = turn_headers["assistant"]

            usr_hdr = find(ids, usr_start_seq, 0)
            if usr_hdr < 0:
                warnings.warn("user_only: user header not found")
                return []
            content_start = usr_hdr + len(usr_start_seq)

            usr_end = find(ids, usr_end_seq, content_start)
            if usr_end < 0:
                warnings.warn("user_only: user end-of-turn not found")
                return []
            asst_hdr = find(ids, asst_start_seq, usr_end)
            if asst_hdr < 0:
                warnings.warn("user_only: assistant header not found after user turn")
                return []

            positions = list(range(content_start, asst_hdr))
            if debug:
                print(f"[region] user_only: content [{content_start}, {asst_hdr})")
            return positions

        elif mode == "system_user_only":
            sys_start_seq, _sys_end_seq = turn_headers["system"]
            asst_start_seq, _asst_end_seq = turn_headers["assistant"]

            sys_hdr = find(ids, sys_start_seq, 0)
            if sys_hdr < 0:
                warnings.warn("system_user_only: system header not found")
                return []
            content_start = sys_hdr + len(sys_start_seq)

            asst_hdr = find(ids, asst_start_seq, content_start)
            if asst_hdr < 0:
                warnings.warn("system_user_only: assistant header not found")
                return []

            positions = list(range(content_start, asst_hdr))
            if debug:
                print(f"[region] system_user_only: content [{content_start}, {asst_hdr})")
            return positions

        return []

    def _apply_multi_layer_scaling(self):
        """Adjust coefficients for multi-layer steering modes.

        Groups vectors by position type (in header_matched mode) or treats
        all as one group, then scales coefficients within each group.
        """
        if self.positions == "header_matched" and self.vector_position_types:
            groups: dict = {}
            for i, pt in enumerate(self.vector_position_types):
                groups.setdefault(pt, []).append(i)
        else:
            groups = {"_all": list(range(len(self.steering_vectors)))}

        for indices in groups.values():
            if len(indices) <= 1:
                continue

            if self.multi_layer == "average":
                n = len(indices)
                for i in indices:
                    self.coefficients[i] /= n

            elif self.multi_layer == "incremental":
                sorted_indices = sorted(indices, key=lambda i: self.layer_indices[i])
                prev_norm = 0.0
                for i in sorted_indices:
                    norm = self.steering_vectors[i].norm().item()
                    if norm > prev_norm:
                        factor = (norm - prev_norm) / norm
                        self.coefficients[i] *= factor
                        prev_norm = norm
                    else:
                        self.coefficients[i] = 0.0

        # Rebuild vectors_by_layer with updated coefficients
        self.vectors_by_layer = {}
        for i, (vector, coeff, layer_idx) in enumerate(
                zip(self.steering_vectors, self.coefficients, self.layer_indices)):
            if layer_idx not in self.vectors_by_layer:
                self.vectors_by_layer[layer_idx] = []
            mean_act = self.mean_activations[i] if self.mean_activations is not None else None
            tau = self.cap_thresholds[i] if self.cap_thresholds is not None else None
            self.vectors_by_layer[layer_idx].append((vector, coeff, i, mean_act, tau))

        if self.debug:
            print(f"[ActivationSteering] multi_layer={self.multi_layer} scaling:")
            for i in range(len(self.steering_vectors)):
                norm = self.steering_vectors[i].norm().item()
                print(f"  vec {i} layer {self.layer_indices[i]}: "
                      f"norm={norm:.2f}, coeff={self.coefficients[i]:.4f}")

    def set_input_ids(self, input_ids: torch.Tensor):
        """Update input_ids and recompute header token positions.

        Finds positions of the full header_token_ids sequence in input_ids
        (not individual tokens), so e.g. a standalone '\\n' in the user
        message won't be mistaken for a header token.

        Call this between generations if the prompt changes.
        """
        if isinstance(input_ids, torch.Tensor):
            ids = input_ids.squeeze().tolist()
        else:
            ids = list(input_ids)

        self._prompt_length = len(ids)
        self._layer_call_counts = {}

        seq = self.header_token_ids
        seq_len = len(seq)

        self._header_positions = {tid: [] for tid in seq}
        header_matches = []

        for i in range(len(ids) - seq_len + 1):
            if ids[i:i + seq_len] == seq:
                for j, tid in enumerate(seq):
                    self._header_positions[tid].append(i + j)
                header_matches.append((i, i + seq_len))

        # Body positions = assistant response tokens in the prompt.
        # For each header match *except the last* (whose body will be
        # generated), body spans from the end of the header to the
        # end-of-turn token (inclusive).  Falls back to scanning for the
        # next header_token_ids[0] (exclusive) when end_of_turn_id is
        # not set — works for models where all turns share a prefix.
        self._prompt_body_positions: List[int] = []
        eot = self.end_of_turn_id
        for idx, (_, body_start) in enumerate(header_matches[:-1]):
            if eot is not None:
                body_end = len(ids)
                for k in range(body_start, len(ids)):
                    if ids[k] == eot:
                        body_end = k + 1  # inclusive
                        break
                else:
                    if self.debug:
                        print(f"[ActivationSteering] WARNING: end_of_turn_id {eot} "
                              f"not found after header match {idx}")
            else:
                body_end = len(ids)
                for k in range(body_start, len(ids)):
                    if ids[k] == seq[0]:
                        body_end = k  # exclusive
                        break
                else:
                    if self.debug:
                        print(f"[ActivationSteering] WARNING: header start token {seq[0]} "
                              f"not found after header match {idx}")
            self._prompt_body_positions.extend(range(body_start, body_end))

    def _resolve_positions(self, seq_len: int, vector_idx: int,
                           is_prefill: bool) -> List[int]:
        """Return sequence positions where vector_idx should be applied.

        Prefill (first forward pass):
          - header vectors → their matched token positions
          - body vectors → previous assistant response positions in the prompt

        KV-cache decode (seq_len < prompt_length, typically 1):
          - body vectors → all positions (these are newly generated tokens)
          - header vectors → nothing

        No-KV-cache decode (seq_len > prompt_length):
          - header vectors → their original prompt positions
          - body vectors → previous response positions + generated tokens
        """
        pos_type = self.vector_position_types[vector_idx]

        if pos_type == "body":
            if is_prefill:
                # Previous assistant responses in the prompt
                return [p for p in self._prompt_body_positions if p < seq_len]
            if seq_len <= self._prompt_length:
                # KV-cache decode: all positions are newly generated tokens
                return list(range(seq_len))
            # No-KV-cache decode: previous responses + newly generated tokens
            return (self._prompt_body_positions
                    + list(range(self._prompt_length, seq_len)))
        else:
            if not is_prefill and seq_len < self._prompt_length:
                # KV-cache decode: no header tokens being generated
                return []
            return [p for p in self._header_positions.get(pos_type, []) if p < seq_len]

    def _locate_layer_list(self):
        """Find the layer list in the model."""
        for path in self._POSSIBLE_LAYER_ATTRS:
            cur = self.model
            for part in path.split("."):
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    break
            else:
                if hasattr(cur, "__getitem__"):
                    return cur, path

        raise ValueError(
            "Could not find layer list on the model. "
            "Add the attribute name to _POSSIBLE_LAYER_ATTRS."
        )

    def _get_layer_module(self, layer_idx):
        """Get the module for a specific layer index."""
        layer_list, path = self._locate_layer_list()

        if not (-len(layer_list) <= layer_idx < len(layer_list)):
            raise IndexError(f"layer_idx {layer_idx} out of range for {len(layer_list)} layers")

        if self.debug:
            print(f"[ActivationSteering] Located layer {path}[{layer_idx}]")

        return layer_list[layer_idx]

    def _create_hook_fn(self, layer_idx):
        """Create a hook function for a specific layer."""
        def hook_fn(module, ins, out):
            return self._apply_layer_interventions(out, layer_idx)
        return hook_fn

    def _apply_layer_interventions(self, activations, layer_idx):
        """Apply only the interventions assigned to this specific layer."""
        if layer_idx not in self.vectors_by_layer:
            return activations

        if torch.is_tensor(activations):
            tensor_out = activations
            was_tuple = False
        elif isinstance(activations, (tuple, list)):
            if not torch.is_tensor(activations[0]):
                return activations
            tensor_out = activations[0]
            was_tuple = True
        else:
            return activations

        if self.positions == "header_matched":
            modified_out = self._apply_header_matched(tensor_out, layer_idx)
        elif self.positions == "prefill_only":
            self._layer_call_counts[layer_idx] = (
                self._layer_call_counts.get(layer_idx, 0) + 1)
            if self._layer_call_counts[layer_idx] > 1:
                return activations
            modified_out = tensor_out
            n_vecs = len(self.vectors_by_layer[layer_idx])
            for vector, coeff, vector_idx, mean_act, tau in self.vectors_by_layer[layer_idx]:
                eff_coeff = coeff / n_vecs
                if self.intervention_type == "addition":
                    modified_out = modified_out + eff_coeff * vector.to(modified_out.device)
                elif self.intervention_type == "replacement":
                    modified_out = self._apply_replacement(modified_out, vector, eff_coeff)
                elif self.intervention_type == "ablation":
                    modified_out = self._apply_ablation(modified_out, vector, eff_coeff)
                elif self.intervention_type == "mean_ablation":
                    modified_out = self._apply_mean_ablation(modified_out, vector, mean_act)
                elif self.intervention_type == "capping":
                    modified_out = self._apply_cap(modified_out, vector, tau)

                if self.debug:
                    v = vector / (vector.norm() + 1e-8)
                    pre = torch.einsum('bld,d->bl', tensor_out, v)
                    post = torch.einsum('bld,d->bl', modified_out, v)
                    print(f"[ActivationSteering] Layer {layer_idx}, vec {vector_idx} "
                        f"(prefill, eff_coeff={eff_coeff:.4f}): "
                        f"pre mean={pre.mean():.3f} | post mean={post.mean():.3f}")
        elif self.positions in ("system_only", "user_only", "system_user_only"):
            self._layer_call_counts[layer_idx] = (
                self._layer_call_counts.get(layer_idx, 0) + 1)
            if self._layer_call_counts[layer_idx] > 1:
                return activations
            modified_out = tensor_out.clone()
            region_pos = self._region_positions
            n_vecs = len(self.vectors_by_layer[layer_idx])
            for vector, coeff, vector_idx, mean_act, tau in self.vectors_by_layer[layer_idx]:
                eff_coeff = coeff / n_vecs
                v = vector.to(modified_out.device)
                if self.intervention_type == "addition":
                    modified_out[:, region_pos, :] += eff_coeff * v
                elif self.intervention_type == "ablation":
                    v_norm = v / (v.norm() + 1e-8)
                    sliced = modified_out[:, region_pos, :]
                    proj = torch.einsum('bpd,d->bp', sliced, v_norm)
                    modified_out[:, region_pos, :] = (
                        sliced - torch.einsum('bp,d->bpd', proj, v_norm) + eff_coeff * v)
                elif self.intervention_type == "replacement":
                    modified_out[:, region_pos, :] = (
                        (1.0 - eff_coeff) * modified_out[:, region_pos, :] + eff_coeff * v)
                elif self.intervention_type == "capping":
                    v_norm = v / (v.norm() + 1e-8)
                    sliced = modified_out[:, region_pos, :]
                    proj = torch.einsum('bpd,d->bp', sliced, v_norm)
                    excess = (proj - tau).clamp(min=0.0)
                    modified_out[:, region_pos, :] = (
                        sliced - torch.einsum('bp,d->bpd', excess, v_norm))
                elif self.intervention_type == "mean_ablation":
                    v_norm = v / (v.norm() + 1e-8)
                    ma = mean_act.to(modified_out.device)
                    sliced = modified_out[:, region_pos, :]
                    proj = torch.einsum('bpd,d->bp', sliced, v_norm)
                    modified_out[:, region_pos, :] = (
                        sliced - torch.einsum('bp,d->bpd', proj, v_norm) + ma)

                if self.debug:
                    v_dbg = v / (v.norm() + 1e-8)
                    pre = torch.einsum('bpd,d->bp', tensor_out[:, region_pos, :], v_dbg)
                    post = torch.einsum('bpd,d->bp', modified_out[:, region_pos, :], v_dbg)
                    print(f"[ActivationSteering] Layer {layer_idx}, vec {vector_idx} "
                        f"({self.positions}, {len(region_pos)} pos, eff_coeff={eff_coeff:.4f}): "
                        f"pre mean={pre.mean():.3f} | post mean={post.mean():.3f}")
        else:
            modified_out = tensor_out
            n_vecs = len(self.vectors_by_layer[layer_idx])
            for vector, coeff, vector_idx, mean_act, tau in self.vectors_by_layer[layer_idx]:
                eff_coeff = coeff / n_vecs
                if self.intervention_type == "addition":
                    modified_out = self._apply_addition(modified_out, vector, eff_coeff)
                elif self.intervention_type == "ablation":
                    modified_out = self._apply_ablation(modified_out, vector, eff_coeff)
                elif self.intervention_type == "mean_ablation":
                    modified_out = self._apply_mean_ablation(modified_out, vector, mean_act)
                elif self.intervention_type == "capping":
                    modified_out = self._apply_cap(modified_out, vector, tau)
                elif self.intervention_type == "replacement":
                    modified_out = self._apply_replacement(modified_out, vector, eff_coeff)

                if self.debug:
                    v = vector / (vector.norm() + 1e-8)
                    pre = torch.einsum('bld,d->bl', tensor_out, v)
                    post = torch.einsum('bld,d->bl', modified_out, v)
                    print(f"[ActivationSteering] Layer {layer_idx}, vec {vector_idx} "
                        f"(eff_coeff={eff_coeff:.4f}): "
                        f"pre mean={pre.mean():.3f} | post mean={post.mean():.3f}")

        if was_tuple:
            return (modified_out, *activations[1:])
        else:
            return modified_out

    def _apply_header_matched(self, tensor_out, layer_idx):
        """Apply interventions with per-vector position matching."""
        modified_out = tensor_out.clone()
        seq_len = modified_out.shape[1]

        self._layer_call_counts[layer_idx] = (
            self._layer_call_counts.get(layer_idx, 0) + 1)
        is_prefill = self._layer_call_counts[layer_idx] == 1

        for vector, coeff, vector_idx, mean_act, tau in self.vectors_by_layer[layer_idx]:
            positions = self._resolve_positions(seq_len, vector_idx, is_prefill)
            if not positions:
                continue

            pos_idx = positions
            v = vector.to(modified_out.device)

            if self.intervention_type == "addition":
                modified_out[:, pos_idx, :] += coeff * v

            elif self.intervention_type == "ablation":
                v_norm = v / (v.norm() + 1e-8)
                sliced = modified_out[:, pos_idx, :]
                proj = torch.einsum('bpd,d->bp', sliced, v_norm)
                modified_out[:, pos_idx, :] = (
                    sliced - torch.einsum('bp,d->bpd', proj, v_norm) + coeff * v)

            elif self.intervention_type == "mean_ablation":
                v_norm = v / (v.norm() + 1e-8)
                ma = mean_act.to(modified_out.device)
                sliced = modified_out[:, pos_idx, :]
                proj = torch.einsum('bpd,d->bp', sliced, v_norm)
                modified_out[:, pos_idx, :] = (
                    sliced - torch.einsum('bp,d->bpd', proj, v_norm) + ma)

            elif self.intervention_type == "capping":
                v_norm = v / (v.norm() + 1e-8)
                sliced = modified_out[:, pos_idx, :]
                proj = torch.einsum('bpd,d->bp', sliced, v_norm)
                excess = (proj - tau).clamp(min=0.0)
                modified_out[:, pos_idx, :] = (
                    sliced - torch.einsum('bp,d->bpd', excess, v_norm))

            elif self.intervention_type == "replacement":
                modified_out[:, pos_idx, :] = (
                    (1.0 - coeff) * modified_out[:, pos_idx, :] + coeff * v)

            if self.debug:
                v_norm_dbg = v / (v.norm() + 1e-8)
                pre = torch.einsum('bpd,d->bp', tensor_out[:, pos_idx, :], v_norm_dbg)
                post = torch.einsum('bpd,d->bp', modified_out[:, pos_idx, :], v_norm_dbg)
                pos_type = self.vector_position_types[vector_idx]
                print(f"[ActivationSteering] Layer {layer_idx}, vec {vector_idx} "
                      f"({pos_type}, {len(positions)} pos): "
                      f"pre mean={pre.mean():.3f} | post mean={post.mean():.3f}")

        return modified_out

    def _apply_addition(self, activations, vector, coeff):
        """Apply standard activation addition: x + coeff * vector"""
        vector = vector.to(activations.device)
        steer = coeff * vector

        if self.positions == "all":
            return activations + steer
        else:
            result = activations.clone()
            result[:, -1, :] += steer
            return result

    def _apply_ablation(self, activations, vector, coeff):
        """Apply ablation: project out direction, then add back with coefficient."""
        vector = vector.to(activations.device)
        vector_norm = vector / (vector.norm() + 1e-8)

        if self.positions == "all":
            projections = torch.einsum('bld,d->bl', activations, vector_norm)
            projected_out = activations - torch.einsum('bl,d->bld', projections, vector_norm)
            return projected_out + coeff * vector
        else:
            result = activations.clone()
            last_pos = result[:, -1, :]
            projection = torch.einsum('bd,d->b', last_pos, vector_norm)
            projected_out = last_pos - torch.einsum('b,d->bd', projection, vector_norm)
            result[:, -1, :] = projected_out + coeff * vector
            return result

    def _apply_mean_ablation(self, activations, vector, mean_activation):
        """Apply mean ablation: project out direction, then add mean activation."""
        vector = vector.to(activations.device)
        mean_activation = mean_activation.to(activations.device)
        vector_norm = vector / (vector.norm() + 1e-8)

        projections = torch.einsum('bld,d->bl', activations, vector_norm)
        projected_out = activations - torch.einsum('bl,d->bld', projections, vector_norm)
        return projected_out + mean_activation

    def _apply_cap(self, activations, vector, tau):
        """Apply capping: cap projection onto vector at threshold tau."""
        vector = vector.to(activations.device)
        v = vector / (vector.norm() + 1e-8)

        if self.positions == "all":
            proj = torch.einsum('bld,d->bl', activations, v)
            excess = (proj - tau).clamp(min=0.0)
            return activations - torch.einsum('bl,d->bld', excess, v)
        else:
            result = activations.clone()
            last = result[:, -1, :]
            proj = torch.einsum('bd,d->b', last, v)
            excess = (proj - tau).clamp(min=0.0)
            result[:, -1, :] = last - torch.einsum('b,d->bd', excess, v)
            return result

    def _apply_replacement(self, activations, vector, coeff):
        """Blend activations toward a target vector: (1-coeff)*original + coeff*vector."""
        vector = vector.to(activations.device)

        if self.positions == "all":
            return (1.0 - coeff) * activations + coeff * vector
        else:
            result = activations.clone()
            result[:, -1, :] = (1.0 - coeff) * result[:, -1, :] + coeff * vector
            return result

    def __enter__(self):
        """Register hooks on all unique layers."""
        for layer_idx in self.vectors_by_layer.keys():
            layer_module = self._get_layer_module(layer_idx)
            hook_fn = self._create_hook_fn(layer_idx)
            handle = layer_module.register_forward_hook(hook_fn)
            self._handles.append(handle)

        if self.debug:
            print(f"[ActivationSteering] Registered {len(self._handles)} hooks")

        return self

    def __exit__(self, *exc):
        """Remove all hooks."""
        self.remove()

    def remove(self):
        """Remove all registered hooks."""
        for handle in self._handles:
            if handle:
                handle.remove()
        self._handles = []

        if self.debug:
            print("[ActivationSteering] Removed all hooks")


def create_feature_ablation_steerer(
    model: torch.nn.Module,
    feature_directions: List[torch.Tensor],
    layer_indices: Union[int, List[int]],
    ablation_coefficients: Union[float, List[float]] = 0.0,
    **kwargs
) -> ActivationSteering:
    """
    Create a steerer for feature ablation.

    Args:
        model: The model to steer
        feature_directions: List of feature direction vectors to ablate
        layer_indices: Layer(s) to intervene at
        ablation_coefficients: Coefficient(s) for ablation. 0.0 = pure ablation, 1.0 = no change
    """
    return ActivationSteering(
        model=model,
        steering_vectors=feature_directions,
        coefficients=ablation_coefficients,
        layer_indices=layer_indices,
        intervention_type="ablation",
        **kwargs
    )


def create_multi_feature_steerer(
    model: torch.nn.Module,
    feature_directions: List[torch.Tensor],
    coefficients: List[float],
    layer_indices: Union[int, List[int]],
    intervention_type: str = "addition",
    **kwargs
) -> ActivationSteering:
    """
    Create a steerer for multiple features.

    Args:
        model: The model to steer
        feature_directions: List of feature direction vectors
        coefficients: List of coefficients (one per feature)
        layer_indices: Layer(s) to intervene at
        intervention_type: "addition" or "ablation"
    """
    return ActivationSteering(
        model=model,
        steering_vectors=feature_directions,
        coefficients=coefficients,
        layer_indices=layer_indices,
        intervention_type=intervention_type,
        **kwargs
    )


def create_mean_ablation_steerer(
    model: torch.nn.Module,
    feature_directions: List[torch.Tensor],
    mean_activations: List[torch.Tensor],
    layer_indices: Union[int, List[int]],
    **kwargs
) -> ActivationSteering:
    """
    Create a steerer for mean ablation.

    Args:
        model: The model to steer
        feature_directions: List of feature direction vectors to ablate
        mean_activations: List of mean activation vectors to replace with
        layer_indices: Layer(s) to intervene at
    """
    return ActivationSteering(
        model=model,
        steering_vectors=feature_directions,
        layer_indices=layer_indices,
        intervention_type="mean_ablation",
        mean_activations=mean_activations,
        coefficients=[0.0] * len(feature_directions),
        positions="all",
        **kwargs
    )


def load_capping_config(config_path: str) -> dict:
    """
    Load a capping config file.

    Args:
        config_path: Path to the .pt config file

    Returns:
        Dict with 'vectors' and 'experiments' keys
    """
    return torch.load(config_path, map_location='cpu', weights_only=False)


def build_capping_steerer(
    model: torch.nn.Module,
    capping_config: dict,
    experiment_id: Union[str, int],
    **kwargs
) -> ActivationSteering:
    """
    Build an ActivationSteering context manager from a capping config experiment.

    Args:
        model: The model to steer
        capping_config: Dict loaded from a capping config file (via load_capping_config)
        experiment_id: Either the experiment ID string (e.g. "layers_46:54-p0.25")
                      or the experiment index
        **kwargs: Additional arguments passed to ActivationSteering

    Returns:
        ActivationSteering context manager configured for capping

    Example:
        from huggingface_hub import hf_hub_download
        from assistant_axis import load_capping_config, build_capping_steerer

        config_path = hf_hub_download(
            repo_id="lu-christina/assistant-axis-vectors",
            filename="qwen-3-32b/capping_config.pt",
            repo_type="dataset"
        )
        config = load_capping_config(config_path)

        with build_capping_steerer(model, config, "layers_46:54-p0.25"):
            response = model.generate(...)
    """
    # Find the experiment
    experiment = None
    if isinstance(experiment_id, int):
        experiment = capping_config['experiments'][experiment_id]
    else:
        for exp in capping_config['experiments']:
            if exp['id'] == experiment_id:
                experiment = exp
                break

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_id}' not found in config")

    # Collect capping interventions
    vectors = []
    cap_thresholds = []
    layer_indices = []

    for intervention in experiment['interventions']:
        if 'cap' not in intervention:
            continue

        vector_name = intervention['vector']
        cap_value = float(intervention['cap'])

        vec_data = capping_config['vectors'][vector_name]
        layer_idx = vec_data['layer']
        vector = vec_data['vector'].to(dtype=torch.float32)

        vectors.append(vector)
        cap_thresholds.append(cap_value)
        layer_indices.append(layer_idx)

    if not vectors:
        raise ValueError(f"No capping interventions found in experiment '{experiment_id}'")

    vectors_tensor = torch.stack(vectors)

    return ActivationSteering(
        model=model,
        steering_vectors=vectors_tensor,
        layer_indices=layer_indices,
        intervention_type="capping",
        cap_thresholds=cap_thresholds,
        coefficients=[0.0] * len(vectors),
        positions="all",
        **kwargs
    )
