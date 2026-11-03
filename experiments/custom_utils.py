"""Build text steering directions and intervene during Show-o2 denoising."""
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Literal

import torch

from show_o2_space.capture import Capture

Selection = int | Sequence[int] | slice | Literal["all", "first", "last"]
Reduction = Literal["none", "sum", "mean", "average"] | Callable
DirectionMethod = Literal["difference", "svd"] | Callable


def _indices(selection: Selection, length: int) -> list[int]:
    if isinstance(selection, str):
        return {"all": list(range(length)), "first": [0], "last": [length - 1]}[selection]
    if isinstance(selection, slice):
        indices = list(range(length))[selection]
    elif isinstance(selection, int):
        indices = [selection]
    else:
        indices = list(selection)
    if not indices or any(index < -length or index >= length for index in indices):
        raise ValueError(f"Select at least one index in [-{length}, {length - 1}].")
    return list(dict.fromkeys(index % length for index in indices))


def _reduction_name(reduction: Reduction) -> str:
    return reduction.__name__ if callable(reduction) else reduction


REDUCERS = {"sum": torch.sum, "mean": torch.mean, "average": torch.mean}


def estimate_direction(
    positive: torch.Tensor, negative: torch.Tensor, method: DirectionMethod = "difference",
) -> torch.Tensor:
    """Estimate one direction from [examples, hidden] matrices at a fixed layer."""
    if positive.ndim != 2 or negative.ndim != 2:
        raise ValueError("Provide one pooled hidden state per example: [examples, hidden].")
    contrast = positive.mean(dim=0) - negative.mean(dim=0)
    if method == "difference":
        return contrast
    states = torch.cat([positive, negative], dim=0)
    if callable(method):
        direction = method(states)
    elif method == "svd":
        if states.square().sum() == 0:
            raise ValueError("SVD requires nonzero hidden states.")
        _, _, vh = torch.linalg.svd(states, full_matrices=False)
        direction = vh[0]
        orientation = torch.dot(direction, contrast)
        largest = direction[direction.abs().argmax()]
        direction = direction * torch.where(orientation == 0, largest, orientation).sign()
    else:
        raise ValueError(f"Unknown direction method: {method}")
    if direction.shape != contrast.shape:
        raise ValueError("A direction method must return one [hidden] vector.")
    return direction


def reduce_states(states: torch.Tensor, reduction: Reduction, dim: int = 0) -> torch.Tensor:
    """Reduce one observation axis while preserving hidden coordinates."""
    if reduction == "none":
        return states.clone()
    dim %= states.ndim
    if dim == states.ndim - 1:
        raise ValueError("Reduce an observation axis, not the hidden dimension.")
    reducer = reduction if callable(reduction) else REDUCERS[reduction]
    result = reducer(states, dim=dim)
    expected = states.shape[:dim] + states.shape[dim + 1:]
    if result.shape != expected:
        raise ValueError(f"Reducer returned {tuple(result.shape)}; expected {tuple(expected)}.")
    return result


def prepare_pair(adapter, pair_pos: str, pair_neg: str, *, alignment="none") -> dict:
    """Tokenize raw text with a shared BOS and optional right-side space padding."""
    if alignment not in ("none", "right_pad"):
        raise ValueError("alignment must be 'none' or 'right_pad'.")
    ids = {
        side: [adapter.token_ids["bos_id"]]
        + adapter.tokenizer(text, add_special_tokens=False)["input_ids"]
        for side, text in (("positive", pair_pos), ("negative", pair_neg))
    }
    padding = dict(positive=0, negative=0)
    if alignment == "right_pad":
        space_id, = adapter.tokenizer(" ", add_special_tokens=False)["input_ids"]
        length = max(map(len, ids.values()))
        for side, tokens in ids.items():
            padding[side] = length - len(tokens)
            tokens.extend([space_id] * padding[side])
    return dict(
        token_ids=ids, padding=padding, alignment=alignment,
        pair_pos=pair_pos, pair_neg=pair_neg,
    )


@torch.no_grad()
def capture_pair(adapter, pair: dict, *, layers: Selection = "all") -> dict:
    """Capture CPU FP32 block outputs in two text-only understanding forwards."""
    indices = _indices(layers, len(adapter.layers))
    activations = {}
    for side, ids in pair["token_ids"].items():
        input_ids = torch.tensor([ids], device=adapter.device)
        length = len(ids)
        mask = torch.full(
            (length, length), float("-inf"), device=adapter.device, dtype=adapter.dtype,
        ).triu(diagonal=1)[None, None]
        with Capture(adapter.layers, indices) as captured, adapter.autocast():
            adapter.model.showo.model(
                input_ids=input_ids, attention_mask=mask, use_cache=False,
                output_hidden_states=False, return_dict=True,
            )
        activations[side] = {layer: state[0].float() for layer, state in captured.states.items()}
    return dict(pair=pair, layers=indices, **activations)


def build_steering_vector(
    captured: dict, *, tokens: Selection = "last",
    token_reduction: Reduction = "mean", layer_reduction: Reduction = "none",
    direction_method: DirectionMethod = "difference",
) -> dict:
    """Pool selected tokens per example, estimate directions, then optionally combine layers."""
    positions = {
        side: _indices(tokens, len(captured["pair"]["token_ids"][side]))
        for side in ("positive", "negative")
    }
    if len(positions["positive"]) != len(positions["negative"]):
        raise ValueError("Tokenwise subtraction needs equal selected lengths; use right_pad.")
    positive, negative, differences, vectors = {}, {}, {}, {}
    for layer in captured["layers"]:
        positive[layer] = captured["positive"][layer][positions["positive"]]
        negative[layer] = captured["negative"][layer][positions["negative"]]
        differences[layer] = positive[layer] - negative[layer]
        pooled_pos = reduce_states(positive[layer], token_reduction)
        pooled_neg = reduce_states(negative[layer], token_reduction)
        if direction_method == "difference":
            vectors[layer] = pooled_pos - pooled_neg
        else:
            if pooled_pos.ndim != 1 or pooled_neg.ndim != 1:
                raise ValueError("Reduce tokens before estimating an example-level direction.")
            vectors[layer] = estimate_direction(
                pooled_pos.unsqueeze(0), pooled_neg.unsqueeze(0), direction_method,
            )
    combined = None
    if layer_reduction != "none":
        combined = reduce_states(torch.stack(list(vectors.values())), layer_reduction)
    metadata = dict(
        **captured["pair"], layers=captured["layers"], selected_positions=positions,
        token_reduction=_reduction_name(token_reduction),
        layer_reduction=_reduction_name(layer_reduction),
        direction_method=_reduction_name(direction_method),
        capture="decoder block output before final norm", difference="positive - negative",
    )
    return dict(
        vectors=vectors, combined=combined, positive=positive, negative=negative,
        differences=differences, metadata=metadata,
    )


def make_steering_vector(
    adapter, *, pair_pos: str, pair_neg: str, alignment="none",
    tokens: Selection = "last", layers: Selection = "all",
    token_reduction: Reduction = "mean", layer_reduction: Reduction = "none",
    direction_method: DirectionMethod = "difference",
) -> dict:
    """Construct a direction using the same steps exposed by the notebook."""
    pair = prepare_pair(adapter, pair_pos, pair_neg, alignment=alignment)
    captured = capture_pair(adapter, pair, layers=layers)
    return build_steering_vector(
        captured, tokens=tokens, token_reduction=token_reduction,
        layer_reduction=layer_reduction,
        direction_method=direction_method,
    )


def add_direction(hidden: torch.Tensor, vector: torch.Tensor, *, alpha: float) -> torch.Tensor:
    """Add a scaled direction to selected hidden states."""
    return hidden + alpha * vector


def edit_component(hidden: torch.Tensor, vector: torch.Tensor, *, alpha: float) -> torch.Tensor:
    """Scale the existing component along the unit direction by 1 + alpha."""
    unit = torch.nn.functional.normalize(vector, dim=-1)
    return hidden + alpha * (hidden @ unit).unsqueeze(-1) * unit


OPERATIONS = {"add": add_direction, "component": edit_component}


@contextmanager
def count_denoising_evaluations(adapter):
    """Count root model forwards during one adapter.generate call."""
    trace = {"evaluations": 0}

    def count(module, args):
        trace["evaluations"] += 1

    handle = adapter.model.register_forward_pre_hook(count)
    try:
        yield trace
    finally:
        handle.remove()


@contextmanager
def steer_denoising(
    adapter, *, vectors: dict, total_evaluations: int,
    layers: Selection = "all", iterations: Selection = "all",
    text: dict | None = None, visual: dict | None = None,
):
    """Edit conditional-row block outputs using Show-o2's current modality offsets."""
    selected_layers = _indices(layers, len(adapter.layers))
    selected_iterations = set(_indices(iterations, total_evaluations))
    rules = {part: rule for part, rule in (("text", text), ("visual", visual)) if rule is not None}
    trace = {"evaluations": 0, "injections": [], "preview": {}}
    positions = {}
    handles = []

    def locate(module, args, kwargs):
        offset, length = kwargs["modality_positions"][0, 0].tolist()
        positions["text"] = list(range(1, offset - 1))
        positions["visual"] = list(range(offset + 1, offset + length))
        trace["evaluations"] += 1

    def hook(layer):
        def apply(module, args, output):
            iteration = trace["evaluations"] - 1
            if iteration not in selected_iterations:
                return output
            hidden = output[0].clone()
            vector = vectors[layer].to(hidden)
            for part, rule in rules.items():
                indices = _indices(rule["tokens"], len(positions[part]))
                target = [positions[part][i] for i in indices]
                before = hidden[0, target, :]
                operation = rule["operation"]
                operation = operation if callable(operation) else OPERATIONS[operation]
                after = operation(before, vector, alpha=rule["alpha"])
                hidden[0, target, :] = after
                trace["injections"].append(dict(
                    iteration=iteration, layer=layer, part=part, tokens=len(target),
                ))
                key = f"{layer}:{part}"
                if key not in trace["preview"]:
                    trace["preview"][key] = dict(
                        iteration=iteration, position=target[0],
                        before=before[0, :6].detach().float().cpu(),
                        after=after[0, :6].detach().float().cpu(),
                    )
            return (hidden, *output[1:])
        return apply

    try:
        handles.append(adapter.model.register_forward_pre_hook(locate, with_kwargs=True))
        for layer in selected_layers:
            handles.append(adapter.layers[layer].register_forward_hook(hook(layer)))
        yield trace
    finally:
        for handle in handles:
            handle.remove()
