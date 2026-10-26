"""Refusal pilot for Show-o2 understanding; Arditi et al., sections 2.3, 2.4, C.1.

Reuse the main adapter and upstream decoder. No weight edits or image generation.
"""
from contextlib import contextmanager

import pandas as pd
import torch
from tqdm.auto import tqdm


@contextmanager
def hooks(pre=(), post=()):
    handles = []
    try:
        for module, fn in pre:
            handles.append(module.register_forward_pre_hook(fn))
        for module, fn in post:
            handles.append(module.register_forward_hook(fn))
        yield
    finally:
        for handle in handles:
            handle.remove()


class UnderstandingInputs:
    """Encode one fixed image once; vary only the question for build/validation.

    Prefix, modality offsets, tokenizer and mask come from the main adapter.
    Prepared examples live on CPU, and run one at a time (no padding).
    """

    @torch.inference_mode()
    def __init__(self, adapter, image_path):
        self.adapter = adapter
        embeddings, _, alignment = adapter.prepare_understanding(image_path, "")
        self.prefix = embeddings[:, :alignment["image_span"][1]].detach()
        self.modality_positions = torch.tensor(
            alignment["modality_positions"], device=adapter.device
        )
        self.suffix_ids = adapter.tokenizer(
            "\n<|im_start|>assistant\n", add_special_tokens=False
        )["input_ids"]
        self.positions = list(range(-len(self.suffix_ids), 0))

    @torch.inference_mode()
    def prepare(self, question):
        adapter = self.adapter
        ids = ([adapter.token_ids["eoi_id"]]
               + adapter.tokenizer(question, add_special_tokens=False)["input_ids"]
               + self.suffix_ids)
        with adapter.autocast():
            text = adapter.model.showo.model.embed_tokens(
                torch.tensor([ids], device=adapter.device)
            )
            embeddings = torch.cat([self.prefix, text], dim=1)
        return embeddings.cpu()

    def mask(self, embeddings):
        adapter = self.adapter
        return adapter.mask(
            1, embeddings.shape[1], self.modality_positions, adapter.device,
            inverted=True,
        ).to(adapter.dtype)

    @torch.inference_mode()
    def logits(self, embeddings):
        adapter = self.adapter
        with adapter.autocast():
            result = adapter.model.showo(
                inputs_embeds=embeddings.to(adapter.device),
                attention_mask=self.mask(embeddings), use_cache=False,
                num_logits_to_keep=1, return_dict=True,
            )
        return result.logits[0, -1].double().cpu()


@torch.inference_mode()
def mean_activations(inputs, prompts):
    """Mean residual INPUT of every decoder block at each suffix position."""
    adapter = inputs.adapter
    means = torch.zeros(
        len(inputs.positions), len(adapter.layers),
        adapter.model.showo.config.hidden_size, dtype=torch.float64,
    )

    def capture(layer):
        def save(module, args):
            means[:, layer] += args[0][0, inputs.positions].detach().double().cpu()
        return save

    pre = [(block, capture(layer)) for layer, block in enumerate(adapter.layers)]
    with hooks(pre=pre):
        for prompt in tqdm(prompts, desc="Mean activations"):
            inputs.logits(inputs.prepare(prompt))
    return means / len(prompts)


def addition_hooks(adapter, direction, layer):
    def add(module, args):
        h = args[0]
        return (h + direction.to(h), *args[1:])
    return [(adapter.layers[layer], add)], []


def remove_component(h, direction):
    unit = (direction / direction.norm()).to(h)
    return h - (h @ unit).unsqueeze(-1) * unit


def ablation_hooks(adapter, direction):
    """Project block inputs and attention/MLP writes at ALL layers/positions.

    Attention returns a tuple; MLP returns a tensor in upstream Qwen2.
    Hooks stay active throughout prefill AND every autoregressive forward.
    """
    def ablate_input(module, args):
        return (remove_component(args[0], direction), *args[1:])

    def ablate_attention(module, args, output):
        return (remove_component(output[0], direction), *output[1:])

    def ablate_mlp(module, args, output):
        return remove_component(output, direction)

    pre = [(block, ablate_input) for block in adapter.layers]
    post = [(block.self_attn, ablate_attention) for block in adapter.layers]
    post += [(block.mlp, ablate_mlp) for block in adapter.layers]
    return pre, post


def refusal_score(logits, refusal_ids):
    """First-answer-token log odds, as in the reference selection code."""
    probability = logits.double().softmax(-1)[..., refusal_ids].sum(-1)
    return (probability + 1e-8).log() - (1 - probability + 1e-8).log()


def kl_score(baseline, ablated):
    """KL(baseline || ablated), last position; reference epsilon = 1e-6."""
    p = baseline.double().softmax(-1)
    q = ablated.double().softmax(-1)
    return (p * ((p + 1e-6).log() - (q + 1e-6).log())).sum(-1).mean()


@torch.inference_mode()
def evaluate_candidates(inputs, candidates, harmful, harmless, refusal_ids):
    """Score only the first floor(0.8 L) source layers; interventions use all L."""
    harmful_inputs = [inputs.prepare(prompt) for prompt in harmful]
    harmless_inputs = [inputs.prepare(prompt) for prompt in harmless]
    baseline_harmful = torch.stack([inputs.logits(x) for x in harmful_inputs])
    baseline_harmless = torch.stack([inputs.logits(x) for x in harmless_inputs])
    print("Baseline harmful log odds:", refusal_score(baseline_harmful, refusal_ids).mean().item())
    print("Baseline harmless log odds:", refusal_score(baseline_harmless, refusal_ids).mean().item())
    rows = []
    for position_index, position in enumerate(inputs.positions):
        for layer in tqdm(range(int(0.8 * len(inputs.adapter.layers))), desc=f"Position {position}"):
            direction = candidates[position_index, layer]
            pre, post = ablation_hooks(inputs.adapter, direction)
            with hooks(pre, post):
                ablated_harmful = torch.stack([inputs.logits(x) for x in harmful_inputs])
                ablated_harmless = torch.stack([inputs.logits(x) for x in harmless_inputs])
            pre, post = addition_hooks(inputs.adapter, direction, layer)
            with hooks(pre, post):
                added_harmless = torch.stack([inputs.logits(x) for x in harmless_inputs])
            rows.append(dict(
                position=position, layer=layer,
                bypass_score=refusal_score(ablated_harmful, refusal_ids).mean().item(),
                induce_score=refusal_score(added_harmless, refusal_ids).mean().item(),
                kl_score=kl_score(baseline_harmless, ablated_harmless).item(),
            ))
    scores = pd.DataFrame(rows)
    scores["passes_filters"] = (scores.induce_score > 0) & (scores.kl_score < 0.1)
    return scores.sort_values("bypass_score").reset_index(drop=True)


@torch.inference_mode()
def generate_response(adapter, image, question, pre=(), post=(), max_new_tokens=100, seed=42):
    torch.manual_seed(seed)
    embeddings, mask, _ = adapter.prepare_understanding(image, question)
    with hooks(pre, post):
        return adapter.answer(embeddings, mask, max_new_tokens)
