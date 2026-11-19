"""Capture conditional text and visual states, keeping denoising time."""
from contextlib import contextmanager

import torch


def denoising_steps(selection, total):
    """Return one-based model evaluations; lists select an equal-weight mean."""
    names = {"middle": [(total + 1) // 2], "last": [total],
             "all": list(range(1, total + 1))}
    if isinstance(selection, str):
        return names[selection] if selection in names else [int(selection)]
    return [selection] if isinstance(selection, int) else sorted(set(selection))


@contextmanager
def capture_generation(adapter, component="backbone"):
    """Capture backbone outputs or flow-head inputs, pooling each modality."""
    blocks = adapter.layers if component == "backbone" else adapter.model.diffusion_head_a
    states = {layer: [] for layer in range(len(blocks))}
    text_states = {layer: [] for layer in states}
    text_first = {layer: [] for layer in states}
    text_last = {layer: [] for layer in states}
    trace = dict(states=states, text_states=text_states, text_first=text_first,
                 text_last=text_last, times=[])
    positions = {}

    def locate(module, args, kwargs):
        offset, length = kwargs["modality_positions"][0, 0].tolist()
        positions["visual"] = slice(offset + 1, offset + length)
        tokens = args[0][0]
        text = torch.arange(1, offset - 1, device=tokens.device)
        special = torch.tensor(adapter.tokenizer.all_special_ids, device=tokens.device)
        positions["text"] = text[~torch.isin(tokens[text], special)]
        trace["times"].append(float(kwargs["t"].flatten()[0]))

    def record(layer, hidden):
        states[layer].append(hidden[0, positions["visual"]].detach().float().mean(0).cpu())
        text_states[layer].append(hidden[0, positions["text"]].detach().float().mean(0).cpu())
        text_first[layer].append(hidden[0, positions["text"][0]].detach().float().cpu())
        text_last[layer].append(hidden[0, positions["text"][-1]].detach().float().cpu())

    def hook(layer):
        def capture(module, args, output):
            record(layer, output[0])
        return capture

    def input_hook(layer):
        def capture(module, args, kwargs):
            record(layer, args[0] if args else kwargs["hidden_states"])
        return capture

    handles = [adapter.model.register_forward_pre_hook(locate, with_kwargs=True)]
    for layer, block in enumerate(blocks):
        handle = (block.register_forward_hook(hook(layer)) if component == "backbone"
                  else block.register_forward_pre_hook(input_hook(layer), with_kwargs=True))
        handles.append(handle)
    try:
        yield trace
    finally:
        for handle in handles:
            handle.remove()


capture_generation_visual = capture_generation
