"""Capture conditional visual states during generation, keeping denoising time."""
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
def capture_generation_visual(adapter):
    """Pool image tokens at every backbone block output and model evaluation."""
    states = {layer: [] for layer in range(len(adapter.layers))}
    trace = dict(states=states, times=[])
    positions = {}

    def locate(module, args, kwargs):
        offset, length = kwargs["modality_positions"][0, 0].tolist()
        positions["visual"] = slice(offset + 1, offset + length)
        trace["times"].append(float(kwargs["t"].flatten()[0]))

    def hook(layer):
        def capture(module, args, output):
            pooled = output[0][0, positions["visual"]].detach().float().mean(0)
            states[layer].append(pooled.cpu())
        return capture

    handles = [adapter.model.register_forward_pre_hook(locate, with_kwargs=True)]
    handles.extend(block.register_forward_hook(hook(layer))
                   for layer, block in enumerate(adapter.layers))
    try:
        yield trace
    finally:
        for handle in handles:
            handle.remove()
