"""Transport a U direction and reuse the denoising hook in the flow head."""
from contextlib import contextmanager

import torch

from experiments.custom_utils import steer_denoising


@torch.no_grad()
def project_flow_direction(adapter, vector):
    """Return P(v) - P(0), without normalization."""
    projection = adapter.model.diff_proj
    parameter = next(projection.parameters())
    vector = vector.to(parameter).reshape(1, 1, -1)
    with adapter.autocast():
        direction = projection(vector) - projection(torch.zeros_like(vector))
    return direction.flatten().float().cpu()


@contextmanager
def steer_flow_head(adapter, *, vectors, layers="all", text=None, visual=None):
    """Add steering at flow-head block inputs, before attention."""
    with steer_denoising(
        adapter, vectors=vectors, layers=layers, text=text, visual=visual,
        blocks=adapter.model.diffusion_head_a, site="input",
    ) as trace:
        yield trace
