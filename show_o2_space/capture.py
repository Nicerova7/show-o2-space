"""Capture decoder-block outputs; no model implementation is copied."""


class Capture:
    def __init__(self, layers, indices, keep_graph=False):
        self.layers = layers
        self.indices = list(dict.fromkeys(indices))
        if any(index < 0 or index >= len(layers) for index in self.indices):
            raise ValueError(f"Layer indices must be between 0 and {len(layers) - 1}")
        self.keep_graph = keep_graph
        self.states = {}
        self.handles = []

    def __enter__(self):
        self.states.clear()
        for index in self.indices:
            self.handles.append(self.layers[index].register_forward_hook(self._hook(index)))
        return self

    def _hook(self, index):
        def save(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if self.keep_graph:
                if not hidden.requires_grad:
                    raise RuntimeError("Use forward_understanding(..., activation_gradients=True)")
                hidden.retain_grad()
                self.states[index] = hidden
            else:
                # Keep the first call only: prefill for understanding, first
                # velocity evaluation for generation. Avoid collecting all steps.
                if index not in self.states:
                    self.states[index] = hidden.detach().cpu().clone()
        return save

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
