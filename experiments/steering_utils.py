"""Token alignment, capture, and front injection for the ActAdd pilot."""

from contextlib import contextmanager


def align_prompt_tokens(positive_text, negative_text, tokenizer, bos_id):
    """Prepend the shared BOS and right-pad the shorter sequence with spaces.

    Space tokens are ordinary causal inputs, not masked PAD tokens, matching
    ActAdd's tokens_right alignment with a space as custom_pad_id.
    """
    space_id, = tokenizer(" ", add_special_tokens=False)["input_ids"]
    positive_ids = [bos_id] + tokenizer(positive_text, add_special_tokens=False)["input_ids"]
    negative_ids = [bos_id] + tokenizer(negative_text, add_special_tokens=False)["input_ids"]
    length = max(len(positive_ids), len(negative_ids))
    padding = dict(positive=length - len(positive_ids), negative=length - len(negative_ids))
    positive_ids += [space_id] * padding["positive"]
    negative_ids += [space_id] * padding["negative"]
    return positive_ids, negative_ids, padding


@contextmanager
def capture_layer_input(layers, layer):
    """Capture the residual stream entering one zero-based decoder block."""
    captured = {}

    def save(module, args):
        captured["hidden"] = args[0].detach().cpu().clone()

    handle = layers[layer].register_forward_pre_hook(save)
    try:
        yield captured
    finally:
        handle.remove()


@contextmanager
def inject_layer_input(layers, layer, steering_vector, steering_steps="all", total_steps=None):
    """Inject on the first N evaluations, on 'last' only, or on 'all'.

    Row 0 is the prompt-conditioned sample; a CFG unconditional row is untouched.
    The caller ensures the source sequence fits within the target text prefix.
    'last' uses total_steps measured by the zero-vector control run.
    """
    length = steering_vector.shape[1]
    start = total_steps - 1 if steering_steps == "last" else 0
    stop = float("inf") if steering_steps == "all" else total_steps if steering_steps == "last" else steering_steps
    info = {"calls": 0, "injections": 0}

    def add(module, args):
        hidden = args[0]
        if start <= info["calls"] < stop:
            hidden = hidden.clone()
            hidden[:1, :length, :] += steering_vector.to(hidden)
            info["injections"] += 1
        info["calls"] += 1
        return (hidden, *args[1:])

    handle = layers[layer].register_forward_pre_hook(add)
    try:
        yield info
    finally:
        handle.remove()
