"""Token alignment and block-input capture for the ActAdd pilot."""

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
