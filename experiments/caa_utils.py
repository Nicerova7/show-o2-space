"""Answer-letter extraction and steering hooks for the CAA pilot."""
from contextlib import contextmanager


def contrast_tokens(tokenizer, example):
    """Use official CAA questions/labels with Qwen chat formatting."""
    prefix = tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": example["question"]}],
        tokenize=False, add_generation_prompt=True,
    )
    letters = [example["answer_matching_behavior"].strip("()"),
               example["answer_not_matching_behavior"].strip("()")]
    # No closing parenthesis, EOS, or chat terminator after the answer letter.
    return {
        name: tokenizer(prefix + letter, add_special_tokens=False,
                        return_tensors="pt")["input_ids"]
        for name, letter in zip(["positive", "negative"], letters)
    }


@contextmanager
def capture_answer_letter(layers, layer):
    """Output residual stream of a zero-based decoder block, before final norm."""
    captured = {}

    def save(module, args, output):
        captured["activation"] = output[0][0, -1, :].detach().float().cpu().clone()

    handle = layers[layer].register_forward_hook(save)
    try:
        yield captured
    finally:
        handle.remove()


@contextmanager
def steer_understanding(layers, layer, vector, coefficient, input_length):
    """For upstream's full-sequence, no-KV-cache mmu_generate sampler."""
    start = input_length - 1
    trace = []

    def add(module, args, output):
        hidden = output[0].clone()
        hidden[:, start:, :] += coefficient * vector.to(hidden)
        trace.append({"sequence_length": hidden.shape[1],
                      "first_position": start,
                      "last_position": hidden.shape[1] - 1,
                      "positions": hidden.shape[1] - start})
        return (hidden, *output[1:])

    handle = layers[layer].register_forward_hook(add)
    try:
        yield trace
    finally:
        handle.remove()
