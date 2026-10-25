"""CAA pilot: answer-letter extraction and understanding-only intervention."""
from contextlib import contextmanager

def contrast_tokens(tokenizer, question, positive, negative, pair_index):
    """Alternate A/B labels; keep each pair identical until its answer letter."""
    positive_index = pair_index % 2
    options = [None, None]
    options[positive_index] = positive
    options[1 - positive_index] = negative
    question = f"{question}\n\n(A) {options[0]}\n(B) {options[1]}"
    prefix = tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )
    letters = ["AB"[positive_index], "AB"[1 - positive_index]]
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
