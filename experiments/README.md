# Experiments

## Custom steering vectors

`custom_utils.py` reuses `show_o2_space.capture.Capture` and the loaded adapter.
Run `colab-custom-steering.ipynb` for the stepwise experiment.
Steps 5 and 6 preview six real coordinates: positive minus negative, then sum divided by token count.

The [counting build dataset](../inputs/counting/one_to_two/README.md) provides 50
simple QA pairs for `one` versus `two`, with supplied answers at the final position.
Run [the counting notebook](counting/colab-counting.ipynb): sections 1–9 build and
save one mean-difference vector per layer. Sections 10–12 optionally apply them.

### API

```python
make_steering_vector(
    adapter,
    pair_pos="A red car.",
    pair_neg="A blue car.",
    alignment="right_pad",
    tokens="last",
    layers="all",
    token_reduction="mean",
    direction_method="difference",
    layer_reduction="none",
)
```

| Parameter | Values | Meaning |
| --- | --- | --- |
| `pair_pos`, `pair_neg` | Strings | One raw contrastive text pair |
| `alignment` | `"none"`, `"right_pad"` | Optionally append visible spaces to the shorter token sequence |
| `tokens` | Integer, list, slice, `"last"`, `"all"` | Select positions independently in each sequence |
| `layers` | Integer, list, slice, `"last"`, `"all"` | Select zero-based decoder blocks |
| `token_reduction` | `"none"`, `"sum"`, `"mean"`, `"average"`, callable | Pool selected tokens separately per example |
| `layer_reduction` | Same reducers | Optionally aggregate per-layer results |
| `direction_method` | `"difference"`, `"svd"`, callable | Estimate a direction from pooled example states |

`0` selects the first position or block, `1` the second, and `-1` the last.
Slices follow Python's end-exclusive convention. Token positions include the shared
BOS. With right padding, the last position may be a space. Padding appends spaces;
it does not shift either sequence to the right or use masked PAD tokens.

Captures are decoder block outputs before final normalization, shaped `[tokens, hidden]`,
stored on CPU in FP32. Two independent text-only causal forwards use frozen weights.
No chat formatting, image input, answer decoding, normalization, or coefficient is added.

The pipeline selects token rows, pools each example, estimates a direction,
then optionally reduces layers. Saved tokenwise differences require equal selected lengths.
`mean` and `average` are aliases. With no token reduction, each layer retains a
`[selected_tokens, hidden]` matrix; other reducers return `[hidden]`.

The default direction is pooled positive minus pooled negative. `"svd"` stacks the
pooled example states without centering, then returns the leading right singular
vector. Its sign follows the mean positive-minus-negative contrast when possible;
otherwise its largest coordinate is positive. This uncentered direction can capture
shared activation structure rather than a contrast. No mean is subtracted from the
stacked states. Layer reduction mixes residual depths and is an explicit experiment.

Custom token/layer reducers use `reducer(states, *, dim)` and remove that axis.
A direction callable uses `method(states)` and receives `[examples, hidden]` pooled
states, positive examples followed by negative examples. It returns one `[hidden]`
vector. The notebook demonstrates mean-centering and PCA inside such a callable;
neither operation is imposed by the extraction pipeline. Its sign is chosen by the
callable itself. SVD and custom direction methods require token pooling first.

`estimate_direction(positive, negative, method)` accepts `[examples, hidden]`
matrices for future multi-pair experiments. The current notebook captures one pair;
it does not yet provide a dataset capture loop. Each layer is estimated separately.

### Results and reuse

- `vectors`: mapping from block index to its reduced direction or tokenwise matrix.
- `combined`: optional reduction across selected layers; otherwise `None`.
- `positive`, `negative`, `differences`: selected states per layer.
- `metadata`: exact texts, token IDs, padding, positions, layers, and reducer names.

Use `prepare_pair`, `capture_pair`, and `build_steering_vector` separately to inspect
intermediates and change reducers without repeating model forwards. The notebook
saves the result, full captures, and revisions. Callable definitions remain in the
notebook; only their names are stored in metadata.

Local verification covers synthetic reduction math, direction estimation, hook cleanup, token alignment,
and end-to-end extraction with a small decoder. Real Show-o2 GPU execution remains
unverified. The original model implementation and `show_o2_space` remain unchanged.

## Denoising application

```python
with count_denoising_evaluations(adapter) as baseline_trace:
    baseline = adapter.generate(prompt, steps=50, seed=42)

with steer_denoising(
    adapter, vectors=vectors, total_evaluations=baseline_trace["evaluations"],
    layers="all", iterations="all", text=None,
    visual=dict(tokens="all", operation="add", alpha=0.1),
) as trace:
    steered = adapter.generate(prompt, steps=50, seed=42)
```

Import both functions from `experiments.custom_utils`. Keep all generation settings
identical between runs. Layers, iterations, and tokens accept indices, lists, slices,
`"first"`, `"last"`, or `"all"`. Iterations count measured model evaluations;
`slice(0, 24)` selects the first 24. The baseline count belongs to that sampler setup.

Text and visual rules are independent. `None` disables a modality. Token positions
are relative to each modality; prompt content excludes special tokens and padding,
and visual positions exclude the time embedding. Only conditional CFG row 0 is edited.
Each selected block output receives its corresponding layer vector.

- `"add"`: `h + alpha*v`.
- `"component"`: `h + alpha*(h @ unit_v)*unit_v`; `-1` removes and `1` doubles it.
- Callable: `operation(hidden, vector, *, alpha)` returns edited selected states.

Hooks are removed at context exit. The trace records evaluations, injections, and
six before/after coordinates per layer/modality. Show-o2 denoising generates no AR
text; autoregressive application is outside this implementation.

CPU checks: `python -m unittest experiments.test_custom_utils -v`.
