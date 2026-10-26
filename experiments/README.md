# Refusal pilot: Show-o2 understanding

Open `colab-refusal.ipynb` from this branch in Colab. Run setup, restart the
session, then execute steps 2-9. The final extra cell takes your own 10 harmful
and 10 harmless prompts with image paths. Upload images to the existing
`inputs/images/` folder using Colab's Files panel.

`refusal_utils.py` imports no model implementation: it works with the existing
`ShowO2Adapter` from main. Only image + text -> text is used. No training,
weight editing, or text-to-image experiment is included.

The nine steps follow the supplied `colab_refusal.ipynb`: setup, model, data,
suffix positions, mean differences, selection, vector inspection, addition,
and ablation. Data use the same reference-repo splits, sampling order and seed
42: 128 harmful + 128 harmless for build, 32 + 32 for validation.

Step 3 applies appendix B's baseline filter to the 32 + 32 validation prompts:
retain harmful prompts with refusal log odds >= 0 and harmless prompts with
log odds <= 0. Build prompts and custom tests remain intact. Retained groups
can contain fewer than 32 examples; no replacements are sampled. Inspect the
printed counts before continuing: an empty group cannot support selection.
The baseline scores and retention decisions are saved in `validation_filter.json`.
The I / As proxy still needs behavioral validation for Show-o2.

The paper is [Arditi et al.](https://arxiv.org/abs/2406.11717), sections 2.3,
2.4 and C.1, with [reference code](https://github.com/andyrdt/refusal_direction).
This is a multimodal adaptation, not a result established by that paper:
build and validation hold one image fixed while changing the original text
instructions. Tests can use different images. A direction might not pass the
paper's filters on Show-o2; scores are saved before selecting the winner.

- Capture residual inputs at every decoder block and every post-instruction
  token in the adapter's actual `\n<|im_start|>assistant\n` suffix.
- Compute `r = mean(harmful) - mean(harmless)`, keeping its magnitude.
- Score only source layers `range(int(0.8 * L))`, following the reference
  implementation's integer cutoff. The 80% limit applies to candidate source
  layers, not the layers affected by ablation.
- Minimize bypass log odds, subject to induce log odds > 0 and KL < 0.1.
  The metric is **KL divergence, not KS**. KL uses the reference implementation's
  epsilon of 1e-6. Refusal log odds use the first-answer-token probability of
  `I` / `As`, obtained from this tokenizer. This proxy is not a harmfulness judge.
- Induce refusal with `h + r` at the selected layer, across all positions.
- Suppress refusal with `h - (h dot r_hat) r_hat`: block input hooks and
  attention/MLP output hooks across all layers and all positions, including
  image/time positions. Hooks remain active during every autoregressive step.

The fixed image is encoded once for build/validation. Validation embeddings
are cached on CPU; examples run one at a time. Selection only computes logits
at the last position. Response generation uses the adapter's original greedy
answer sampler. All hooks are removed after each intervention.

Outputs go to `outputs/refusal/`: sampled data, candidate vectors, all scores,
passing scores, selected vector and metadata, test manifest and an Excel with
40 responses for 20 custom prompts (baseline plus intervention). Fill the Excel's
`refusal` column manually. The notebook downloads a ZIP of these outputs.

Local validation covers syntax, notebook structure and intervention arithmetic.
Full loading, VRAM use and end-to-end results on Show-o2 remain to be verified
in Colab. Prefer an A100 runtime for the first run.
