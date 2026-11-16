# Safety transfer

[Open U→G shared backbone in Colab](https://colab.research.google.com/github/Nicerova7/show-o2-space/blob/experiment/custom-steering-vectors/experiments/safety-transfer/colab-u2g-shared.ipynb)

Run `colab-u2g-shared.ipynb` in order. Restart after installation. It reproduces
the safety experiment's 128 harmful / 128 harmless text-only extraction, saves
unit directions per layer, then uses `pairs.json` for 32 validation pairs and
50 test pairs. Both members of each pair are generated.

Only additive steering is used: `hidden + strength * vector`, on conditional
prompt text in shared decoder blocks at every flow evaluation. BOS, image
markers, time embeddings, image tokens and the unconditional CFG row are excluded.
Choose layer(s) and signed strengths in validation. Freeze them for test.
The intervention reuses `custom_utils.steer_denoising`, as in counting;
`safety_transfer_utils.py` only handles generation batches and manual review.

Download each image archive, manually label its `review.xlsx`, then upload the
filled workbook in the next cell. Metrics summarize human graphic severity,
subject/context preservation and image quality. No automatic refusal rules.
Uploaded scores are joined by image ID to the original run manifest. Summary
columns ending in `_change` subtract the same-kind baseline; negative graphic
change indicates reduction, while negative context/quality change indicates loss.
At defaults validation generates 320 images, test 200; the run uses one seed.

Data: [NASK-PIB/SafeSteerDataset](https://huggingface.co/datasets/NASK-PIB/SafeSteerDataset),
Apache-2.0; [Conditioned Activation Transport for T2I Safety Steering](https://arxiv.org/abs/2603.03163).
The supplied parquet files were mapped by original split and zero-based row.
`provenance.json` records SHA-256 hashes of the supplied files. All 82 selected
records were checked against those files without changing their prompts.
Custom splits have disjoint situation names, but related injury concepts remain.
Validation contains 16 interpersonal-violence / 16 self-harm pairs; test 25 / 25.
Labels are source labels, not validated ground truth. The selected prompts include
some non-malicious assistance after injury: graphicness and harmful intent differ.
Safe prompts are controls, not exact pixel or scene targets. The unchanged 128/128
extraction sample has not been individually remapped here.

The generation-specific flow-head intervention belongs in a separate future
notebook in this directory. G→U is also outside this notebook.

Local checks cover notebook syntax, split counts and synthetic hooks/metrics.
Actual Show-o2 generation and GPU behavior require running Colab.
