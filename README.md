# show-o2-space

Small tools for extracting, analyzing, and working with internal representations
in Show-o2. The research question is still being developed. S-Space is methodological
inspiration; a spatial interpretation is optional.

Current scope: image-question understanding, text-to-image generation, and selected
transformer-layer hidden states. Model weights stay frozen. Runs on **Colab Linux
with a GPU**, using `showlab/show-o2-1.5B` at **432 x 432**.

## 1. Open Colab

### Red-steering experiment

Generate a baseline and extract a red-minus-whitespace vector with ActAdd token
alignment. Choose `layer` and strength `c` (default: 5), then inject from the first
position. Compare `steering_steps=0`, `12`, `24`, or `"last"`; `"all"` applies throughout.
Show-o2 GPU validation is pending.

[![Open steering in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nicerova7/show-o2-space/blob/experiment/text-red-steering/colab-steering.ipynb)

### General demo

Run image understanding, extract hidden states, and generate images with Show-o2.

[![Open demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nicerova7/show-o2-space/blob/main/colab.ipynb)

For either notebook: select **GPU**, run setup, **restart the session**, then load
the model. GPU execution of the steering notebook remains unverified.

## 2. Run understanding

In the notebook, section 3 saves `hidden_states.pt` and `metadata.json` without
decoding text. Its optional answer cell saves `answer.txt` using the same prepared
inputs. Rerun extraction when changing the image or question.

For a separate CLI run from the project directory, with the notebook's Python environment:

```bash
python -m show_o2_space.understanding inputs/images/example.jpg "Describe this image."
```

Upload your own image in the notebook, then change the path and question:

```bash
python -m show_o2_space.understanding inputs/images/dogs.jpg "How many dogs are there?" --layers 0 8 16
```

Saves `outputs/understanding/answer.txt`, `metadata.json`, and, when layers are
requested, `hidden_states.pt`. Preprocessing matches the original demo: shortest-side
resize and **center crop**, which can remove objects at the edges.

## 3. Run generation

```bash
python -m show_o2_space.generation "Create a dog" --seed 42
```

Saves `outputs/generation/image.png` and `metadata.json`. Optional arguments:
`--steps 50`, `--guidance 5`, `--layers 0 8 16`, and `--output outputs/my-run`.
Prompts are truncated to 290 tokens, matching the original 1024-position setup.
Use a different output directory to preserve each run; the default files are overwritten.
This research wrapper saves decoded pixels directly and does not run the upstream
demo's separate safety-checker postprocessing or Weights & Biases logging.

## 4. Inspect hidden states

```python
import torch
saved = torch.load("outputs/understanding/hidden_states.pt", map_location="cpu", weights_only=True)
print(saved["hidden_states"][8].shape)  # [batch, sequence positions, hidden dimensions]
print(saved["metadata"]["alignment"])
```

Layer indices are **zero-based decoder block indices**, from
`adapter.model.showo.model.layers`. Captures are block outputs **before the final
backbone norm**, not embedding-layer states or vision-encoder states.

- Understanding captures a single prompt forward pass in the notebook (or the
  first answer-sampler pass in the CLI), before any answer
  token is appended. Metadata maps text token IDs, time position, and image span.
  Image positions are embeddings, not vocabulary tokens; their IDs are recorded as -1.
- Generation captures the first velocity evaluation only, at the start of sampling.
  With guidance, batch row 0 is conditional and row 1 is unconditional. These are
  intermediate noisy-image representations, not representations of the final image.
- Captures are detached CPU tensors by default; only selected layers are stored.
  Saved files contain no live autograd graph. Do not compare sequence positions
  between the tasks without accounting for their different prompt layouts.

For later gradient experiments, use the Python API in the notebook environment:

```python
from show_o2_space.adapter import ShowO2Adapter
from show_o2_space.capture import Capture

adapter = ShowO2Adapter()
embeddings, mask, alignment = adapter.prepare_understanding(
    "inputs/images/example.jpg", "Describe this image."
)
with Capture(adapter.layers, [8], keep_graph=True) as capture:
    result = adapter.forward_understanding(embeddings, mask, activation_gradients=True)
    # Define your scalar objective from result.logits, then call objective.backward().
    # capture.states[8].grad will contain activation gradients.
```

This enables differentiation through a single understanding forward pass while all
parameters remain frozen. The upstream answer/image samplers use `no_grad`;
differentiating through decoding or the image sampler is outside this milestone.
The full gradient graph uses more VRAM. In the notebook, run gradient experiments
in an ordinary Python cell and reuse its existing `adapter`; omit the constructor in the
example above. Each standalone CLI command still starts a new process and loads
the model again.

## Files

```text
colab.ipynb                 persistent adapter, extraction, optional answers, generation
show_o2_space/
  adapter.py               imports upstream, loads frozen weights, prepares inputs
  understanding.py         image + question CLI
  generation.py            prompt -> image CLI
  capture.py               selected decoder-layer hooks
inputs/images/             your images
outputs/                   answers, images, tensors, metadata
experiments/               future research scripts
```

## Implementation notes and inspiration

The model implementation is **imported, not copied or modified**. The notebook sets
`PYTHONPATH` and the notebook import path to `/content/Show-o/show-o2` and
the project directory. All package imports are at the tops of their files. If you start
a separate shell, first run `export PYTHONPATH=/content/Show-o/show-o2:/content/show-o2-space`.
The notebook pins commit
`45a5a2de01d1ebd10cd5864d29310a76476cdf23`; dependency versions follow Show-o2's
own `build_env.sh`, trimmed to the inference imports. NumPy allows Colab's
installed version and SentencePiece uses 0.2.1 for newer Python wheels. PyTorch
and torchvision come from Colab, so this is not the old pinned Python 3.11 setup. Model weights currently use
Hugging Face's default revision; pin those too before a reproducible study.

Read these upstream files when extending the adapter:

| File under `Show-o/show-o2/` | Purpose |
| --- | --- |
| `inference_mmu.py` | Image encoding, chat sequence, answer generation |
| `inference_t2i.py` | Noise, guidance, flow sampling, VAE decoding |
| `configs/showo2_1.5b_demo_432x432.yaml` | Resolution, sequence lengths, sampler defaults |
| `models/modeling_showo2_qwen2_5.py` | Dual image paths, fusion, backbone, diffusion head |
| `models/qwen2.py` | Shared decoder blocks used for capture |
| `models/misc.py` | Tokenizer and generation sequence construction |
| `models/wan21_vae.py`, `transport/` | Image latent encoding/decoding and flow sampler |

[S-Space](https://github.com/MirroS-Lab/S-Space) separates model access, extraction,
and experiments. Its frozen-model adapters and intermediate-representation analysis
motivate that separation here. We do not copy its spatial axes, datasets, or model-specific
interventions. Potential next steps are output-contrast gradients and activation edits,
after establishing reliable inference and choosing a research objective.

Sources: [Show-o2 implementation](https://github.com/showlab/Show-o/tree/45a5a2de01d1ebd10cd5864d29310a76476cdf23/show-o2),
[Show-o2 checkpoint](https://huggingface.co/showlab/show-o2-1.5B),
[Wan VAE](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/blob/main/Wan2.1_VAE.pth),
[S-Space model adapter](https://github.com/MirroS-Lab/S-Space/blob/main/sspace/core/models/qwen.py).
Show-o's original code and licenses remain in its separate checkout.

## Verification status

Understanding answer generation completed in Colab using the previous
CLI notebook flow. A 12 GiB RAM / T4 runtime was killed during loading; peak memory
was not measured. Generation and activation gradients remain unverified in Colab.

The revised notebook reuses Colab's own Python kernel and one adapter. Its prompt
extraction and optional answer cells are separate. Local checks cover syntax and
notebook execution flow with a small stand-in model; these do not validate real
Show-o2 loading, numerical outputs, or GPU memory use. Run the revised notebook
on a GPU before treating it as a validated baseline.
No tests directory, CI, training pipeline, or benchmark framework is included.
