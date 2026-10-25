# CAA with Show-o2 1.5B

[Open in Colab](https://colab.research.google.com/github/Nicerova7/show-o2-space/blob/experiment/caa-understanding/colab-caa-understanding.ipynb)

The CAA pilot adds two files to the original repository structure:

- `colab-caa-understanding.ipynb`: setup, model loading, 100 official CAA pairs, token audit, subtraction and averaging, image + prompt, and three steering coefficients.
- `experiments/caa_utils.py`: contrast-pair tokenization and extraction/steering hooks.

The notebook imports `ShowO2Adapter` and `MODEL_ID` from `show_o2_space.adapter` to load the model, prepare images, and generate text. It imports the CAA functions from `experiments.caa_utils`, following the structure of `text-red-steering`.

Use a BF16 GPU (A100 or L4) and High-RAM. Run setup, restart the session, and continue with model loading.

Setup downloads the CAA repository. The notebook reads `/content/CAA/datasets/generate/sycophancy/generate_dataset.json` and samples 100 examples without replacement with `random.Random(SEED).sample`. It preserves the official questions, options, and labels without reordering A/B. The selected examples are saved as `selected_examples.json` in the output ZIP.

Push your local changes before using the Colab link, which loads the published branch.

Extraction uses text. Positive examples use `answer_matching_behavior`; negative examples use `answer_not_matching_behavior`. Each pair differs only in its final answer letter. We use Qwen chat formatting instead of `tokenize_llama_chat`. Qwen merges `(A` into one token, so the official `(A)` / `(B)` labels are represented as standalone letters. Extraction uses `-1`, exactly at A/B. The Llama example's `-2` does not transfer literally to Qwen. The audit displays the selected tokens before model execution.

The vector is `mean(h_positive - h_negative)`, without additional normalization. Extraction and intervention use the selected decoder block output, before final normalization. `LAYER = 12` is an editable starting point: block index 12 is the 13th block, since indexing starts at zero.

Evaluation takes an image + prompt and generates text with `c = -2, 0, 2`. The hook adds `c*v` from the last input token through the end of the sequence. The original sampler recomputes the full sequence without a KV cache: the intervention range grows from 1 to 2 to 3 positions. The notebook displays this trace and downloads the results.

Images are resized and center-cropped to 432×432; the crop is displayed. Show-o is downloaded separately at commit `45a5a2de01d1ebd10cd5864d29310a76476cdf23`.

Branch: `experiment/caa-understanding`, created from `main`. Local checks cover syntax, notebook schema, answer letters with the actual Qwen tokenizer, and hook arithmetic/ranges with small tensors. Full model execution with weights belongs in Colab.

References: [CAA paper](https://arxiv.org/html/2312.06681v4#S3), [CAA repository](https://github.com/nrimsky/CAA), [Show-o2](https://github.com/showlab/Show-o/tree/45a5a2de01d1ebd10cd5864d29310a76476cdf23/show-o2).

The original `main` structure is preserved: `show_o2_space/`, `inputs/`, `outputs/`, `experiments/`, and `colab.ipynb`. The pilot follows `experiment/text-red-steering`: a dedicated notebook with its utilities in `experiments/`.
