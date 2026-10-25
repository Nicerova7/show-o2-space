"""python -m show_o2_space.understanding image.jpg "Question?" --layers 0 8 16"""
import argparse
import json
from pathlib import Path

import torch

from .adapter import ShowO2Adapter, MODEL_ID
from .capture import Capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("question")
    parser.add_argument("--layers", type=int, nargs="+", default=[])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/understanding"))
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    torch.manual_seed(args.seed)
    adapter = ShowO2Adapter()
    embeddings, mask, alignment = adapter.prepare_understanding(args.image, args.question)
    with Capture(adapter.layers, args.layers) as captured:
        answer = adapter.answer(embeddings, mask, args.max_new_tokens)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(model=MODEL_ID, resolution=432, image=str(args.image.resolve()),
                    question=args.question, answer=answer, layers=args.layers, seed=args.seed,
                    dtype=str(adapter.dtype), capture="prefill decoder-block outputs before final norm",
                    alignment=alignment)
    (args.output / "answer.txt").write_text(answer + "\n", encoding="utf-8")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.layers:
        torch.save({"hidden_states": captured.states, "metadata": metadata}, args.output / "hidden_states.pt")
    print(answer)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
