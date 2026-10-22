"""python -m show_o2_space.generation "Create a dog" --layers 0 8 16"""
import argparse
import json
from pathlib import Path

import torch

from .adapter import ShowO2Adapter, MODEL_ID
from .capture import Capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--layers", type=int, nargs="+", default=[])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/generation"))
    args = parser.parse_args()
    if args.steps < 2 or args.guidance < 0:
        parser.error("--steps must be >= 2 and --guidance must be >= 0")
    adapter = ShowO2Adapter()
    with Capture(adapter.layers, args.layers) as captured:
        image = adapter.generate(args.prompt, args.steps, args.guidance, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    image.save(args.output / "image.png")
    metadata = dict(model=MODEL_ID, resolution=432, prompt=args.prompt, steps=args.steps,
                    guidance=args.guidance, seed=args.seed, layers=args.layers, dtype=str(adapter.dtype),
                    capture="first velocity evaluation; decoder-block outputs before final norm",
                    batch_order=["conditional", "unconditional"] if args.guidance > 0 else ["conditional"],
                    prompt_token_limit=290)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.layers:
        torch.save({"hidden_states": captured.states, "metadata": metadata}, args.output / "hidden_states.pt")
    print(f"Saved to {args.output / 'image.png'}")


if __name__ == "__main__":
    main()
