"""Strip optimizer/scheduler/scaler state from a training checkpoint to produce a
release-friendly model-only ``.pth`` file.

Usage:
    python release/prep_ckpt.py \\
        --input  /path/to/training_run/latest.pth \\
        --output release/segmenter_filteredmixdata_all.pth

The output file contains a single top-level ``model`` key (state_dict) plus a
small ``meta`` dict (epoch, global_step, source path) for traceability.
"""
import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to the training-time .pth")
    parser.add_argument("--output", required=True, help="Where to write the model-only .pth")
    args = parser.parse_args()

    print(f"[prep] loading {args.input}")
    src = torch.load(args.input, map_location="cpu", weights_only=False)
    if "model" not in src:
        raise ValueError(f"Source ckpt has no 'model' field. Top-level keys: {list(src.keys())}")

    out = {
        "model": src["model"],
        "meta": {
            "epoch": src.get("epoch"),
            "global_step": src.get("global_step"),
            "source_path": os.path.abspath(args.input),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    print(f"[prep] writing {args.output}")
    torch.save(out, args.output)

    in_mb = os.path.getsize(args.input) / (1024 ** 2)
    out_mb = os.path.getsize(args.output) / (1024 ** 2)
    print(f"[prep] done. input={in_mb:.1f} MB  output={out_mb:.1f} MB  saved {in_mb-out_mb:.1f} MB")


if __name__ == "__main__":
    main()
