"""Download the released TrajTok-v2 segmenter checkpoint from HuggingFace Hub.

Usage:
    python scripts/download_ckpt.py [--output PATH]

Default output: ./checkpoints/segmenter_filteredmixdata_all.pth
"""
import argparse
import os
import sys


HF_REPO = "michaelzch001/trajtokv2-segmenter"
HF_FILENAME = "model.pth"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output", default="./checkpoints/segmenter_filteredmixdata_all.pth",
        help="Local path to save the checkpoint",
    )
    parser.add_argument(
        "--repo", default=HF_REPO,
        help=f"HuggingFace repo id (default: {HF_REPO})",
    )
    parser.add_argument(
        "--filename", default=HF_FILENAME,
        help=f"File name within the repo (default: {HF_FILENAME})",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Downloading {args.repo}/{args.filename}  →  {args.output}")
    local_path = hf_hub_download(
        repo_id=args.repo,
        filename=args.filename,
        local_dir=out_dir,
    )
    # Move/rename to requested output path if HF placed it under a subdir.
    if os.path.abspath(local_path) != os.path.abspath(args.output):
        os.replace(local_path, args.output)
    sz_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Done. Saved {sz_mb:.0f} MB to {args.output}")


if __name__ == "__main__":
    main()
