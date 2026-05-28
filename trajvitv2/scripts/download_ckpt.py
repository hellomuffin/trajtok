"""Download the released TrajViT-v2 (Panda-70M small-scale) checkpoint from HuggingFace Hub.

NOTE: this checkpoint is trained on a small Panda-70M subset (~600 K clips) for
~20 epochs. It is released as a starting point for fine-tuning, NOT as a
production model. See the README for caveats.
"""
import argparse
import os
import sys

HF_REPO = "michaelzch001/trajtokv2-trajvitv2"
HF_FILENAME = "model.pth"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="./checkpoints/trajvitv2_filteredmixdata_new.pth")
    parser.add_argument("--repo", default=HF_REPO)
    parser.add_argument("--filename", default=HF_FILENAME)
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("pip install huggingface_hub", file=sys.stderr); sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    print(f"Downloading {args.repo}/{args.filename}  →  {args.output}")
    local_path = hf_hub_download(repo_id=args.repo, filename=args.filename, local_dir=out_dir)
    if os.path.abspath(local_path) != os.path.abspath(args.output):
        os.replace(local_path, args.output)
    sz_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Done. Saved {sz_mb:.0f} MB to {args.output}")


if __name__ == "__main__":
    main()
