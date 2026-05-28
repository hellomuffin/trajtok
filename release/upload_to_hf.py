"""One-button HuggingFace Hub uploader for the TrajTok-v2 released checkpoints.

This script:
  1. Creates two public HF repos under your account (skips if they already exist)
  2. Uploads the prepped checkpoint files
  3. Uploads the model cards (README.md inside each repo)

Required:
  * `huggingface_hub` installed:    pip install huggingface_hub
  * HF token with write permission. Set ONE of:
        - HUGGINGFACE_TOKEN env var
        - HF_TOKEN env var
        - run `huggingface-cli login` (saves to ~/.huggingface/token)

Usage (after running release/prep_ckpt.py on both ckpts):

    HF_TOKEN=hf_xxx python release/upload_to_hf.py \\
        --user hellomuffin \\
        --segmenter_ckpt release/segmenter_filteredmixdata_all.pth \\
        --trajvitv2_ckpt release/trajvitv2_filteredmixdata_new.pth

Add ``--dry_run`` to print what WOULD be uploaded without touching HF.

Each repo is created with:
  * apache-2.0 license
  * public visibility (per user choice; pass --private to override)
  * README.md = the corresponding model card from release/
"""
import argparse
import os
import sys


# Default repo names — change with --segmenter_repo / --trajvitv2_repo if desired
DEFAULT_SEGMENTER_REPO  = "trajtokv2-segmenter"
DEFAULT_TRAJVITV2_REPO  = "trajtokv2-trajvitv2"

# Names the model files will land under inside each repo
HF_CKPT_FILENAME = "model.pth"


def upload_one(api, user: str, repo_name: str, ckpt_path: str, card_path: str,
               private: bool, dry_run: bool):
    repo_id = f"{user}/{repo_name}"
    print(f"\n[{repo_id}] preparing")

    if dry_run:
        print(f"  would create repo  (private={private})")
        print(f"  would upload ckpt  {ckpt_path}  →  {HF_CKPT_FILENAME}")
        print(f"  would upload card  {card_path}  →  README.md")
        return

    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    print(f"  repo OK")

    print(f"  uploading model card  ({os.path.getsize(card_path) / 1024:.1f} KB)")
    api.upload_file(
        path_or_fileobj=card_path, path_in_repo="README.md",
        repo_id=repo_id, repo_type="model",
        commit_message="Add model card",
    )

    print(f"  uploading checkpoint  ({os.path.getsize(ckpt_path) / (1024**2):.1f} MB) — this can take a few minutes")
    api.upload_file(
        path_or_fileobj=ckpt_path, path_in_repo=HF_CKPT_FILENAME,
        repo_id=repo_id, repo_type="model",
        commit_message="Add released checkpoint",
    )
    print(f"  done → https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", required=True, help="HF username (e.g. hellomuffin)")
    parser.add_argument("--segmenter_ckpt", default="release/segmenter_filteredmixdata_all.pth")
    parser.add_argument("--trajvitv2_ckpt", default="release/trajvitv2_filteredmixdata_new.pth")
    parser.add_argument("--segmenter_repo", default=DEFAULT_SEGMENTER_REPO)
    parser.add_argument("--trajvitv2_repo", default=DEFAULT_TRAJVITV2_REPO)
    parser.add_argument("--segmenter_card", default="release/segmenter_MODEL_CARD.md")
    parser.add_argument("--trajvitv2_card", default="release/trajvitv2_MODEL_CARD.md")
    parser.add_argument("--private", action="store_true", help="Create private repos (default: public)")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_segmenter", action="store_true")
    parser.add_argument("--skip_trajvitv2", action="store_true")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: pip install huggingface_hub", file=sys.stderr); sys.exit(1)

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    api = HfApi(token=token) if token else HfApi()  # falls back to ~/.huggingface/token

    # Verify auth before doing anything
    try:
        whoami = api.whoami()
        print(f"[auth] logged in as {whoami['name']}")
    except Exception as e:
        print(f"ERROR: HF authentication failed. Set HUGGINGFACE_TOKEN or run "
              f"`huggingface-cli login`. Underlying error: {e}", file=sys.stderr)
        sys.exit(1)

    # Sanity check files exist
    targets = []
    if not args.skip_segmenter:
        for p in (args.segmenter_ckpt, args.segmenter_card):
            if not os.path.isfile(p):
                print(f"ERROR: missing file {p}", file=sys.stderr); sys.exit(1)
        targets.append(("segmenter", args.segmenter_repo, args.segmenter_ckpt, args.segmenter_card))
    if not args.skip_trajvitv2:
        for p in (args.trajvitv2_ckpt, args.trajvitv2_card):
            if not os.path.isfile(p):
                print(f"ERROR: missing file {p}", file=sys.stderr); sys.exit(1)
        targets.append(("trajvitv2", args.trajvitv2_repo, args.trajvitv2_ckpt, args.trajvitv2_card))

    for tag, repo_name, ckpt, card in targets:
        upload_one(api, args.user, repo_name, ckpt, card, args.private, args.dry_run)

    print()
    print("All uploads complete. Next step: edit segmenter/scripts/download_ckpt.py and "
          "trajvitv2/scripts/download_ckpt.py to point at the new repo IDs (we leave them "
          "as placeholders since you may want to rename).")


if __name__ == "__main__":
    main()
