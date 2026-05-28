"""Patch a molmo2 source tree to include our TrajVLM additions.

TrajVLM extends Molmo2 with:
  * a new vision backbone (SigLip2 ViT + segmenter-driven trajectory grouping)
  * supporting modules (perceiver, DINOv3 wrapper, segmenter ckpt loader)
  * launch scripts (trajvlm_pretrain.py, trajvlm_sft.py)
  * a custom preprocessor (trajvit_preprocessor.py)

This script copies our vendored files INTO the molmo2 tree at the paths
where Molmo2's import system expects them. After running it, you can
launch our training via the regular molmo2 entry points (e.g.
``torchrun launch_scripts/trajvlm_pretrain.py ...``).

Usage:
    python scripts/apply_molmo2_patches.py --molmo2_root /path/to/molmo2

Idempotent: safe to re-run after pulling molmo2 upstream.
"""
import argparse
import os
import shutil
import sys


PATCHES = [
    # (src relative to this trajvlm package, dst relative to molmo2 root)
    ("trajtok_trajvlm/nn/siglip2_trajgroup_vision_backbone.py",
     "olmo/nn/siglip2_trajgroup_vision_backbone.py"),
    ("trajtok_trajvlm/nn/trajvit_perceiver.py",
     "olmo/nn/trajvit_perceiver.py"),
    ("trajtok_trajvlm/nn/trajvit_dinov3.py",
     "olmo/nn/trajvit_dinov3.py"),
    ("trajtok_trajvlm/checkpoints/load_trajvit_segmenter.py",
     "olmo/checkpoints/load_trajvit_segmenter.py"),
    ("trajtok_trajvlm/checkpoints/load_trajvit_segmenter_full.py",
     "olmo/checkpoints/load_trajvit_segmenter_full.py"),
    ("trajtok_trajvlm/preprocessing/trajvit_preprocessor.py",
     "olmo/preprocessing/trajvit_preprocessor.py"),
    ("trajtok_trajvlm/launch_scripts/trajvlm_pretrain.py",
     "launch_scripts/trajvlm_pretrain.py"),
    ("trajtok_trajvlm/launch_scripts/trajvlm_sft.py",
     "launch_scripts/trajvlm_sft.py"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--molmo2_root", required=True, help="Path to a cloned molmo2 source tree")
    parser.add_argument("--rewrite_imports", action="store_true",
                        help="Rewrite the trajtok_trajvlm.* imports inside copied files back to olmo.* so they "
                             "resolve against the molmo2 namespace. Set this if you want molmo2 to be the only "
                             "package on PYTHONPATH (recommended).")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    this_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    # trajvlm/
    if not os.path.isdir(args.molmo2_root):
        print(f"ERROR: --molmo2_root {args.molmo2_root} does not exist", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(os.path.join(args.molmo2_root, "olmo")):
        print(f"ERROR: {args.molmo2_root} does not look like a molmo2 source tree (no olmo/)", file=sys.stderr)
        sys.exit(1)

    print(f"[patch] target molmo2 root: {args.molmo2_root}")
    print(f"[patch] {'(dry run)' if args.dry_run else ''}")
    for src_rel, dst_rel in PATCHES:
        src = os.path.join(this_dir, src_rel)
        dst = os.path.join(args.molmo2_root, dst_rel)
        if not os.path.isfile(src):
            print(f"  SKIP  {src_rel}  (source missing)")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if args.dry_run:
            print(f"  would copy  {src}  →  {dst}")
            continue
        shutil.copyfile(src, dst)
        if args.rewrite_imports:
            with open(dst) as f:
                content = f.read()
            content = content.replace("from trajtok_trajvlm.nn.", "from olmo.nn.")
            content = content.replace("from trajtok_trajvlm.checkpoints.", "from olmo.checkpoints.")
            content = content.replace("from trajtok_trajvlm.preprocessing.", "from olmo.preprocessing.")
            with open(dst, "w") as f:
                f.write(content)
        print(f"  OK    {dst_rel}")
    print()
    print("Patch applied. Now run training via molmo2's launch_scripts/trajvlm_*.py.")


if __name__ == "__main__":
    main()
