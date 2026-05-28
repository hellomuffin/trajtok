"""MOSE val evaluation for the trajectory segmenter.

Same metrics as eval_davis.py (Hungarian-matched VEQ + STQ_EN) — only the
dataset directory layout differs.

Usage:
    python scripts/eval_mose.py \\
        --ckpt checkpoints/segmenter_filteredmixdata_all.pth \\
        --mose_root /path/to/MOSE \\
        --output_dir results/mose_eval

MOSE layout this script expects (the standard release):

    {mose_root}/
        valid/
            JPEGImages/<video>/*.jpg
            Annotations/<video>/*.png    # palette PNG, pixel value = instance id

We treat MOSE val as the eval split (`--split valid` by default). MOSE annotations
use pixel value 0 for background and contiguous IDs 1..N for instances.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch
from PIL import Image

# Reuse the DAVIS eval helpers — they're parameterised enough to apply directly.
from scripts.eval_davis import (
    build_model, segmenter_predict, save_clip_viz, _IMAGENET_MEAN, _IMAGENET_STD,
)
from trajtok_segmenter.eval.seg_metric import veq_scores, stq_en


def list_mose_videos(mose_root: str, split: str = "valid"):
    """Return sorted list of MOSE video names from the `JPEGImages/` directory."""
    img_root = os.path.join(mose_root, split, "JPEGImages")
    if not os.path.isdir(img_root):
        raise FileNotFoundError(
            f"Expected {img_root}. Make sure --mose_root points at the MOSE release "
            "(with valid/JPEGImages/ and valid/Annotations/)."
        )
    return sorted(d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d)))


def load_mose_clip(mose_root: str, split: str, video_name: str, num_frames: int, image_res: int):
    """Same shape contract as load_davis_clip — see eval_davis.py."""
    img_dir = os.path.join(mose_root, split, "JPEGImages", video_name)
    ann_dir = os.path.join(mose_root, split, "Annotations", video_name)

    img_files = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
    if not img_files:
        raise FileNotFoundError(f"No .jpg files in {img_dir}")

    if len(img_files) >= num_frames:
        idx = np.linspace(0, len(img_files) - 1, num_frames).round().astype(int)
    else:
        idx = list(range(len(img_files))) + [len(img_files) - 1] * (num_frames - len(img_files))
        idx = np.asarray(idx, dtype=int)
    chosen = [img_files[i] for i in idx]

    raw_frames, frames_tensor, gt_masks = [], [], []
    for fname in chosen:
        img = np.asarray(Image.open(os.path.join(img_dir, fname)).convert("RGB"))
        raw_frames.append(img)
        resized = np.asarray(
            Image.fromarray(img).resize((image_res, image_res), Image.BICUBIC),
            dtype=np.float32,
        ) / 255.0
        resized = (resized - _IMAGENET_MEAN) / _IMAGENET_STD
        frames_tensor.append(resized.transpose(2, 0, 1))

        mask_fname = fname.replace(".jpg", ".png")
        mask_path = os.path.join(ann_dir, mask_fname)
        if not os.path.isfile(mask_path):
            # MOSE doesn't always annotate every frame — fall back to zeros (no-op
            # for IoU because gt_size becomes 0).
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
        else:
            mask = np.asarray(Image.open(mask_path), dtype=np.int32)
        gt_masks.append(mask)

    return (
        torch.from_numpy(np.stack(frames_tensor, 0)),
        np.stack(gt_masks, 0),
        np.stack(raw_frames, 0),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mose_root", required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--output_dir", default="results/mose_eval")
    parser.add_argument("--config", default=None)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--image_res", type=int, default=224)
    parser.add_argument("--num_traj", type=int, default=128)
    parser.add_argument("--iou_thr", type=float, default=0.5)
    parser.add_argument("--save_viz", type=int, default=0)
    parser.add_argument("--max_videos", type=int, default=0)
    args = parser.parse_args()

    if args.config is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.config = os.path.join(here, "..", "configs", "pretrain.yaml")
    os.makedirs(args.output_dir, exist_ok=True)

    model = build_model(args.ckpt, args.config, args.num_traj)
    videos = list_mose_videos(args.mose_root, split=args.split)
    if args.max_videos > 0:
        videos = videos[: args.max_videos]
    print(f"[mose:{args.split}] {len(videos)} videos, {args.num_frames} frames each")

    per_video, veq_all, stq_all, sq_all, aq_all = [], [], [], [], []
    for vi, vname in enumerate(videos):
        try:
            frames, gt_masks, raw_frames = load_mose_clip(
                args.mose_root, args.split, vname, args.num_frames, args.image_res,
            )
        except Exception as e:
            print(f"[skip] {vname}: {e}")
            continue

        pred_ids = segmenter_predict(model, frames, out_size=gt_masks.shape[1:])
        pred_t = torch.from_numpy(pred_ids).long()
        gt_t = torch.from_numpy(gt_masks).long()
        gt_t[gt_t == 255] = 0

        veq, veq_sq, veq_rq = veq_scores(pred_t, gt_t, ignore_id=0, iou_thr=args.iou_thr)
        stq_dict = stq_en(pred_t, gt_t, ignore_id=0, iou_thr=args.iou_thr)

        per_video.append({"video": vname, "veq": veq, "stq_en": stq_dict["stq_en"],
                          "sq": stq_dict["sq"], "aq": stq_dict["aq"],
                          "n_gt_ids": int((gt_t.unique() != 0).sum().item())})
        veq_all.append(veq); stq_all.append(stq_dict["stq_en"])
        sq_all.append(stq_dict["sq"]); aq_all.append(stq_dict["aq"])
        print(f"  [{vi+1:>3d}/{len(videos)}] {vname:25s}  VEQ={veq:.3f}  STQ_EN={stq_dict['stq_en']:.3f}")

        if vi < args.save_viz:
            save_clip_viz(raw_frames, pred_ids,
                          out_dir=os.path.join(args.output_dir, "viz", vname),
                          num_traj=args.num_traj)

    aggregate = {
        "veq_mean": float(np.mean(veq_all)) if veq_all else 0.0,
        "stq_en_mean": float(np.mean(stq_all)) if stq_all else 0.0,
        "sq_mean": float(np.mean(sq_all)) if sq_all else 0.0,
        "aq_mean": float(np.mean(aq_all)) if aq_all else 0.0,
        "n_videos": len(per_video), "split": args.split, "ckpt": args.ckpt,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(aggregate, f, indent=2)
    if per_video:
        with open(os.path.join(args.output_dir, "per_video.csv"), "w") as f:
            w = csv.DictWriter(f, fieldnames=list(per_video[0].keys()))
            w.writeheader(); [w.writerow(r) for r in per_video]

    print()
    print("=" * 50)
    print(f"MOSE {args.split}: VEQ={aggregate['veq_mean']:.3f}  STQ_EN={aggregate['stq_en_mean']:.3f}")


if __name__ == "__main__":
    main()
