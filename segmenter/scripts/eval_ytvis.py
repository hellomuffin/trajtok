"""YouTube-VIS 2019 / 2021 val evaluation for the trajectory segmenter.

Uses the same Hungarian-matched VEQ + STQ_EN metrics as eval_davis.py. Loads
videos + per-frame annotations from the COCO-format YT-VIS annotation file.

Usage:
    python scripts/eval_ytvis.py \\
        --ckpt checkpoints/segmenter_filteredmixdata_all.pth \\
        --ytvis_root /path/to/ytvis2019 \\
        --ann_file /path/to/ytvis2019/instances_valid.json \\
        --output_dir results/ytvis_eval

YT-VIS layout this script expects (the standard COCO-format release):

    {ytvis_root}/
        valid/JPEGImages/<video>/<frame>.jpg
    {ann_file}                  # COCO-style JSON: videos + annotations (per-frame RLE/poly)

The annotation file has the structure
    {
      "videos": [{"id": V, "file_names": ["<video>/000.jpg", ...]}],
      "annotations": [{"video_id": V, "id": T, "segmentations": [seg_per_frame, ...]}],
      ...
    }
where each `segmentations[t]` is a COCO-RLE dict (or None for missing frames).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

# Make sibling scripts importable when this file is run directly via `python scripts/eval_ytvis.py`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_davis import (
    build_model, segmenter_predict, save_clip_viz, _IMAGENET_MEAN, _IMAGENET_STD,
)
from trajtok_segmenter.eval.seg_metric import veq_scores, stq_en


def _decode_rle(seg, H: int, W: int) -> np.ndarray:
    """Decode a COCO-RLE segmentation to a (H, W) uint8 mask. Returns zeros for None."""
    if seg is None:
        return np.zeros((H, W), dtype=np.uint8)
    try:
        from pycocotools import mask as mask_utils
    except ImportError as e:
        raise RuntimeError(
            "YT-VIS eval requires pycocotools (pip install pycocotools)."
        ) from e
    if isinstance(seg, list):
        # polygon
        rles = mask_utils.frPyObjects(seg, H, W)
        rle = mask_utils.merge(rles)
    elif isinstance(seg, dict) and isinstance(seg["counts"], list):
        rle = mask_utils.frPyObjects(seg, H, W)
    else:
        rle = seg
    return mask_utils.decode(rle)


def load_ytvis_clip(ytvis_root: str, video_info: dict, anns_for_video: list,
                    num_frames: int, image_res: int, sub_dir: str = "valid"):
    """Sample num_frames from a YT-VIS video; build per-frame instance-id mask.

    Returns (frames_tensor, gt_masks (T,H,W) int32, raw_frames (T,H,W,3) uint8).
    """
    img_root = os.path.join(ytvis_root, sub_dir, "JPEGImages")
    file_names = video_info["file_names"]                  # full paths relative to img_root
    n_frames = len(file_names)
    if n_frames >= num_frames:
        idx = np.linspace(0, n_frames - 1, num_frames).round().astype(int)
    else:
        idx = list(range(n_frames)) + [n_frames - 1] * (num_frames - n_frames)
        idx = np.asarray(idx, dtype=int)

    raw_frames, frames_tensor, gt_masks = [], [], []
    H = W = None
    for t in idx:
        img = np.asarray(Image.open(os.path.join(img_root, file_names[t])).convert("RGB"))
        raw_frames.append(img)
        if H is None:
            H, W = img.shape[:2]
        resized = np.asarray(
            Image.fromarray(img).resize((image_res, image_res), Image.BICUBIC),
            dtype=np.float32,
        ) / 255.0
        resized = (resized - _IMAGENET_MEAN) / _IMAGENET_STD
        frames_tensor.append(resized.transpose(2, 0, 1))

        # Compose per-frame instance mask: pixel value = (1 + ann_idx) — 0 reserved for bg
        frame_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
        for ai, ann in enumerate(anns_for_video):
            seg = ann["segmentations"][t] if t < len(ann["segmentations"]) else None
            if seg is None:
                continue
            inst_mask = _decode_rle(seg, img.shape[0], img.shape[1])
            frame_mask[inst_mask > 0] = ai + 1
        gt_masks.append(frame_mask)

    return (
        torch.from_numpy(np.stack(frames_tensor, 0)),
        np.stack(gt_masks, 0),
        np.stack(raw_frames, 0),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--ytvis_root", required=True)
    parser.add_argument("--ann_file", required=True)
    parser.add_argument("--sub_dir", default="valid")
    parser.add_argument("--output_dir", default="results/ytvis_eval")
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

    print(f"[ytvis] loading annotations from {args.ann_file}")
    with open(args.ann_file) as f:
        coco = json.load(f)
    videos = coco["videos"]
    anns_by_vid = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_vid[a["video_id"]].append(a)

    if args.max_videos > 0:
        videos = videos[: args.max_videos]
    print(f"[ytvis] {len(videos)} videos, {args.num_frames} frames each")

    per_video, veq_all, stq_all, sq_all, aq_all = [], [], [], [], []
    for vi, vinfo in enumerate(videos):
        vname = vinfo["file_names"][0].split("/")[0]
        anns = anns_by_vid.get(vinfo["id"], [])
        if not anns:
            print(f"  [skip] {vname}: no annotations")
            continue
        try:
            frames, gt_masks, raw_frames = load_ytvis_clip(
                args.ytvis_root, vinfo, anns, args.num_frames, args.image_res, sub_dir=args.sub_dir,
            )
        except Exception as e:
            print(f"  [skip] {vname}: {e}")
            continue

        pred_ids = segmenter_predict(model, frames, out_size=gt_masks.shape[1:])
        pred_t = torch.from_numpy(pred_ids).long()
        gt_t = torch.from_numpy(gt_masks).long()

        veq, _, _ = veq_scores(pred_t, gt_t, ignore_id=0, iou_thr=args.iou_thr)
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
        "n_videos": len(per_video), "ckpt": args.ckpt,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(aggregate, f, indent=2)
    if per_video:
        with open(os.path.join(args.output_dir, "per_video.csv"), "w") as f:
            w = csv.DictWriter(f, fieldnames=list(per_video[0].keys()))
            w.writeheader(); [w.writerow(r) for r in per_video]

    print()
    print("=" * 50)
    print(f"YT-VIS: VEQ={aggregate['veq_mean']:.3f}  STQ_EN={aggregate['stq_en_mean']:.3f}")


if __name__ == "__main__":
    main()
