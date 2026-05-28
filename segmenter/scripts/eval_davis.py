"""DAVIS-2017 val evaluation for the trajectory segmenter.

Evaluates class-agnostic spatio-temporal grouping quality with the same metrics
used in the paper:

  * **VEQ** — per-clip Video-Equivalent Quality
              (Hungarian-matched IoU pooled across frames)
  * **STQ_EN** — Spatio-Temporal Quality
              (geometric mean of segmentation-quality + association-quality)

Both metrics are class-agnostic: the segmenter predicts arbitrary trajectory
IDs (0..K-1) and we Hungarian-match them to GT instance IDs at IoU >= 0.5
before scoring.

Usage:
    python scripts/eval_davis.py \\
        --ckpt checkpoints/segmenter_filteredmixdata_all.pth \\
        --davis_root /path/to/DAVIS \\
        --output_dir results/davis_eval \\
        --num_frames 8 \\
        [--save_viz 5]                # save overlay for the first N videos
        [--num_traj 128]              # K trajectory tokens (must match ckpt)
        [--image_res 224]             # input resolution
        [--max_videos 0]              # 0 = all val videos

DAVIS-2017 layout this script expects (the standard release):

    {davis_root}/
        ImageSets/2017/val.txt          # one video name per line
        JPEGImages/480p/<video>/*.jpg
        Annotations/480p/<video>/*.png  # palette PNG, pixel value = instance id

Outputs (under --output_dir):
    metrics.json      # aggregate VEQ + STQ_EN across all videos
    per_video.csv     # per-video metric breakdown
    viz/<video>/      # (if --save_viz N > 0) overlays for the first N videos
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict
from PIL import Image

from trajtok_segmenter.model.segmenter import SimpleSegmenter
from trajtok_segmenter.eval.seg_metric import veq_scores, stq_en


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- #
# DAVIS loader
# --------------------------------------------------------------------------- #

def list_davis_val_videos(davis_root: str) -> List[str]:
    """Return the canonical ordered list of DAVIS-2017 val video names."""
    val_txt = os.path.join(davis_root, "ImageSets", "2017", "val.txt")
    if not os.path.isfile(val_txt):
        raise FileNotFoundError(
            f"Expected {val_txt}. Make sure --davis_root points at the standard "
            "DAVIS-2017 release (with ImageSets/2017/val.txt, JPEGImages/480p/, "
            "Annotations/480p/)."
        )
    with open(val_txt) as f:
        return [line.strip() for line in f if line.strip()]


def load_davis_clip(
    davis_root: str,
    video_name: str,
    num_frames: int,
    image_res: int,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Sample `num_frames` evenly from a DAVIS video and return:
        frames_tensor    (num_frames, 3, image_res, image_res)  float CHW, normalised
        gt_masks         (num_frames, H_gt, W_gt) int32         per-pixel instance id (0 = bg)
        raw_frames       (num_frames, H_gt, W_gt, 3) uint8       for visualisation
    """
    img_dir = os.path.join(davis_root, "JPEGImages", "480p", video_name)
    ann_dir = os.path.join(davis_root, "Annotations", "480p", video_name)

    img_files = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
    if not img_files:
        raise FileNotFoundError(f"No .jpg files in {img_dir}")

    # Uniformly sample num_frames from the clip.
    if len(img_files) >= num_frames:
        idx = np.linspace(0, len(img_files) - 1, num_frames).round().astype(int)
    else:
        # Pad by repeating the last frame.
        idx = list(range(len(img_files))) + [len(img_files) - 1] * (num_frames - len(img_files))
        idx = np.asarray(idx, dtype=int)
    chosen = [img_files[i] for i in idx]

    raw_frames, frames_tensor, gt_masks = [], [], []
    for fname in chosen:
        img = np.asarray(Image.open(os.path.join(img_dir, fname)).convert("RGB"))
        raw_frames.append(img)

        # Resize + normalise for model input
        resized = np.asarray(
            Image.fromarray(img).resize((image_res, image_res), Image.BICUBIC),
            dtype=np.float32,
        ) / 255.0
        resized = (resized - _IMAGENET_MEAN) / _IMAGENET_STD
        frames_tensor.append(resized.transpose(2, 0, 1))     # (3, H, W)

        # GT mask: palette PNG with pixel values = instance ids
        mask_fname = fname.replace(".jpg", ".png")
        mask = np.asarray(Image.open(os.path.join(ann_dir, mask_fname)), dtype=np.int32)
        gt_masks.append(mask)

    return (
        torch.from_numpy(np.stack(frames_tensor, 0)),       # (T, 3, H, W)
        np.stack(gt_masks, 0),                              # (T, H_gt, W_gt)
        np.stack(raw_frames, 0),                            # (T, H_gt, W_gt, 3)
    )


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

@torch.no_grad()
def segmenter_predict(
    model: SimpleSegmenter,
    frames: torch.Tensor,        # (T, 3, H, W)
    out_size: Tuple[int, int],   # (H_gt, W_gt) — predicted_label upsampled to this for matching
) -> np.ndarray:
    """Run the segmenter on a clip and return per-pixel trajectory IDs at `out_size`.

    Returns: int32 array of shape (T, H_gt, W_gt) with IDs in [0, num_traj).
    """
    device = next(model.parameters()).device
    x = frames.unsqueeze(0).to(device)                       # (1, T, 3, H, W)
    logits = model(x)                                        # (1, T*h*w, K)
    T = frames.shape[0]
    h = w = int(round((logits.shape[1] / T) ** 0.5))
    # Argmax over K -> (1, T, h, w)
    pred = logits.argmax(dim=-1).reshape(1, T, h, w).float()
    # Nearest-neighbour upsample to GT resolution for IoU computation
    pred_up = F.interpolate(pred, size=out_size, mode="nearest")
    return pred_up.squeeze(0).cpu().numpy().astype(np.int32)


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #

def _palette(K: int) -> np.ndarray:
    """Distinct colors for K trajectory IDs."""
    import colorsys
    rng = np.random.RandomState(0)
    hues = rng.permutation(K) / K
    rgb = np.array([colorsys.hsv_to_rgb(h, 0.85, 0.95) for h in hues])
    return (rgb * 255).astype(np.uint8)


def save_clip_viz(
    raw_frames: np.ndarray,                # (T, H, W, 3) uint8
    pred_ids: np.ndarray,                  # (T, H, W) int32
    out_dir: str,
    num_traj: int,
    alpha: float = 0.55,
):
    """Dump per-frame mask + overlay for one clip."""
    os.makedirs(out_dir, exist_ok=True)
    palette = _palette(num_traj)
    for t in range(raw_frames.shape[0]):
        color = palette[pred_ids[t] % num_traj]
        overlay = (alpha * color + (1 - alpha) * raw_frames[t]).astype(np.uint8)
        Image.fromarray(overlay).save(os.path.join(out_dir, f"frame_{t:03d}_overlay.png"))
        Image.fromarray(color).save(os.path.join(out_dir, f"frame_{t:03d}_mask.png"))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def build_model(ckpt_path: str, config_path: str, num_traj: int) -> SimpleSegmenter:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    traj_cfg = edict(cfg["traj_model"])
    bb_cfg = edict(cfg["backbone"])
    per_cfg = edict(cfg["perceiver"])
    traj_cfg.num_traj = num_traj

    model = SimpleSegmenter(
        config=traj_cfg, backbone_config=bb_cfg, perceiver_config=per_cfg,
        high_res=False,
    ).cuda().eval()

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in sd:
        sd = sd["model"]
    # Strip outer SegmentWrapper prefix if present.
    sd = {k[len("vision_encoder."):] if k.startswith("vision_encoder.") else k: v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--davis_root", required=True)
    parser.add_argument("--output_dir", default="results/davis_eval")
    parser.add_argument("--config", default=None, help="Defaults to configs/pretrain.yaml")
    parser.add_argument("--num_frames", type=int, default=8, help="Frames sampled per clip")
    parser.add_argument("--image_res", type=int, default=224, help="Input resolution to segmenter")
    parser.add_argument("--num_traj", type=int, default=128, help="K trajectories (must match ckpt)")
    parser.add_argument("--iou_thr", type=float, default=0.5, help="Hungarian match IoU threshold")
    parser.add_argument("--save_viz", type=int, default=0, help="Save overlays for the first N videos (0 = none)")
    parser.add_argument("--max_videos", type=int, default=0, help="Cap videos evaluated (0 = all)")
    args = parser.parse_args()

    if args.config is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.config = os.path.join(here, "..", "configs", "pretrain.yaml")

    os.makedirs(args.output_dir, exist_ok=True)

    model = build_model(args.ckpt, args.config, args.num_traj)
    videos = list_davis_val_videos(args.davis_root)
    if args.max_videos > 0:
        videos = videos[: args.max_videos]
    print(f"[davis] evaluating {len(videos)} videos at {args.num_frames} frames each")

    per_video: List[Dict] = []
    veq_all, sq_all, aq_all, stq_all = [], [], [], []

    for vi, vname in enumerate(videos):
        try:
            frames, gt_masks, raw_frames = load_davis_clip(
                args.davis_root, vname, args.num_frames, args.image_res,
            )
        except Exception as e:
            print(f"[skip] {vname}: load failed — {e}")
            continue

        H_gt, W_gt = gt_masks.shape[1:]
        pred_ids = segmenter_predict(model, frames, out_size=(H_gt, W_gt))

        # Compute per-clip VEQ + STQ (ignore_id=0 since DAVIS uses 0 = background;
        # DAVIS palette also uses 255 as void in some annotations, treat both as ignore).
        pred_t = torch.from_numpy(pred_ids).long()
        gt_t = torch.from_numpy(gt_masks).long()
        # Mask out void pixels (255) in GT so they don't count for any track.
        gt_t[gt_t == 255] = 0

        veq, veq_sq, veq_rq = veq_scores(pred_t, gt_t, ignore_id=0, iou_thr=args.iou_thr)
        stq_dict = stq_en(pred_t, gt_t, ignore_id=0, iou_thr=args.iou_thr)

        per_video.append({
            "video": vname,
            "veq": veq,
            "veq_sq": veq_sq,
            "veq_rq": veq_rq,
            "stq_en": stq_dict["stq_en"],
            "sq": stq_dict["sq"],
            "aq": stq_dict["aq"],
            "n_gt_ids": int((gt_t.unique() != 0).sum().item()),
        })
        veq_all.append(veq)
        sq_all.append(stq_dict["sq"])
        aq_all.append(stq_dict["aq"])
        stq_all.append(stq_dict["stq_en"])
        print(f"  [{vi+1:>2d}/{len(videos)}] {vname:25s}  VEQ={veq:.3f}  STQ_EN={stq_dict['stq_en']:.3f}  (n_gt={per_video[-1]['n_gt_ids']})")

        if vi < args.save_viz:
            save_clip_viz(
                raw_frames, pred_ids,
                out_dir=os.path.join(args.output_dir, "viz", vname),
                num_traj=args.num_traj,
            )

    aggregate = {
        "veq_mean": float(np.mean(veq_all)) if veq_all else 0.0,
        "stq_en_mean": float(np.mean(stq_all)) if stq_all else 0.0,
        "sq_mean": float(np.mean(sq_all)) if sq_all else 0.0,
        "aq_mean": float(np.mean(aq_all)) if aq_all else 0.0,
        "n_videos": len(per_video),
        "num_frames_per_clip": args.num_frames,
        "image_res": args.image_res,
        "num_traj": args.num_traj,
        "iou_thr": args.iou_thr,
        "ckpt": args.ckpt,
    }

    # Persist outputs
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(aggregate, f, indent=2)
    with open(os.path.join(args.output_dir, "per_video.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_video[0].keys()))
        writer.writeheader()
        for row in per_video:
            writer.writerow(row)

    print()
    print("=" * 50)
    print(f"DAVIS-2017 val: VEQ={aggregate['veq_mean']:.3f}  STQ_EN={aggregate['stq_en_mean']:.3f}")
    print(f"  (SQ={aggregate['sq_mean']:.3f}, AQ={aggregate['aq_mean']:.3f})")
    print(f"  over {aggregate['n_videos']} videos")
    print(f"Wrote {args.output_dir}/metrics.json and per_video.csv")


if __name__ == "__main__":
    main()
