"""Zero-shot text→video retrieval evaluation for TrajViT-v2.

Reproduces the paper's video retrieval numbers on MSR-VTT (1k-A split),
ActivityNet-captions val, and DiDeMo val.

Usage:
    python scripts/eval_video_retrieval.py \\
        --ckpt checkpoints/trajvitv2_panda.pth \\
        --dataset msrvtt \\
        --json /path/to/msrvtt_test_1kA.json \\
        --video_root /path/to/msrvtt/videos \\
        --output_dir results/msrvtt_eval

Annotation JSON format (per record):
    {"video": "path/to/clip.mp4", "caption": "a dog runs across the lawn", ...}

We compute symmetric retrieval (T→V and V→T) and report R@1, R@5, R@10, MedianR.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from PIL import Image

# Reuse the segmenter package's SegmentCLIP wrapper.
from trajtok_segmenter.model.model_pretrain import SegmentCLIP
from trajtok_segmenter.text.tokenization_bert import BertTokenizer


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_clip_uniform(path: str, num_frames: int, image_res: int) -> torch.Tensor:
    """Read `num_frames` evenly-sampled, resized + normalised frames from a video.

    Returns: (num_frames, 3, image_res, image_res) float32 tensor.
    """
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(path, num_threads=1)
    n = len(vr)
    idx = np.linspace(0, max(n - 1, 0), num_frames).round().astype(int)
    frames = vr.get_batch(idx).asnumpy()                         # (T, H, W, 3) uint8

    resized = np.stack([
        np.asarray(Image.fromarray(f).resize((image_res, image_res), Image.BICUBIC), dtype=np.float32)
        for f in frames
    ], 0) / 255.0
    resized = (resized - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(resized.transpose(0, 3, 1, 2))      # (T, 3, H, W)


def load_test_annotations(json_path: str, video_root: str) -> List[Dict]:
    with open(json_path) as f:
        data = json.load(f)
    out = []
    for rec in data:
        # Most retrieval annotation files store either {video, caption} or
        # {video_id, captions: [...]}. Handle both.
        video = rec.get("video") or rec.get("video_path") or rec.get("video_id")
        if not video:
            continue
        full = video if os.path.isabs(video) else os.path.join(video_root, video)
        captions = rec.get("caption") or rec.get("captions") or rec.get("text")
        if isinstance(captions, str):
            captions = [captions]
        for cap in captions:
            out.append({"video": full, "caption": cap})
    return out


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #

@torch.no_grad()
def encode_videos(model: SegmentCLIP, tokenizer, video_paths: List[str], num_frames: int, image_res: int,
                  batch_size: int = 4) -> torch.Tensor:
    """Encode each unique video to a normalised image embedding."""
    device = next(model.parameters()).device
    embeds = []
    for i in range(0, len(video_paths), batch_size):
        batch = video_paths[i : i + batch_size]
        clips = torch.stack([
            load_clip_uniform(p, num_frames, image_res) for p in batch
        ], 0).to(device)                                          # (B, T, 3, H, W)
        # Dummy mask/graph (segmenter doesn't need real ones for inference).
        T, H, W = clips.shape[1], clips.shape[3], clips.shape[4]
        dummy_mask = torch.zeros(len(batch), T, H // 4, W // 4, dtype=torch.long, device=device)
        dummy_graph = torch.zeros(len(batch), 128, T, dtype=torch.long, device=device)
        # encode_image returns (image_embeds, pooled_image_embeds).
        image_embeds, pooled = model.encode_image((clips, dummy_mask, dummy_graph))
        # Project to contrastive dim, L2-normalise.
        feat = model.vision_proj(pooled[:, 0])                    # (B, embed_dim)
        feat = torch.nn.functional.normalize(feat, dim=-1)
        embeds.append(feat.cpu())
        if (i // batch_size) % 10 == 0:
            print(f"  [video] {i + len(batch)}/{len(video_paths)}")
    return torch.cat(embeds, 0)                                    # (N_videos, D)


@torch.no_grad()
def encode_texts(model: SegmentCLIP, tokenizer, captions: List[str], batch_size: int = 64) -> torch.Tensor:
    """Encode captions to normalised text embeddings."""
    device = next(model.parameters()).device
    embeds = []
    for i in range(0, len(captions), batch_size):
        batch = captions[i : i + batch_size]
        tok = tokenizer(batch, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
        text_embeds, pooled = model.encode_text(tok)
        feat = model.text_proj(pooled[:, 0])
        feat = torch.nn.functional.normalize(feat, dim=-1)
        embeds.append(feat.cpu())
    return torch.cat(embeds, 0)


# --------------------------------------------------------------------------- #
# Retrieval scoring
# --------------------------------------------------------------------------- #

def compute_retrieval_metrics(sim: torch.Tensor, gt_idx: List[int]) -> Dict[str, float]:
    """sim: (N_texts, N_videos) similarity matrix. gt_idx[t] = ground-truth video idx for text t."""
    sim_np = sim.numpy()
    ranks = []
    for t, gt in enumerate(gt_idx):
        order = np.argsort(-sim_np[t])
        rank = int(np.where(order == gt)[0][0])
        ranks.append(rank)
    ranks = np.asarray(ranks)
    return {
        "R@1": float(np.mean(ranks < 1) * 100),
        "R@5": float(np.mean(ranks < 5) * 100),
        "R@10": float(np.mean(ranks < 10) * 100),
        "MedianR": float(np.median(ranks) + 1),
        "MeanR": float(np.mean(ranks) + 1),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def build_model(ckpt_path: str, config_path: str):
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = edict(cfg_dict)
    tokenizer = BertTokenizer.from_pretrained(cfg.text_encoder)
    model = SegmentCLIP(config=cfg, tokenizer=tokenizer).cuda().eval()

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in sd:
        sd = sd["model"]
    msg = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    return model, tokenizer, cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", required=True, choices=["msrvtt", "activitynet", "didemo"])
    parser.add_argument("--json", required=True, help="Annotation JSON path")
    parser.add_argument("--video_root", required=True, help="Root prefix for video paths in the JSON")
    parser.add_argument("--config", default=None, help="Defaults to configs/pretrain.yaml")
    parser.add_argument("--output_dir", default="results/retrieval_eval")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--image_res", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    if args.config is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.config = os.path.join(here, "..", "configs", "pretrain.yaml")
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer, _ = build_model(args.ckpt, args.config)
    records = load_test_annotations(args.json, args.video_root)
    print(f"[{args.dataset}] {len(records)} text-video pairs")

    # Dedupe videos so we encode each unique clip once.
    unique_videos: List[str] = []
    video_to_idx: Dict[str, int] = {}
    for r in records:
        if r["video"] not in video_to_idx:
            video_to_idx[r["video"]] = len(unique_videos)
            unique_videos.append(r["video"])
    gt_idx = [video_to_idx[r["video"]] for r in records]
    captions = [r["caption"] for r in records]

    print(f"  unique videos: {len(unique_videos)}, captions: {len(captions)}")
    text_feats = encode_texts(model, tokenizer, captions, batch_size=64)         # (N_caps, D)
    video_feats = encode_videos(
        model, tokenizer, unique_videos,
        num_frames=args.num_frames, image_res=args.image_res, batch_size=args.batch_size,
    )                                                                            # (N_vids, D)

    # T→V similarity
    sim_t2v = text_feats @ video_feats.T                                          # (N_caps, N_vids)
    t2v = compute_retrieval_metrics(sim_t2v, gt_idx)

    # V→T (per video, pick the best caption rank; group captions by video)
    caps_by_video = defaultdict(list)
    for caption_idx, video_idx in enumerate(gt_idx):
        caps_by_video[video_idx].append(caption_idx)
    sim_v2t = video_feats @ text_feats.T                                          # (N_vids, N_caps)
    ranks_v2t = []
    sim_v2t_np = sim_v2t.numpy()
    for v_idx, cap_idxs in caps_by_video.items():
        order = np.argsort(-sim_v2t_np[v_idx])
        # Best rank across the captions that genuinely describe this video.
        best = min(int(np.where(order == c)[0][0]) for c in cap_idxs)
        ranks_v2t.append(best)
    ranks_v2t = np.asarray(ranks_v2t)
    v2t = {
        "R@1": float(np.mean(ranks_v2t < 1) * 100),
        "R@5": float(np.mean(ranks_v2t < 5) * 100),
        "R@10": float(np.mean(ranks_v2t < 10) * 100),
        "MedianR": float(np.median(ranks_v2t) + 1),
        "MeanR": float(np.mean(ranks_v2t) + 1),
    }

    out = {"dataset": args.dataset, "t2v": t2v, "v2t": v2t,
           "num_pairs": len(records), "num_videos": len(unique_videos),
           "num_frames": args.num_frames, "image_res": args.image_res, "ckpt": args.ckpt}
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    print()
    print("=" * 70)
    print(f"{args.dataset.upper()}  T→V:  R@1={t2v['R@1']:.2f}  R@5={t2v['R@5']:.2f}  R@10={t2v['R@10']:.2f}  MedR={t2v['MedianR']:.1f}")
    print(f"{args.dataset.upper()}  V→T:  R@1={v2t['R@1']:.2f}  R@5={v2t['R@5']:.2f}  R@10={v2t['R@10']:.2f}  MedR={v2t['MedianR']:.1f}")


if __name__ == "__main__":
    main()
