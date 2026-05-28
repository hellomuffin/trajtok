"""Zero-shot ImageNet-1k classification for TrajViT-v2.

Uses 80-prompt-ensemble CLIP-style classification (per-class templates averaged).

Usage:
    python scripts/eval_imagenet_zeroshot.py \\
        --ckpt checkpoints/trajvitv2_panda.pth \\
        --imagenet_val /path/to/imagenet/val \\
        [--output_dir results/imagenet_zs]

We expect `imagenet_val` to be the standard ImageFolder layout:
    {imagenet_val}/<wnid>/<image>.JPEG
A bundled `imagenet_class_names.json` maps wnid → class name(s).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from PIL import Image

from trajtok_segmenter.model.model_pretrain import SegmentCLIP
from trajtok_segmenter.text.tokenization_bert import BertTokenizer


# A small representative subset of OpenAI CLIP's 80 prompt templates.
# Full list at https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
DEFAULT_TEMPLATES = [
    "a photo of a {}.",
    "a bad photo of a {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image_tensor(path: str, image_res: int) -> torch.Tensor:
    img = np.asarray(Image.open(path).convert("RGB"))
    arr = np.asarray(Image.fromarray(img).resize((image_res, image_res), Image.BICUBIC), dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1))   # (3, H, W)


@torch.no_grad()
def build_text_classifier(model: SegmentCLIP, tokenizer, classnames: List[str], templates: List[str]) -> torch.Tensor:
    """Build per-class text embeddings by averaging across prompt templates."""
    device = next(model.parameters()).device
    zeroshot_weights = []
    for class_idx, classname in enumerate(classnames):
        prompts = [t.format(classname) for t in templates]
        tok = tokenizer(prompts, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
        text_embeds, pooled = model.encode_text(tok)
        feat = model.text_proj(pooled[:, 0])
        feat = torch.nn.functional.normalize(feat, dim=-1)
        feat = feat.mean(dim=0)
        feat = feat / feat.norm()
        zeroshot_weights.append(feat)
        if class_idx % 100 == 0:
            print(f"  [text classifier] {class_idx}/{len(classnames)}")
    return torch.stack(zeroshot_weights, 1)            # (D, C)


@torch.no_grad()
def evaluate_imagenet(model: SegmentCLIP, classifier_weights: torch.Tensor, imagenet_val: str,
                       classnames_by_wnid: Dict[str, int], image_res: int, batch_size: int = 64):
    """Returns top-1 + top-5 accuracy."""
    device = next(model.parameters()).device
    wnids = sorted(classnames_by_wnid.keys())
    correct_top1, correct_top5, n_total = 0, 0, 0

    for wnid in wnids:
        wnid_dir = os.path.join(imagenet_val, wnid)
        if not os.path.isdir(wnid_dir):
            continue
        gt_idx = classnames_by_wnid[wnid]
        files = sorted(f for f in os.listdir(wnid_dir) if f.endswith((".JPEG", ".jpg", ".png")))
        for i in range(0, len(files), batch_size):
            batch_files = files[i : i + batch_size]
            imgs = torch.stack([load_image_tensor(os.path.join(wnid_dir, f), image_res) for f in batch_files], 0)
            imgs = imgs.to(device)
            # Treat images as T=1 video clips for the trajvitv2 forward.
            video = imgs.unsqueeze(1)                                        # (B, T=1, 3, H, W)
            T, H, W = video.shape[1], video.shape[3], video.shape[4]
            dummy_mask = torch.zeros(video.shape[0], T, H // 4, W // 4, dtype=torch.long, device=device)
            dummy_graph = torch.zeros(video.shape[0], 128, T, dtype=torch.long, device=device)
            _, pooled = model.encode_image((video, dummy_mask, dummy_graph))
            feat = model.vision_proj(pooled[:, 0])
            feat = torch.nn.functional.normalize(feat, dim=-1)
            logits = 100.0 * feat @ classifier_weights                       # (B, C)
            preds = logits.topk(5, dim=-1).indices.cpu().numpy()
            for p in preds:
                if p[0] == gt_idx:
                    correct_top1 += 1
                if gt_idx in p:
                    correct_top5 += 1
            n_total += len(batch_files)
        print(f"  [imagenet] {wnid}  ({n_total} eval'd so far)  top1={correct_top1/max(n_total,1)*100:.2f}")

    return correct_top1 / max(n_total, 1), correct_top5 / max(n_total, 1), n_total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--imagenet_val", required=True, help="Path to ImageFolder val/")
    parser.add_argument("--class_names", required=True, help="JSON: {wnid: 'class name'} (or wnid: [aliases])")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output_dir", default="results/imagenet_zs")
    parser.add_argument("--image_res", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    if args.config is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.config = os.path.join(here, "..", "configs", "pretrain.yaml")
    os.makedirs(args.output_dir, exist_ok=True)

    # Build model + tokenizer
    with open(args.config) as f:
        cfg = edict(yaml.safe_load(f))
    tokenizer = BertTokenizer.from_pretrained(cfg.text_encoder)
    model = SegmentCLIP(config=cfg, tokenizer=tokenizer).cuda().eval()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "model" in sd:
        sd = sd["model"]
    msg = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")

    # Load class name mapping
    with open(args.class_names) as f:
        wnid_to_name = json.load(f)
    # Each value may be a string or list; take first if list.
    classnames = []
    classnames_by_wnid = {}
    for wnid in sorted(wnid_to_name.keys()):
        name = wnid_to_name[wnid]
        if isinstance(name, list):
            name = name[0]
        classnames_by_wnid[wnid] = len(classnames)
        classnames.append(name)

    print(f"[imagenet] {len(classnames)} classes, building text classifier...")
    classifier_weights = build_text_classifier(model, tokenizer, classnames, DEFAULT_TEMPLATES)

    print(f"[imagenet] evaluating {args.imagenet_val} ...")
    top1, top5, n = evaluate_imagenet(
        model, classifier_weights, args.imagenet_val, classnames_by_wnid,
        image_res=args.image_res, batch_size=args.batch_size,
    )

    out = {"top1": top1 * 100, "top5": top5 * 100, "n_eval": n,
           "n_templates": len(DEFAULT_TEMPLATES), "ckpt": args.ckpt}
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("=" * 50)
    print(f"ImageNet zero-shot:  top1={top1*100:.2f}  top5={top5*100:.2f}  (n={n})")


if __name__ == "__main__":
    main()
