# TrajViT-v2

📄 **Paper:** [arXiv:2602.22779](https://arxiv.org/abs/2602.22779)

**TrajViT-v2** wraps the trajectory segmenter with a vision transformer and a
CLIP-style image-text contrastive objective. The trajectory tokens (128 per
clip from the segmenter) are contextualised by a ViT-Large transformer, then
the [CLS] feature is contrasted against BERT text embeddings via InfoNCE.

```
video / image
     │
     ↓
SimpleSegmenter (DINOv3 + perceiver)         →  trajectory tokens  z  (K=128, D=512)
     │
     ↓
ViT-Large transformer over z                  →  contextualised z' (K+1=129, D=512)
     │
     ↓
[CLS]-pool → vision_proj (linear) ────────────→ image embedding for InfoNCE
                                                  ↕
                                            text embedding ← BERT(caption)
```

This package **reuses the segmenter package's code** ([`../segmenter`](../segmenter))
— the same training loop (`trajtok_segmenter.train.pretrain`) supports both
modes; we just flip `vit_type=simpletrajvitv2` and enable `loss_weight.itc=1.0`.

The trajvitv2-specific bits in this directory are:
- [`configs/pretrain.yaml`](configs/pretrain.yaml) — contrastive-mode config
- [`scripts/train.sh`](scripts/train.sh) — launcher (warm-starts the segmenter from the released ckpt)
- [`scripts/eval_video_retrieval.py`](scripts/eval_video_retrieval.py) — MSR-VTT / ActivityNet / DiDeMo retrieval
- [`scripts/eval_imagenet_zeroshot.py`](scripts/eval_imagenet_zeroshot.py) — ImageNet-1k zero-shot classification
- [`scripts/download_ckpt.py`](scripts/download_ckpt.py) — HF Hub fetch for the released ckpt

## Quickstart

### 1. Install

```bash
# Install the segmenter package (provides shared model code + training loop)
pip install -e ../segmenter
# Then this package
pip install -e .
```

### 2. Download the released checkpoint

```bash
python scripts/download_ckpt.py
# → ./checkpoints/trajvitv2_filteredmixdata_new.pth
```

**⚠ Caveat**: this checkpoint is trained on a **small image+video mixture
(~1.3 M pairs: `big_image_new` + `big_video_new`)** for 20 epochs as a
proof-of-concept of the trajectory-CLIP architecture. It will **not match
SOTA video-text retrieval numbers** from papers trained on 100 M+ clips.
Use it as a *starting point* for fine-tuning or as a sanity-check baseline.

### 3. Zero-shot video retrieval on MSR-VTT

```bash
python scripts/eval_video_retrieval.py \
  --ckpt checkpoints/trajvitv2_filteredmixdata_new.pth \
  --dataset msrvtt \
  --json /path/to/MSRVTT/msrvtt_test_1kA.json \
  --video_root /path/to/MSRVTT/videos
# → results/retrieval_eval/metrics.json
```

### 4. Zero-shot ImageNet classification

```bash
python scripts/eval_imagenet_zeroshot.py \
  --ckpt checkpoints/trajvitv2_filteredmixdata_new.pth \
  --imagenet_val /path/to/imagenet/val \
  --class_names examples/imagenet_class_names.json
# → results/imagenet_zs/metrics.json
```

## Training

### Data

The released checkpoint was trained on **`filteredmixdata_new`** — a small
~1.3 M-pair filtered image+video mixture:

| Source | Samples | Type |
|---|---|---|
| `big_image_new` | ~300 K | filtered image-caption pairs |
| `big_video_new` | ~1 M | filtered video-caption pairs |

These are user-curated mixtures internal to the paper authors; the
configuration also supports public alternatives (`panda_4m`, `coco`, `cc3m`,
etc.) — see the `available_corpus` block in
[`configs/pretrain.yaml`](configs/pretrain.yaml) for the full list. Provide a
JSON manifest of the form:

```json
[
  {"video": "path/to/clip.mp4", "caption": "a dog runs through tall grass"},
  ...
]
```

### Launch

```bash
TRAJTOK_DATA_ROOT=/path/to/data \
TRAJTOK_OUTPUT_DIR=/path/to/results \
TRAJTOK_DINOV3_ROOT=/path/to/dinov3 \
bash scripts/train.sh \
  --ngpus 8 \
  --seg_ckpt /path/to/segmenter_filteredmixdata_all.pth   # ← warm-start from released segmenter
  --train_corpus filteredmixdata_new \
  --exp_name myrun \
  --epoch 20 --log_wandb
```

Warm-starting from the released segmenter is **highly recommended** — without
it the model has to learn trajectory grouping from scratch alongside the
contrastive objective, which is much slower to converge.

## Released-checkpoint quality

The published checkpoint is trained on ~1.3 M image+video pairs (vs. 100 M –
5 B for production CLIP/SigLIP), so video-retrieval and zero-shot
classification numbers will be substantially below state-of-the-art trained
on web-scale data. Run the eval scripts above against your own data to see
the exact numbers on your benchmark / split combination. We omit pinned
numbers here because they depend on annotation file versions
(MSR-VTT 1k-A vs 1k-B, ActivityNet-Captions test vs val, etc.) and would
encourage cargo-cult reproduction.

Use this checkpoint as: (a) a sanity-check baseline for trajectory-CLIP
architectures, or (b) a starting point for fine-tuning on larger or
domain-specific data.

## Repository layout

```
trajvitv2/
├── README.md                       (this file)
├── pyproject.toml
├── configs/
│   └── pretrain.yaml              (contrastive training config)
├── scripts/
│   ├── train.sh                   (launch trajvitv2 training; warm-starts from released segmenter)
│   ├── download_ckpt.py           (HF Hub fetch)
│   ├── eval_video_retrieval.py    (MSR-VTT / ActivityNet / DiDeMo)
│   └── eval_imagenet_zeroshot.py  (ImageNet-1k via 80-prompt ensemble)
├── examples/                       (place to bundle small data: class names, demo clips)
└── trajtok_trajvitv2/
    ├── eval/                       (extension hooks; mostly empty — eval scripts in scripts/)
    └── viz/                        (extension hooks for custom visualisation pipelines)
```

## Caveats

- **Small-scale ckpt** — see Step 2 above.
- **Reuses segmenter package**: training imports `trajtok_segmenter.train.pretrain`
  with `vit_type=simpletrajvitv2`. If you customise heavily, fork that
  package rather than monkey-patching.
- **In-loop video retrieval evals** that the segmenter package's
  `retrieval_utils` once supported are disabled in the OSS config because
  they depend on additional benchmark data layouts. Use the standalone
  `scripts/eval_*.py` instead.

## Citation

See [`../README.md`](../README.md).
