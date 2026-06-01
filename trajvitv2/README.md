# TrajViT-v2

📄 **Paper:** [arXiv:2602.22779](https://arxiv.org/abs/2602.22779)

**TrajViT-v2** wraps the trajectory segmenter with a vision transformer and a
CLIP-style image-text contrastive objective. The trajectory tokens (128 per
clip from the segmenter) are contextualised by a ViT-Large transformer, then
the [CLS] feature is contrasted against BERT text embeddings via InfoNCE.

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

Training data is **not bundled** with this release. The contrastive head
needs `{image|video, caption}` pairs; trajectory masks/graphs are also required
since the segmenter is co-trained.

#### Expected layout

```
${TRAJTOK_DATA_ROOT}/
├── metadata/
│   ├── my_images.json
│   └── my_videos.json
├── images/
│   ├── img_001.jpg
│   └── img_001_mask.npz          ← sidecar mask, key="arr_0", shape (H,W)
└── videos/
    ├── clip_001.mp4
    ├── clip_001_mask.npz         ← per-frame masks, key="arr_0", shape (T,H,W)
    └── clip_001_graph.npz        ← trajectory graph, key="tensor", shape (N_instances,T)
```

#### Manifest schema

```json
// my_images.json
[
  {"image": "/abs/or/rel/path/to/images/img_001.jpg", "caption": "a dog runs through tall grass"},
  ...
]

// my_videos.json
[
  {"video": "/abs/or/rel/path/to/videos/clip_001.mp4", "caption": "a chef chops onions on a wooden board"},
  ...
]
```

The full schema (sidecar mask/graph conventions, how `image_root_prefix` is
joined with paths, optional explicit `"mask"`/`"graph"` keys) is documented in
[`../segmenter/README.md#data-preparation`](../segmenter/README.md#data-preparation).

#### Register your corpus

Edit [`configs/pretrain.yaml`](configs/pretrain.yaml) → add to `available_corpus`:

```yaml
available_corpus:
  my_images: ['${anno_root_filtered}/my_images.json', '/', image]
  my_videos: ['${anno_root_filtered}/my_videos.json', '/', video]
  my_mix:
    - ${available_corpus.my_images}
    - ${available_corpus.my_videos}
```

Then launch with `--train_corpus my_mix` (see "Launch" below).

### Launch

```bash
TRAJTOK_DATA_ROOT=/path/to/data \
TRAJTOK_OUTPUT_DIR=/path/to/results \
TRAJTOK_DINOV3_ROOT=/path/to/dinov3 \
bash scripts/train.sh \
  --ngpus 8 \
  --seg_ckpt /path/to/segmenter_filteredmixdata_all.pth   # ← warm-start from released segmenter
  --train_corpus my_mix \
  --exp_name myrun \
  --epoch 20 --log_wandb
```

Warm-starting from the released segmenter is **highly recommended** — without
it the model has to learn trajectory grouping from scratch alongside the
contrastive objective, which is much slower to converge.

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

## Citation

See [`../README.md`](../README.md).
