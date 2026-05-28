#!/bin/bash
# Launch TrajViT-v2 contrastive pretraining.
#
# Reuses the segmenter's training entrypoint (trajtok_segmenter.train.pretrain)
# with vit_type=simpletrajvitv2 → activates the SegmentCLIP head.
#
# Usage:
#   bash scripts/train.sh [--ngpus 8] [--nnodes 1] \
#                         [--seg_ckpt PATH]   # warm-start segmenter (recommended)
#                         [--train_corpus filteredmixdata_new] \
#                         [--exp_name myrun] [--epoch 20] [--lr 1e-4]
#
# Required env vars:
#   TRAJTOK_DATA_ROOT       — root of dataset annotations + tars (default ./data)
#   TRAJTOK_OUTPUT_DIR      — checkpoints + logs (default ./results)
#   TRAJTOK_DINOV3_ROOT     — cloned DINOv3 repo (default ./dinov3)

set -e

ngpus=8
nnodes=1
exp_name="default"
train_corpus="filteredmixdata_new"
batch_size_video=8
batch_size_image=64
epoch=20
lr=1e-4
seg_ckpt=None                            # released segmenter ckpt → warm-starts segmenter portion
ckpt=None                                # resume from prior trajvitv2 ckpt
resume=false
log_wandb=false
wandb_group="oss"

extra_overrides=()
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --ngpus) ngpus="$2"; shift 2 ;;
    --nnodes) nnodes="$2"; shift 2 ;;
    --exp_name) exp_name="$2"; shift 2 ;;
    --train_corpus) train_corpus="$2"; shift 2 ;;
    --batch_size_video) batch_size_video="$2"; shift 2 ;;
    --batch_size_image) batch_size_image="$2"; shift 2 ;;
    --epoch) epoch="$2"; shift 2 ;;
    --lr) lr="$2"; shift 2 ;;
    --seg_ckpt) seg_ckpt="$2"; shift 2 ;;
    --ckpt) ckpt="$2"; shift 2 ;;
    --resume) resume=true; shift ;;
    --log_wandb) log_wandb=true; shift ;;
    --wandb_group) wandb_group="$2"; shift 2 ;;
    *) extra_overrides+=("$1"); shift ;;
  esac
done

if [[ "$ckpt" != "None" ]]; then resume=true; fi

OUTPUT_ROOT="${TRAJTOK_OUTPUT_DIR:-./results}"
output_dir="${OUTPUT_ROOT}/trajvitv2_${train_corpus}_${exp_name}"
mkdir -p "$output_dir"
echo "output_dir = $output_dir"

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
SEGMENTER_PKG="${HERE}/../../segmenter"

common_args=(
  -m trajtok_segmenter.train.pretrain
  "${HERE}/../configs/pretrain.yaml"
  "output_dir=${output_dir}"
  "train_corpus=${train_corpus}"
  "vit_type=simpletrajvitv2"
  "image_res=224"
  "image_pretrained_path=${seg_ckpt}"
  "pretrained_path=${ckpt}"
  "resume=${resume}"
  "batch_size.video=${batch_size_video}"
  "batch_size.image=${batch_size_image}"
  "scheduler.epochs=${epoch}"
  "optimizer.lr=${lr}"
  "wandb.enable=${log_wandb}"
  "wandb.group=${wandb_group}"
)

PYTHONPATH="${SEGMENTER_PKG}:${HERE}/..:${PYTHONPATH}" \
torchrun --nnodes="${nnodes}" --nproc_per_node="${ngpus}" \
  "${common_args[@]}" "${extra_overrides[@]}"
