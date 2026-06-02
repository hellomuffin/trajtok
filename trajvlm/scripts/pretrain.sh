#!/bin/bash
# TrajVLM pretraining launcher (single-node example).
#
# Architecture: SigLIP2 ViT (Molmo2 stock) + SimpleSegmenter (released ckpt) +
#               cross-attn TrajPerceiver @ 1152 dim + projector MLP →
#               Qwen3-4B-Instruct LLM.
# Data: image-only PixMoCap captioning + pointing + tulu4 NLP (Molmo2 pretrain mixture).
# Duration: ~4 epochs of PixMoCap (~22k steps at gbs=128).
#
# Usage (1 node × 8 GPUs):
#   bash scripts/pretrain.sh --save_folder /path/to/save --seg_ckpt /path/to/segmenter.pth
#
# Required env vars (also documented in the README):
#   MOLMO_DATA_DIR             — where SigLIP2 + Qwen3 weights + PixMoCap data live
#   WANDB_PROJECT / WANDB_ENTITY — optional, for wandb logging

set -e

ngpus=8
nnodes=1
save_folder="./results/trajvlm_pretrain"
seg_ckpt="None"
global_batch_size=128
device_microbatch=4

extra_args=()
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --ngpus) ngpus="$2"; shift 2 ;;
    --nnodes) nnodes="$2"; shift 2 ;;
    --save_folder) save_folder="$2"; shift 2 ;;
    --seg_ckpt) seg_ckpt="$2"; shift 2 ;;
    --global_batch_size) global_batch_size="$2"; shift 2 ;;
    --device_microbatch) device_microbatch="$2"; shift 2 ;;
    *) extra_args+=("$1"); shift ;;
  esac
done

mkdir -p "${save_folder}"

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}

# Resolve repo root (this script's parent's parent) so PYTHONPATH lets
# `from olmo... import ...` work whether you run from anywhere.
HERE="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
export PYTHONPATH="${HERE}:${PYTHONPATH}"

torchrun --nnodes=${nnodes} --nproc-per-node=${ngpus} \
  ${HERE}/launch_scripts/trajvlm_pretrain.py \
  --save_folder=${save_folder} \
  --save_overwrite \
  --pretrained_segmenter_path=${seg_ckpt} \
  --global_batch_size ${global_batch_size} \
  --device_train_microbatch_size ${device_microbatch} \
  --no_compile \
  "${extra_args[@]}"
