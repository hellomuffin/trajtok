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
# Multi-node: set MASTER_ADDR + MASTER_PORT + NODE_RANK manually and adjust
#             --nnodes accordingly (see torchrun docs).
#
# Required external dependencies (see README):
#   * molmo2 source (pip install -e <molmo2_clone>)
#   * SigLIP2-so400m-14-384.pt   (loaded via MOLMO_DATA_DIR/pretrained_image_encoders/)
#   * Qwen3-4B-Instruct          (loaded via MOLMO_DATA_DIR/pretrained_llms/)
#   * Released TrajTok segmenter (downloaded via ../segmenter/scripts/download_ckpt.py)
#
# Required env vars:
#   MOLMO_DATA_DIR             — where SigLIP2 + Qwen3 weights + PixMoCap data live
#   WANDB_PROJECT / WANDB_ENTITY — optional, for wandb logging

set -e

ngpus=8
nnodes=1
save_folder="./results/trajvlm_pretrain"
seg_ckpt="None"                                # set to released segmenter ckpt path
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

# Numerical / NCCL knobs (mirror Molmo2 defaults)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}

# Note: the trajvlm_pretrain.py launch script we vendor under
# trajtok_trajvlm/launch_scripts/ keeps its `from olmo.X` imports — molmo2
# must be importable. The simplest pattern is to keep this launcher running
# from the molmo2 repo root (so its olmo/ is on PYTHONPATH alongside ours).
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

torchrun --nnodes=${nnodes} --nproc-per-node=${ngpus} \
  -m trajtok_trajvlm.launch_scripts.trajvlm_pretrain \
  --save_folder=${save_folder} \
  --save_overwrite \
  --pretrained_segmenter_path=${seg_ckpt} \
  --global_batch_size ${global_batch_size} \
  --device_train_microbatch_size ${device_microbatch} \
  --no_compile \
  "${extra_args[@]}"
