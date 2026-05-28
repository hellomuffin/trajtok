#!/bin/bash
# TrajVLM SFT launcher (single-node example).
#
# Fine-tunes a trajvlm_pretrain ckpt on the Molmo2 default SFT mixture
# (broad image+video coverage, paper LRs: connector 5e-6 / vit 5e-6 / llm 1e-5).
#
# Usage (1 node × 8 GPUs):
#   bash scripts/sft.sh \\
#       --pretrain_ckpt /path/to/trajvlm_pretrain/stepNNNNN \\
#       --save_folder /path/to/sft_save
#
# See ./pretrain.sh for required env vars + external dependencies.

set -e

ngpus=8
nnodes=1
save_folder="./results/trajvlm_sft"
pretrain_ckpt=""
mixture="molmo2"                           # see trajtok_trajvlm/launch_scripts/sft.get_training_mixture
device_batch_size=2

extra_args=()
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --ngpus) ngpus="$2"; shift 2 ;;
    --nnodes) nnodes="$2"; shift 2 ;;
    --save_folder) save_folder="$2"; shift 2 ;;
    --pretrain_ckpt) pretrain_ckpt="$2"; shift 2 ;;
    --mixture) mixture="$2"; shift 2 ;;
    --device_batch_size) device_batch_size="$2"; shift 2 ;;
    *) extra_args+=("$1"); shift ;;
  esac
done

if [[ -z "${pretrain_ckpt}" ]]; then
  echo "Error: --pretrain_ckpt is required (path to a trajvlm_pretrain step* checkpoint)"
  exit 1
fi
mkdir -p "${save_folder}"

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}

torchrun --nnodes=${nnodes} --nproc-per-node=${ngpus} \
  -m trajtok_trajvlm.launch_scripts.trajvlm_sft \
  ${pretrain_ckpt} \
  ${mixture} \
  --save_folder=${save_folder} \
  --device_batch_size ${device_batch_size} \
  --no_compile \
  "${extra_args[@]}"
