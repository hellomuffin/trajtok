#!/bin/bash
# Launch trajectory-segmenter pretraining.
#
# Usage:
#   bash scripts/train.sh [--ngpus 8] [--nnodes 1] [--exp_name myrun] \
#                         [--train_corpus filteredmixdata_all] \
#                         [--epoch 20] [--lr 1e-4]
#
# Required env vars (or pass via CLI overrides):
#   TRAJTOK_DATA_ROOT       — root of dataset annotations + tars (default ./data)
#   TRAJTOK_OUTPUT_DIR      — checkpoints + logs (default ./results)
#   TRAJTOK_DINOV3_ROOT     — cloned dinov3 repo (default ./dinov3)
#   TRAJTOK_DINOV3_WEIGHTS_DIR — DINOv3 weights dir (default = TRAJTOK_DINOV3_ROOT)
#
# OmegaConf dotlist overrides can be appended verbatim after the named args, e.g.
#   bash scripts/train.sh --ngpus 8 batch_size.image=128 optimizer.lr=5e-5

set -e

# ---- defaults ----
ngpus=8
nnodes=1
exp_name="default"
train_corpus="filteredmixdata_all"
test_corpus="all"
model="simplesegmenter"               # 'simplesegmenter' (this package) or 'simpletrajvitv2' (with CLIP head; see ../trajvitv2)
batch_size_video=8
batch_size_image=64
epoch=20
lr=1e-4
num_traj=128
mask_down_factor=1
vit_name="vit-large"
embed_dim=512
backbone_model="dinov3_small"
ckpt=None
resume=false
log_wandb=false
wandb_group="oss"

# ---- parse args ----
extra_overrides=()
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --ngpus) ngpus="$2"; shift 2 ;;
    --nnodes) nnodes="$2"; shift 2 ;;
    --exp_name) exp_name="$2"; shift 2 ;;
    --train_corpus) train_corpus="$2"; shift 2 ;;
    --test_corpus) test_corpus="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --batch_size_video) batch_size_video="$2"; shift 2 ;;
    --batch_size_image) batch_size_image="$2"; shift 2 ;;
    --epoch) epoch="$2"; shift 2 ;;
    --lr) lr="$2"; shift 2 ;;
    --num_traj) num_traj="$2"; shift 2 ;;
    --vit_name) vit_name="$2"; shift 2 ;;
    --embed_dim) embed_dim="$2"; shift 2 ;;
    --backbone_model) backbone_model="$2"; shift 2 ;;
    --ckpt) ckpt="$2"; shift 2 ;;
    --resume) resume=true; shift ;;
    --log_wandb) log_wandb=true; shift ;;
    --wandb_group) wandb_group="$2"; shift 2 ;;
    *) extra_overrides+=("$1"); shift ;;
  esac
done

if [[ "$ckpt" != "None" ]]; then resume=true; fi

OUTPUT_ROOT="${TRAJTOK_OUTPUT_DIR:-./results}"
output_dir="${OUTPUT_ROOT}/${train_corpus}_${exp_name}_${model}"
mkdir -p "$output_dir"
echo "output_dir = $output_dir"

# ---- launch ----
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
common_args=(
  -m trajtok_segmenter.train.pretrain
  "${HERE}/../configs/pretrain.yaml"
  "output_dir=${output_dir}"
  "train_corpus=${train_corpus}"
  "test_corpus=${test_corpus}"
  "vit_type=${model}"
  "image_res=224"
  "mask_down_factor=${mask_down_factor}"
  "pretrained_path=${ckpt}"
  "resume=${resume}"
  "batch_size.video=${batch_size_video}"
  "batch_size.image=${batch_size_image}"
  "scheduler.epochs=${epoch}"
  "optimizer.lr=${lr}"
  "traj_model.model_name=${vit_name}"
  "traj_model.embed_dim=${embed_dim}"
  "traj_model.num_traj=${num_traj}"
  "backbone.backbone_model=${backbone_model}"
  "wandb.enable=${log_wandb}"
  "wandb.group=${wandb_group}"
)

PYTHONPATH="${HERE}/..:${PYTHONPATH}" \
torchrun --nnodes="${nnodes}" --nproc_per_node="${ngpus}" \
  "${common_args[@]}" "${extra_overrides[@]}"
