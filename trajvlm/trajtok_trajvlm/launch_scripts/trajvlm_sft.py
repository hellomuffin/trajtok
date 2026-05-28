"""TrajVLM SFT — fine-tune trajvlm_pretrain_v1 checkpoint on the Molmo2 SFT
mixture (broad image+video coverage) with paper-aligned LRs.

Mirrors ``launch_scripts/sft.py`` but:
  * Patches ``get_model_types`` so the pretrain config (saved under
    ``_model_name='video_olmo'``) is loaded as :class:`TrajVlmConfig`, not as
    the stock Molmo2Config — this preserves the trajvit-specific
    vision_backbone fields and ``pretrained_segmenter_path``.
  * Replaces the stock ``get_model`` so the video preprocessor uses
    :class:`TrajVitVideoConfig` at 128 frames × 16-frame clips (the spec the
    paper's SFT trains on; pretrain ran image-only with num_frames=1).
  * Reuses ``get_training_mixture`` from ``sft.py`` directly.

Run (production, 2 nodes × 8 GPUs):
    torchrun --nproc-per-node=8 launch_scripts/trajvlm_sft.py \\
        <pretrain_ckpt_dir> molmo2 \\
        --save_folder=/path/to/sft_save \\
        --global_batch_size=128 --device_train_microbatch_size=2 --no_compile
"""
from __future__ import annotations

# ---- Re-type molmo2 dataclasses BEFORE they're touched by the trainer modules ----
# Identical shim to trajvlm_pretrain.py — subclasses Molmo2Config and
# Molmo2PreprocessorConfig with TrajVit-typed fields. We then point
# get_model_types('video_olmo') at our subclass so TrainConfig.load resolves
# the pretrain ckpt's config.yaml to TrajVlmConfig.
import dataclasses as _dc
from olmo.models.molmo2.molmo2 import Molmo2Config
from olmo.models.molmo2.molmo2_preprocessor import Molmo2PreprocessorConfig
from trajtok_trajvlm.nn.siglip2_trajgroup_vision_backbone import (
    SigLip2TrajGroupVisionBackboneConfig,
)
from trajtok_trajvlm.preprocessing.trajvit_preprocessor import TrajVitImageConfig, TrajVitVideoConfig
from olmo.models import model_config as _model_config


@_dc.dataclass
class TrajVlmPreprocessorConfig(Molmo2PreprocessorConfig):
    image: TrajVitImageConfig = _dc.field(default_factory=TrajVitImageConfig)
    video: TrajVitVideoConfig = _dc.field(default_factory=TrajVitVideoConfig)


@_dc.dataclass
class TrajVlmConfig(Molmo2Config):
    vision_backbone: SigLip2TrajGroupVisionBackboneConfig = _dc.field(
        default_factory=SigLip2TrajGroupVisionBackboneConfig,
    )
    mm_preprocessor: TrajVlmPreprocessorConfig = _dc.field(default_factory=TrajVlmPreprocessorConfig)


TrajVlmConfig._model_name = Molmo2Config._model_name  # "video_olmo"


def _patched_get_model_types():
    from olmo.models.molmo.molmo import MolmoConfig
    from olmo.models.molmo_point.molmo_point import MolmoPointConfig
    return {
        MolmoConfig._model_name: MolmoConfig,
        TrajVlmConfig._model_name: TrajVlmConfig,       # overrides Molmo2Config
        MolmoPointConfig._model_name: MolmoPointConfig,
        "token_indexing_video_molmo2": MolmoPointConfig,
    }


_model_config.get_model_types = _patched_get_model_types


# Now the rest of the imports.
import argparse  # noqa: E402
import dataclasses  # noqa: E402
from os.path import join  # noqa: E402
from typing import cast  # noqa: E402

import numpy as np  # noqa: E402
from omegaconf import OmegaConf, omegaconf  # noqa: E402

from olmo.data.data_loader import DataLoaderConfig  # noqa: E402
from olmo.data.dynamic_packer import PackingConfig  # noqa: E402
from olmo.eval.eval_utils import get_evaluation  # noqa: E402
from olmo.models.molmo.molmo import MolmoConfig  # noqa: E402
from olmo.preprocessing.multicrop_preprocessor import MultiCropConfig  # noqa: E402
from olmo.preprocessing.video_preprocessor import VideoPreprocessorConfig  # noqa: E402
from olmo.torch_util import get_world_size  # noqa: E402
from olmo.train.optim import OptimizerConfig, OptimizerType, SchedulerConfig, SchedulerType  # noqa: E402
from olmo.train.run_trainer import run_trainer  # noqa: E402
from olmo.train.trainer_config import (  # noqa: E402
    BatchDivisor,
    CompilerConfig,
    FSDPConfig,
    SpeedMonitorConfig,
    TrainConfig,
    WandbConfig,
)
from olmo.util import clean_opt, prepare_torchrun_environment, select_checkpoint  # noqa: E402

# trainer_config also caches get_model_types lookups — re-bind there too.
from olmo.train import trainer_config as _trainer_config  # noqa: E402
_trainer_config.get_model_types = _patched_get_model_types

# Reuse Molmo2 SFT mixtures verbatim — user explicitly chose "Molmo2 default SFT mix".
from launch_scripts.sft import get_training_mixture  # noqa: E402


def get_trajvlm_model(checkpoint, *, num_frames: int, frames_per_clip: int):
    """Load the pretrain ckpt's config (TrajVlmConfig after our shim), then
    apply SFT-style fine-tuning settings (mirrors molmo2/sft.py's get_model)
    while swapping the image-only video preprocessor for the real
    TrajVitVideoConfig at 128 frames × 16-frame clips.
    """
    # `MolmoConfig.load` dispatches via `_model_name` -> get_model_types(),
    # which (per our shim above) resolves "video_olmo" to TrajVlmConfig.
    model_cfg = MolmoConfig.load(join(checkpoint, "config.yaml"), key="model")
    assert isinstance(model_cfg, TrajVlmConfig), (
        f"expected TrajVlmConfig, got {type(model_cfg).__name__}. "
        f"Did the get_model_types shim register correctly?"
    )

    # Image preprocessor: keep the trajvit-specific one from pretrain.
    if isinstance(model_cfg.mm_preprocessor.image, TrajVitImageConfig):
        image_preproc = model_cfg.mm_preprocessor.image
    else:
        # Fallback: reconstruct from base fields. Should not happen for our ckpt.
        image_preproc = TrajVitImageConfig(
            image_res=model_cfg.vision_backbone.image_res,
            num_traj=model_cfg.vision_backbone.num_traj,
            normalize_on_gpu=model_cfg.vision_backbone.normalize_on_gpu,
        )

    # Video preprocessor: pretrain used (num_frames=1, frames_per_clip=1) since
    # PixMoCap is image-only; SFT needs the full video spec.
    video_preproc = TrajVitVideoConfig(
        image_res=model_cfg.vision_backbone.image_res,
        num_traj=model_cfg.vision_backbone.num_traj,
        num_frames=num_frames,
        frames_per_clip=frames_per_clip,
        normalize_on_gpu=model_cfg.vision_backbone.normalize_on_gpu,
    )

    model_cfg = TrajVlmConfig(
        llm=model_cfg.llm,
        vision_backbone=model_cfg.vision_backbone,
        data_formatter=model_cfg.data_formatter,
        mm_preprocessor=TrajVlmPreprocessorConfig(
            video=video_preproc,
            image=image_preproc,
        ),
        bi_directional_attn=model_cfg.bi_directional_attn,
    )

    # ---- SFT-time fine-tuning settings (mirror molmo2/sft.py get_model) ----
    model_cfg.vision_backbone.pooling_attention_mask = True
    model_cfg.data_formatter.pointing_format = "html-v2"
    model_cfg.mm_preprocessor.video.max_subtitle_tokens = None
    model_cfg.data_formatter.p_multi_point_all_image = 0.5
    model_cfg.data_formatter.p_choice_content_in_mc = 1.0

    model_cfg.llm.residual_dropout = 0.1
    model_cfg.llm.response_residual_dropout = 0.0
    model_cfg.data_formatter.prompt_templates = "uber_model_v2"
    model_cfg.data_formatter.message_format = "qwen3"
    model_cfg.data_formatter.system_prompt = "demo_or_style_v2"
    model_cfg.mm_preprocessor.loss_token_weighting = "root_subsegments_root_tokens"

    # Multi-image settings — the trajvit image preprocessor doesn't support
    # multi-image yet, so these are inherited from the base preprocessor but
    # safe to leave as defaults. (TrajVitImageConfig.build_image_preprocessor
    # returns multi_image_preprocessor=None.)

    # Good enough for 128 frames at ~28 tokens/frame.
    model_cfg.llm.max_sequence_length = 4096 * 4

    return model_cfg


def main():
    prepare_torchrun_environment()

    parser = argparse.ArgumentParser(prog="Train trajvlm with multitask SFT")
    parser.add_argument("checkpoint", help="Path to trajvlm pretrain checkpoint to start from")
    parser.add_argument("mixture", default="molmo2")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seq_len", type=int, default=16384)
    parser.add_argument("--device_batch_size", default=2, type=int)
    parser.add_argument("--max_loss_examples", default=2048, type=int)
    parser.add_argument("--max_inf_eval_examples", default=1280, type=int)
    parser.add_argument("--prefetch_factor", default=4, type=int)
    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument("--cp_degree", default=1, type=int)
    parser.add_argument("--num_frames", default=128, type=int,
                        help="Total frames per video sample; split into ceil(num_frames/frames_per_clip) clips.")
    parser.add_argument("--frames_per_clip", default=16, type=int)
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile. Trajvlm pretrain ran with --no_compile due to "
                             "the perceiver+DTensor interaction; mirror that for SFT until verified.")
    args, other_args = parser.parse_known_args()

    if args.mixture == "debug":
        loss_eval_tasks = ["llava_video_oe_academic", "pixmo_ask_model_anything"]
        eval_tasks = ["chart_qa", "mvbench", "mevis_track_eval_1fps:test"]
    elif args.mixture.startswith("pointing"):
        loss_eval_tasks = []
        eval_tasks = ["pointing_eval_v2:test", "vixmo_points_count:val", "pixmo_count_counting:validation"]
    elif args.mixture.startswith("vpointing"):
        loss_eval_tasks = []
        eval_tasks = ["vixmo_points_count:val"]
    elif args.mixture.startswith("image-only-v5"):
        loss_eval_tasks = []
        eval_tasks = ["chart_qa", "info_qa", "coco_2014_vqa_multi", "pointing_eval_v2:test"]
    else:
        loss_eval_tasks = ["llava_video_oe_academic", "pixmo_ask_model_anything", "pixmo_cap"]
        eval_tasks = [
            "chart_qa", "info_qa", "coco_2014_vqa_multi",
            "pixmo_clocks",
            "pointing_eval_v2:test",
            "muir_bench:test",
            "mvbench",
            "vixmo_points_count:val",
        ]

    training_mixture = get_training_mixture(args.mixture)
    seq_len = args.seq_len

    checkpoint = select_checkpoint(args.checkpoint)
    model_cfg = get_trajvlm_model(
        checkpoint,
        num_frames=args.num_frames,
        frames_per_clip=args.frames_per_clip,
    )

    if args.debug:
        checkpoint = None
        # Dummy model: tiny LLM + tiny ViT init for fast smoke test.
        model_cfg.llm.init_path = None
        model_cfg.llm.n_layers = 1
        vit = model_cfg.vision_backbone.vit
        vit.init_path = None
        vit.image_num_layers = 2
        # Don't try to warm-start the segmenter in debug — checkpoint may not exist.
        model_cfg.vision_backbone.pretrained_segmenter_path = None
        args.num_workers = 2
        args.prefetch_factor = 2

    num_workers = args.num_workers
    evaluations = []
    for task in eval_tasks:
        evaluation = get_evaluation(
            task,
            None,
            device_batch_size=args.device_batch_size * 2,
            max_examples=args.max_inf_eval_examples,
            num_workers=num_workers,
        )
        evaluation.data.pad = None
        evaluation.data.max_text_seq_len = 128
        evaluation.data.persistent_workers = True
        evaluation.data.prefetch_factor = args.prefetch_factor
        evaluations.append(evaluation)

    loss_evaluations = []
    for task in loss_eval_tasks:
        evaluation = get_evaluation(
            task,
            seq_len=seq_len,
            for_inference=False,
            device_batch_size=args.device_batch_size * 2,
            max_examples=args.max_loss_examples,
            num_workers=num_workers,
        )
        evaluation.data.max_text_seq_len = None
        evaluation.data.pad = "to_max"
        evaluation.data.persistent_workers = True
        evaluation.data.prefetch_factor = args.prefetch_factor
        loss_evaluations.append(evaluation)

    log_interval = 1 if args.debug else 20
    cfg = TrainConfig(
        run_name="trajvlm_sft",
        save_folder=omegaconf.MISSING,
        seed=6198,
        dry_run=False,

        wandb=None if args.debug else WandbConfig(
            name="${run_name}",
            project="${oc.env:WANDB_PROJECT}",
            group=None,
            entity="${oc.env:WANDB_ENTITY}",
            log_interval=log_interval,
            allow_resume=False,
            finish_on_sigterm=True,
        ),
        compile=None if (args.debug or args.no_compile) else CompilerConfig(mode="default", dynamic=False),
        fused_loss=False,
        allow_resume=True,
        model=model_cfg,
        save_overwrite=True,
        data=DataLoaderConfig(
            kwargs_mixture=training_mixture,
            shuffle=True,
            split="train",
            drop_last=True,
            sequence_length=seq_len,
            max_text_seq_len=None,
            num_workers=num_workers,
            pad="to_max",
            pin_memory=True,
            prefetch_factor=args.prefetch_factor,
            seed=50189,
            packing=PackingConfig(
                buffer_size=48, image_weight=30, shortcut_max_len_images=False,
                cp_world_size=args.cp_degree,
            ),
        ),
        ft_connector=True,
        ft_llm=not args.debug,
        ft_vit=not args.debug,
        optimizer=OptimizerConfig(
            name=OptimizerType.adamw,
            # Paper §4 SFT LRs: connector 5e-6, vit 5e-6, llm 1e-5.
            connector_learning_rate=5e-6,
            vit_learning_rate=5e-6,
            llm_learning_rate=1e-5,
            frame_selector_learning_rate=1e-4,
        ),
        scheduler=SchedulerConfig(
            name=SchedulerType.multimodal,
            connector_t_warmup=200,
            vit_t_warmup=200,
            llm_t_warmup=200,
            frame_selector_t_warmup=200,
            alpha_f=0.1,
            warmup_min_lr=0.0,
        ),
        fsdp=FSDPConfig(fsdp2=True),
        load_path=None,
        initial_model_checkpoint=checkpoint,
        save_interval=2000,
        save_num_checkpoints_to_keep=1,
        global_train_batch_size=get_world_size() if args.debug else 128,
        device_train_microbatch_size=args.device_batch_size,
        time_limit=None,
        max_duration=300000,
        stop_at=300000,                       # bypass OmegaConf interpolation
        max_grad_norm=1,
        batch_divisor=BatchDivisor.global_batch,
        precision="amp_bf16",
        console_log_interval=log_interval,
        compile_loss=not (args.debug or args.no_compile),
        speed_monitor=SpeedMonitorConfig(window_size=20),
        softmax_auxiliary_loss=True,
        softmax_auxiliary_loss_scale=1e-4,
        inf_evaluators=evaluations,
        evaluators=loss_evaluations,
        inf_eval_interval=-1,
        eval_interval=-1,
        save_final_unsharded_checkpoint=False,
        save_final_optim=False,
        response_logits_only=True,
    )

    cfg.parallelism.context_parallel_config.degree = args.cp_degree

    conf = OmegaConf.create(cfg)
    if other_args:
        conf.merge_with_dotlist([clean_opt(arg) for arg in other_args])
    conf = OmegaConf.to_object(conf)

    if conf.parallelism.context_parallel_config.degree > 1:
        conf.model.cp_enabled = True
        conf.compile = None

    run_trainer(conf)


if __name__ == "__main__":
    main()
