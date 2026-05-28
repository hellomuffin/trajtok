"""TrajVLM pretrain — image captioning (PixMoCap) with our DINOv3-small + perceiver
trajectory connector + Qwen3-4B-Instruct LLM.

Mirrors `launch_scripts/pretrain.py` for Molmo2 but swaps:
  * `MolmoVisionBackboneConfig`  → `TrajVitVisionBackboneConfig`
  * `MultiCropConfig`            → `TrajVitPreprocessorConfig` for both image & video

Run (debug, single GPU):
    torchrun --nproc-per-node=1 launch_scripts/trajvlm_pretrain.py \
        --save_folder=/tmp/trajvlm_dbg --debug --save_overwrite

Run (production, 16 GPUs):
    torchrun --nproc-per-node=8 launch_scripts/trajvlm_pretrain.py \
        --save_folder=/path/to/save --global_batch_size=128 \
        --wandb.name=trajvlm_pretrain_v1
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import replace
from typing import cast

# ---- Re-type molmo2 dataclasses BEFORE they're touched by the trainer modules ----
# Same shim pattern as before; we re-bind the 'video_olmo' model name to our
# subclass so TrainConfig.load resolves resumed configs to TrajVlmConfig.
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


# Keep the same _model_name so checkpoints saved/loaded as 'video_olmo' resolve to us.
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

# Now the rest of the imports (which transitively construct structured schemas).
from omegaconf import OmegaConf, omegaconf  # noqa: E402

from olmo.data.data_loader import DataLoaderConfig, KwargsMixture, WeightedDataset  # noqa: E402
from olmo.data.dynamic_packer import PackingConfig  # noqa: E402
from olmo.eval.eval_utils import get_evaluation  # noqa: E402
from olmo.eval.loss_evaluator import LossDatasetEvaluatorConfig  # noqa: E402
from olmo.model_configs import LLMS  # noqa: E402
from olmo.models.model import FSDPWrapStrategy  # noqa: E402
from olmo.models.molmo2.molmo2 import Molmo2  # noqa: E402
from olmo.preprocessing.data_formatter import DataFormatter  # noqa: E402

# trainer_config also caches get_model_types lookups — re-bind there too.
from olmo.train import trainer_config as _trainer_config  # noqa: E402
_trainer_config.get_model_types = _patched_get_model_types

from olmo.train.optim import OptimizerConfig, OptimizerType, SchedulerConfig, SchedulerType
from olmo.train.run_trainer import run_trainer
from olmo.train.trainer_config import (
    BatchDivisor,
    CompilerConfig,
    FSDPConfig,
    FSDPPrecision,
    SpeedMonitorConfig,
    TrainConfig,
    WandbConfig,
)
from olmo.util import clean_opt, prepare_torchrun_environment

log = logging.getLogger("train")


# Default segmenter warm-start checkpoint. Override with --pretrained_segmenter_path
# (or set TRAJTOK_SEG_CKPT in your environment). "none" disables the warm-start.
# Download the released ckpt with `segmenter/scripts/download_ckpt.py`
# (it lands at ./checkpoints/segmenter_filteredmixdata_all.pth by default).
DEFAULT_SEG_CKPT = os.environ.get(
    "TRAJTOK_SEG_CKPT",
    "./checkpoints/segmenter_filteredmixdata_all.pth",
)


def _resolve_init_path(s):
    """LlmConfig.init_path is an OmegaConf interpolation like
    `${oc.env:MOLMO_DATA_DIR}/...`. We instantiate LlmConfig directly (not via
    OmegaConf), so the interpolation never fires. Substitute manually."""
    if "${oc.env:MOLMO_DATA_DIR}" in s:
        return s.replace("${oc.env:MOLMO_DATA_DIR}", os.environ["MOLMO_DATA_DIR"])
    return s


def build_model_cfg(
    llm_name: str,
    *,
    pretrained_segmenter_path: str | None,
    num_traj: int = 128,
    image_res: int = 378,
    debug: bool = False,
) -> Molmo2Config:
    llm = LLMS[llm_name]
    llm = replace(
        llm,
        init_path=_resolve_init_path(llm.init_path) if llm.init_path else None,
        residual_dropout=0.0,
        response_residual_dropout=0.1,
        additional_vocab_size=128,
    )
    # SigLIP2 vit config (Molmo2's pretrained siglip2-so400m-14-384)
    from olmo.model_configs import VISION_BACKBONES
    vit_cfg = VISION_BACKBONES["siglip2"]
    # Resolve env interpolation in init_path (we build outside OmegaConf).
    if vit_cfg.init_path and "${oc.env:MOLMO_DATA_DIR}" in vit_cfg.init_path:
        vit_cfg = replace(
            vit_cfg,
            init_path=vit_cfg.init_path.replace(
                "${oc.env:MOLMO_DATA_DIR}", os.environ["MOLMO_DATA_DIR"]
            ),
        )

    vis = SigLip2TrajGroupVisionBackboneConfig(
        vit=vit_cfg,
        pretrained_segmenter_path=pretrained_segmenter_path,
        num_traj=num_traj,
        frames_per_pool=8,
        connector_activation_checkpointing=not debug,
    )
    image_preproc = TrajVitImageConfig(
        image_res=image_res,
        num_traj=num_traj,
        normalize_on_gpu=False,
    )
    # For image-only pretrain (PixMoCap), keep the video preprocessor at a
    # trivial spec so the per-sample buffer isn't dominated by a hypothetical
    # 128-frame video; we'll fix this in SFT.
    video_preproc = TrajVitVideoConfig(
        image_res=image_res,
        num_traj=num_traj,
        num_frames=1,
        frames_per_clip=1,
        normalize_on_gpu=False,
    )
    return TrajVlmConfig(
        llm=llm,
        vision_backbone=vis,
        data_formatter=DataFormatter(
            system_prompt="style_and_length_v2",
            message_format="qwen3",
            pointing_format="html-v1",
            always_start_with_space=False,
        ),
        mm_preprocessor=TrajVlmPreprocessorConfig(
            image=image_preproc,
            video=video_preproc,
        ),
    )


def main():
    prepare_torchrun_environment()
    parser = argparse.ArgumentParser(prog="TrajVLM pretrain")
    parser.add_argument("--llm", default="qwen3_4b_instruct", choices=list(LLMS.keys()))
    parser.add_argument("--save_folder", required=True)
    parser.add_argument("--save_overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true",
                        help="Single GPU debug: tiny batch + short duration + no wandb")
    parser.add_argument("--mixed_debug", action="store_true",
                        help="Like --debug but USE the full caption/pointing/NLP mixture path."
                             " Catches batch-shape bugs that --debug (image-only) misses.")
    parser.add_argument("--global_batch_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=2536,
                        help="Match Molmo2 default (2536 — fits caption + pointing + NLP-mixed examples)")
    parser.add_argument("--n_eval_examples", type=int, default=2048)
    parser.add_argument("--device_eval_batch_size", type=int, default=4)
    parser.add_argument("--device_train_microbatch_size", type=int, default=4)
    parser.add_argument("--warmup_factor", type=int, default=1)
    parser.add_argument(
        "--pretrained_segmenter_path", default=DEFAULT_SEG_CKPT,
        help="Path to our TrajViT segmenter latest.pth (or 'none' to start from scratch).",
    )
    parser.add_argument("--num_traj", type=int, default=128)
    parser.add_argument("--image_res", type=int, default=378,
                        help="SigLIP2 native is 378×378.")
    parser.add_argument("--max_duration", type=int, default=None,
                        help="Override training steps (default: 4 epochs of PixMoCap captions, mirroring Molmo2)")
    # Data-mixture knobs (mirror Molmo2's pretrain.py defaults)
    parser.add_argument("--nlp", default=0.1, type=float,
                        help="Fraction of NLP data (tulu4) in the mixture. 0.0 to disable.")
    parser.add_argument("--pointing", default=0.3, type=float,
                        help="Fraction of pointing data (pixmo_points*, pixmo_count, cosyn_point). 0.0 to disable.")
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile. Use this if you hit triton cache errors.")
    args, other_args = parser.parse_known_args()

    if args.pretrained_segmenter_path.lower() == "none":
        seg_ckpt = None
    else:
        seg_ckpt = args.pretrained_segmenter_path

    # ---- model ----
    model_cfg = build_model_cfg(
        args.llm,
        pretrained_segmenter_path=seg_ckpt,
        num_traj=args.num_traj,
        image_res=args.image_res,
        debug=args.debug,
    )

    if args.debug:
        # Single-GPU debug: truncate the LLM + SigLIP2 layers to fit in 1 H100.
        # Mirrors what launch_scripts/sft.py does in --debug.
        model_cfg.llm.init_path = None
        model_cfg.llm.n_layers = 1
        vit = model_cfg.vision_backbone.vit
        vit.init_path = None
        vit.image_num_layers = 2
        # Skip the segmenter warm-start in debug; the loader needs the parent
        # backbone's state_dict to line up, and we test that on multi-GPU runs.
        # If a seg path was provided, keep it — but the smoke test runs cleaner
        # without it (random seg + hedge in _build_query_mask avoids NaN).

    # ---- duration: 4 epochs on PixMoCap captions, mirroring Molmo2 pretrain ----
    if args.max_duration is not None:
        duration = args.max_duration
    elif args.debug or args.mixed_debug:
        duration = 200
    else:
        from olmo.data.pixmo_datasets import PixMoCap
        n = len(PixMoCap("train", "captions"))
        duration = 4 * (n + args.global_batch_size - 1) // args.global_batch_size

    is_debug = args.debug or args.mixed_debug
    log_interval = 5 if is_debug else 20
    eval_interval = 50 if is_debug else 1000
    global_batch_size = 8 if is_debug else args.global_batch_size

    # ---- evaluation: mirror Molmo2 pretrain (caption_loss + caption_val) ----
    evaluator = LossDatasetEvaluatorConfig(
        label="caption_loss",
        data=DataLoaderConfig(
            dataset="pixmo_cap_with_transcripts",
            mixture=None,
            seed=95818,                # explicit; OmegaConf flags MISSING ('???') on resume
            shuffle=False,
            split="validation",
            drop_last=True,
            sequence_length=args.seq_len,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        ),
    )
    second_evaluator = LossDatasetEvaluatorConfig(
        label="caption_val",
        data=replace(evaluator.data, dataset="pixmo_cap"),
    )

    # ---- data: match Molmo2 pretrain mixture (caption + pointing + NLP) ----
    dataset = None
    mixture = None
    kwargs_mixture = None
    inf_evaluators = []
    if args.pointing and (not args.debug or args.mixed_debug):
        kwargs_mixture = [
            KwargsMixture(1.0 - args.pointing - args.nlp, [WeightedDataset("pixmo_cap_with_transcripts")]),
            KwargsMixture(args.pointing, [
                WeightedDataset("pixmo_points_train"),
                WeightedDataset("pixmo_count_train"),
                WeightedDataset("pixmo_points_high_freq_train"),
                WeightedDataset("cosyn_point"),
            ]),
        ]
        if args.nlp:
            # NLP examples can be longer than image-only ones — assert + extend duration a bit.
            assert args.seq_len > 2304, f"--seq_len must be > 2304 when --nlp > 0 (got {args.seq_len})"
            duration = int(duration * 1.0)   # NLP adds tokens, but we already budgeted 4 epochs
            kwargs_mixture.append(KwargsMixture(args.nlp, [WeightedDataset("tulu4_max_2304")]))
        # Optional inference-time evals from Molmo2 pretrain (pointing).
        try:
            for task in ["point_bench:test", "pixmo_count_counting:validation"]:
                ev = get_evaluation(
                    task, None,
                    device_batch_size=args.device_eval_batch_size,
                    max_examples=args.n_eval_examples,
                    num_workers=2,
                )
                ev.data.pad = None
                ev.data.max_text_seq_len = 196
                ev.data.persistent_workers = True
                ev.data.prefetch_factor = 4
                # Resolve "${console_log_interval}" string (we bypass OmegaConf).
                if isinstance(getattr(ev, "console_log_interval", None), str):
                    ev.console_log_interval = log_interval
                inf_evaluators.append(ev)
        except Exception as e:
            log.warning(f"could not build inf_evaluators (continuing): {e}")
    elif args.nlp and (not args.debug or args.mixed_debug):
        assert args.seq_len > 2304, f"--seq_len must be > 2304 when --nlp > 0 (got {args.seq_len})"
        mixture = {
            "pixmo_cap_with_transcripts": 1 - args.nlp,
            "tulu4_max_2304": args.nlp,
        }
    else:
        dataset = "pixmo_cap_with_transcripts"

    # ---- TrainConfig ----
    cfg = TrainConfig(
        save_folder=args.save_folder,
        seed=6198,
        dry_run=False,
        wandb=None if is_debug else WandbConfig(
            name="${run_name}",
            project=os.environ.get("WANDB_PROJECT", "trajvlm"),
            group=None,
            entity=os.environ.get("WANDB_ENTITY"),
            log_interval=log_interval,
        ),
        compile=None if (is_debug or args.no_compile) else CompilerConfig(mode="default", dynamic=False),
        fused_loss=False,
        compile_loss=not (is_debug or args.no_compile),
        model=model_cfg,
        data=DataLoaderConfig(
            dataset=dataset,
            mixture=mixture,
            kwargs_mixture=kwargs_mixture,
            shuffle=True,
            split="train",
            drop_last=True,
            sequence_length=args.seq_len,
            seed=95818,
            num_workers=2,
            pad="to_max",
            pin_memory=True,
            # Mirror Molmo2 pretrain: packing on when NLP mixed in.
            packing=PackingConfig(48, shortcut_max_len_images=False) if (args.nlp and not args.debug) else None,
        ),
        ft_connector=True,
        ft_llm=True,
        ft_vit=True,
        optimizer=OptimizerConfig(
            name=OptimizerType.adamw,
            # Mirror Molmo2-pretrain LRs (paper does not specify pretrain LRs explicitly).
            # SFT will follow paper's exact 1e-5 / 5e-6 / 5e-6.
            connector_learning_rate=2e-4,
            vit_learning_rate=6e-6,
            llm_learning_rate=2e-5,
            frame_selector_learning_rate=1e-4,
            metrics_log_interval=-1,
        ),
        scheduler=SchedulerConfig(
            name=SchedulerType.multimodal,
            connector_t_warmup=200 // args.warmup_factor,
            vit_t_warmup=2000 // args.warmup_factor,
            llm_t_warmup=2000 // args.warmup_factor,
            alpha_f=0.1,
            warmup_min_lr=0.0,
        ),
        fsdp=FSDPConfig(
            use_orig_params=True,
            wrapping_strategy=FSDPWrapStrategy.by_block_and_size,
            precision=FSDPPrecision.float,
        ),
        load_path=None,
        initial_model_checkpoint=None,
        save_overwrite=args.save_overwrite or is_debug,
        save_interval=4000 if not is_debug else 50,
        allow_resume=True,
        save_num_checkpoints_to_keep=1,
        save_final_unsharded_checkpoint=False,
        global_train_batch_size=global_batch_size,
        device_train_microbatch_size=args.device_train_microbatch_size,
        time_limit=None,
        max_duration=duration,
        stop_at=duration,                    # we don't go through the OmegaConf round-trip
        # so the "${max_duration}" interpolation wouldn't resolve.
        max_grad_norm=1,
        batch_divisor=BatchDivisor.global_batch,
        precision="amp_bf16",
        console_log_interval=log_interval,
        speed_monitor=SpeedMonitorConfig(window_size=20),
        softmax_auxiliary_loss=True,
        softmax_auxiliary_loss_scale=1e-4,
        eval_interval=eval_interval,
        inf_eval_interval=2000,
        response_logits_only=True,
        inf_evaluators=inf_evaluators,
        evaluators=[evaluator, second_evaluator],
    )

    # Allow CLI dotlist overrides via omegaconf, but be tolerant of unknown keys.
    if other_args:
        conf = OmegaConf.structured(cfg)
        conf.merge_with_dotlist([clean_opt(arg) for arg in other_args])
        cfg = cast(TrainConfig, OmegaConf.to_object(conf))

    run_trainer(cfg)


if __name__ == "__main__":
    main()
