from __future__ import annotations

import cProfile
import dataclasses
import gc
import logging
import math
import os
import random
import re
import signal
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field, replace
from datetime import timedelta
from os.path import join
from pathlib import Path
from pstats import SortKey
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from beaker import Beaker
from beaker.exceptions import BeakerError
from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions, \
    set_model_state_dict

from ..eval.evaluators import SavePredictions

try:
    from beaker.client import ExperimentClient
except ImportError:
    # for older versions of beaker py
    from beaker import Experiment as ExperimentClient

from packaging import version
from requests import RequestException
from torch import nn
from torch.nn.functional import l1_loss, mse_loss
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, \
    FullStateDictConfig
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from wandb.sdk.data_types.base_types.wb_value import WBValue

from .trainer_config import (
    SpeedMonitorConfig, CheckpointType,
    TrainConfig, BatchDivisor,
)
from olmo.data.iterable_dataset_mixture import IterableDatasetMixture, WorkerState, \
    IterableDataMixtureCheckpoint
from olmo.eval.inf_evaluator import InfDatasetEvaluator
from olmo.eval.loss_evaluator import LossMetrics, LossDatasetEvaluator
from olmo.exceptions import OLMoConfigurationError
from olmo.models.molmo.molmo import Molmo
from olmo.train.optim import Optimizer, Scheduler, SchedulerUnits
from olmo.torch_util import (
    barrier,
    gc_cuda,
    get_fs_local_rank,
    get_global_rank,
    get_world_size,
    move_to_device,
    peak_gpu_memory,
    synchronize_flag,
    synchronize_value, get_local_world_size, clip_grad_norm, save_debug_batch, )
from olmo.dist_util import get_dp_process_group
from olmo.io import PathOrStr, clear_directory, is_url, normalize_path
from olmo.train.checkpointer import Checkpointer, save_unsharded
from ..data.dynamic_packer import EXAMPLE_SUBSEGMENT_INCREMENT
from ..util import flatten_lists, format_timedelta

try:
    from megablocks.layers.moe import (
        batched_load_balancing_loss,
        clear_load_balancing_loss,
        get_load_balancing_loss,
    )
except ImportError:
    pass


log = logging.getLogger(__name__)


@dataclass
class BeakerLogger:
    """Logs progress to Ai2 beaker's server"""

    WANDB_REGEX = ".*( \(https://wandb.ai/.*\))$"
    beaker: Beaker
    experiment_id: str
    log_interval: int
    _workload: Any = None
    _original_description: str = None
    _is_v1 = None

    def __post_init__(self):
        self._is_v1 = hasattr(self.beaker.experiment, "get")
        if self._is_v1:
            self._workload = self.beaker.experiment.get(self.experiment_id)
            self._original_description = self._workload.description
        else:
            self._workload = self.beaker.workload.get(self.experiment_id)
            self._original_description = self._workload.experiment.description

    def get_beaker_url(self):
        if self._is_v1:
            return self.beaker.experiment.url(self._workload)
        else:
            return self.beaker.workload.url(self._workload)

    def _set_description(self, description):
        try:
            if self._is_v1:
                self.beaker.experiment.set_description(self._workload, description)
            else:
                self.beaker.workload.update(self._workload, description=description)
        except (RequestException, BeakerError) as e:
            log.warning(f"Failed to update Beaker experiment description: {e}")

    def log_init(self):
        self._set_description(f"[Init] " + self._original_description)

    def add_wandb(self, wandb_url):
        # If there is an old wandb url (such as if the run was preempted), remove it
        match = re.match(self.WANDB_REGEX, self._original_description)
        if match:
            log.info(f"Removing old wandb url {wandb_url}")
            self._original_description = self._original_description[:match.start(1)]

        self._original_description = self._original_description + " (" + wandb_url + ")"
        self._set_description(f"[Init] " + self._original_description)

    def log_progress(self, on_step, target_step, eta=None):
        if eta:
            self._set_description(f"[{100*on_step/target_step:04.1f}%; eta={eta}] " + self._original_description)
        else:
            self._set_description(f"[{100*on_step/target_step:04.1f}%] " + self._original_description)

    def log_evaluation(self, eval_name, on_step, target_step):
        self._set_description(f"[{100*on_step/target_step:04.1f}%, {eval_name}] " + self._original_description)

    def finish(self):
        self._set_description(f"[Done] " + self._original_description)


@dataclass
class BatchStatsMonitor:
    """Tracks batch-statics to log to wandb"""

    max_window_size: int = 20
    sync_nodes: bool = True
    _batch_stats: Deque[Dict[str, float]] = field(default_factory=lambda: deque([]))

    def log_batch(self, batch):
        input_ids = batch["input_ids"]
        non_masked = (input_ids >= 0).to(dtype=torch.float32)
        stats = {
            "batch/non_masked_tokens": non_masked.sum(-1).mean(),
            "batch/per_non_masked_tokens": non_masked.mean(),
            "batch/examples_truncated": non_masked[:, -1].mean(),
            "batch/per_non_masked_images": 1.0 - torch.all(batch["images"] == -1, -1).float().mean()
        }
        if "loss_masks" in batch:
            mask = (batch["loss_masks"] > 0).to(dtype=torch.float32)
            stats["batch/loss_tokens"] = mask.sum(-1).mean()
            stats["batch/per_loss_tokens"] = mask.mean()
        if "subsegment_ids" in batch:
            subsegment_ids = batch["subsegment_ids"]
            n_packed = (subsegment_ids.max(-1).values // EXAMPLE_SUBSEGMENT_INCREMENT).float().mean() + 1
            stats["batch/n_packed"] = n_packed
            n_segments = 0
            for ex_subsegment_ids in subsegment_ids:
                values = torch.unique(ex_subsegment_ids)
                # Count unique non-padding and non-image subsegments
                n_segments += ((values != -1) & (values % EXAMPLE_SUBSEGMENT_INCREMENT != 10000)).sum()
            stats["batch/n_segments"] = n_segments / len(subsegment_ids)
        else:
            stats["batch/n_packed"] = torch.ones((), device=input_ids.device)
            stats["batch/n_segments"] = torch.ones((), device=input_ids.device)

        self._batch_stats.append(stats)
        if len(self._batch_stats) > self.max_window_size:
            self._batch_stats.popleft()

    def reset(self) -> None:
        self._batch_stats.clear()

    def check(self, device):
        stats = defaultdict(list)
        for batch in self._batch_stats:
            for k, v in batch.items():
                stats[k].append(v)

        out = {}
        for k, v in stats.items():
            v = torch.stack(v).mean()
            if self.sync_nodes:
                v = v.to(device)
                dist.all_reduce(v)
                v.div_(get_world_size())
            out[k] = v.item()
        return out


@dataclass
class SpeedMonitor:
    """Logs speen information for wandb"""

    cfg: SpeedMonitorConfig
    global_total_tokens: int = 0
    stats: Deque[Tuple[float, int, int]] = field(default_factory=lambda: deque([]))

    def batch_start(self, global_total_tokens: int, device_batch_num_tokens: int, device_batch_num_loss_tokens: int, record: bool = True) -> None:
        self.global_total_tokens = global_total_tokens
        if record:
            if len(self.stats) >= self.cfg.window_size:
                self.stats.popleft()
            self.stats.append((
                time.monotonic(),
                device_batch_num_tokens,
                device_batch_num_loss_tokens
            ))

    def reset(self) -> None:
        self.stats.clear()

    def check(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {"throughput/total_tokens": self.global_total_tokens}
        if self.stats:
            interval_seconds = time.monotonic() - self.stats[0][0]
            interval_batches = len(self.stats)
            interval_tokens = sum(x[1] for x in self.stats)
            interval_loss_tokens = sum(x[2] for x in self.stats)
            metrics["throughput/device/loss_tokens_per_second"] = interval_loss_tokens / interval_seconds
            metrics["throughput/device/tokens_per_second"] = interval_tokens / interval_seconds
            metrics["throughput/device/batches_per_second"] = interval_batches / interval_seconds
        return metrics


@dataclass
class LRMonitor:
    """Logs learning rate information for wandb"""
    optim: torch.optim.Optimizer

    def check(self) -> Dict[str, float]:
        group_lrs = {}
        for group in self.optim.param_groups:
            if group['group_name'] in group_lrs:
                assert group_lrs[group['group_name']] == group['lr']
            else:
                group_lrs[group['group_name']] = group['lr']
        return {f"optim/{name}_lr": lr for name, lr in group_lrs.items()}


def cross_entropy_loss(
    logits, labels, ignore_index: int = -100, reduction: str = "mean", compute_z_loss: bool = False, z_loss_scale: float = 1e-4,
):
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)

    if not compute_z_loss:
        return loss, None

    z_squared = logits.logsumexp(-1).pow(2)
    if reduction == "mean":
        z_squared = (z_squared * (labels != ignore_index)).mean()
    elif reduction == "sum":
        z_squared = (z_squared * (labels != ignore_index)).sum()

    z_loss = z_loss_scale * z_squared

    return loss, z_loss


@dataclass
class Trainer:
    cfg: TrainConfig
    model: Molmo
    mesh: DeviceMesh
    fsdp_model: FSDP
    optim: Optimizer
    scheduler: Scheduler
    train_loader: DataLoader
    device: torch.device
    evaluators: List[LossDatasetEvaluator]
    inference_evaluators: List[InfDatasetEvaluator]
    checkpointer: Checkpointer
    epoch: Optional[int] = None
    global_step: int = 0

    global_train_examples_seen_this_epoch: int = 0
    """Tracks the global number of training examples seen in the current epoch for the purpose of restoring
    the data loader position on restarts."""

    global_train_tokens_seen: int = 0
    """Tracks the global total number of tokens trained on."""

    checkpoints: List[Path] = field(default_factory=list)
    unsharded_checkpoints: List[Path] = field(default_factory=list)
    ephemeral_checkpoints: List[Path] = field(default_factory=list)
    min_train_loss: float = float("inf")
    cur_train_loss: float = float("inf")
    loss_fn: Callable[..., torch.Tensor] = field(default_factory=lambda: cross_entropy_loss)  # type: ignore
    beaker_logger: BeakerLogger = None
    last_sharded_checkpoint_step: Optional[int] = None
    last_unsharded_checkpoint_step: Optional[int] = None
    _train_metrics: Any = None
    _start_time: float = 0.0
    _start_step: Optional[int] = None
    _train_start_time: Optional[float] = None
    _gc_init_state: bool = True
    _cancelled: bool = False
    _cancel_reason: Optional[str] = None
    _global_batch_size_average: List[float] = field(default_factory=list)
    _data_worker_states: Optional[Dict[int, WorkerState]] = field(default_factory=dict)

    def __post_init__(self):
        # If save folder is a local directory, make sure we're using a shared filesystem.
        if not is_url(self.cfg.save_folder) and get_fs_local_rank() != get_global_rank():
            raise OLMoConfigurationError(
                "Checkpointing to a local directory requires a shared filesystem. "
                "If you do have a shared filesystem please set the env var 'OLMO_SHARED_FS=1' "
                "or set 'FS_LOCAL_RANK' to the global rank for each process."
            )

        dp_process_group = get_dp_process_group(self.mesh) if self.mesh is not None else None
        self.dp_world_size = get_world_size(dp_process_group) if dp_process_group is not None else get_world_size()
        self.cp_degree = get_world_size() // self.dp_world_size
        self.cp_enabled = self.cp_degree > 1

        self._train_metrics = LossMetrics(self.device, reduce_loss_metrics_manually=self.cp_enabled)

        if not self.cfg.inf_eval_config.distributed:
            log.info("Setting up inference model...")
            with torch.device("cpu"):
                with torch.device('meta'):
                    olmo_model = self.cfg.model.build_model()
                olmo_model.warmup_cache(self.device)
                self.model = olmo_model

        if self.evaluators:
            assert len(set(x.label for x in self.evaluators)) == len(self.evaluators), "non-unique eval labels"
        if self.inference_evaluators:
            assert len(set(x.label for x in self.inference_evaluators)) == len(self.inference_evaluators), "non-unique eval labels"

        if self.cfg.fused_loss:
            import flash_attn
            from flash_attn.ops.triton.cross_entropy import (  # type: ignore
                cross_entropy_loss,
            )

            # The `ignored_index` parameter of `cross_entropy_loss` was changed to `ignore_index` in v2.5.8 with commit https://github.com/Dao-AILab/flash-attention/commit/ec6d22143b5d375e253b2ebfc563b26a43f43684
            ce_loss_use_ignore_index_param = version.parse(flash_attn.__version__) >= version.parse("2.5.8")

            def fused_loss_fn(
                logits, labels, ignore_index: int = -100, reduction: str = "mean",
                compute_z_loss: bool = False, z_loss_scale=1
            ):
                if ce_loss_use_ignore_index_param:
                    ignore_index_kwarg = {"ignore_index": ignore_index}
                else:
                    ignore_index_kwarg = {"ignored_index": ignore_index}

                loss, z_loss = cross_entropy_loss(
                    logits,
                    labels,
                    label_smoothing=0.0,
                    logit_scale=1.0,
                    lse_square_scale=z_loss_scale if compute_z_loss else 0.0,
                    inplace_backward=False,
                    process_group=None,
                    **ignore_index_kwarg,
                )

                mask = labels != ignore_index

                if reduction == "mean":
                    loss = loss.sum() / mask.sum()
                elif reduction == "sum":
                    loss = loss.sum()
                else:
                    loss = loss

                if not compute_z_loss:
                    return loss, None

                if reduction == "mean":
                    z_loss = z_loss.sum() / mask.sum()
                elif reduction == "sum":
                    z_loss = z_loss.sum()
                else:
                    z_loss = z_loss

                return loss, z_loss

            self.loss_fn = fused_loss_fn

        if self.cfg.compile_loss:
            if torch.cuda.is_available():
                self._loss_fn = torch.compile(self.loss_fn, dynamic=self.cfg.response_logits_only)
            else:
                log.warning(
                    "compile_loss was set to True, but CUDA is not available. Compiling only works with CUDA. Ignoring."
                )

    @property
    def dataset(self) -> IterableDataset:
        return self.train_loader

    @property
    def tokens_per_batch(self) -> int:
        return self.cfg.global_train_batch_size * self.cfg.model.max_sequence_length

    @property
    def batches_per_epoch(self) -> int:
        return self.dataset.total_size // self.cfg.global_train_batch_size

    @property
    def max_epochs(self) -> int:
        if isinstance(self.cfg.max_duration, str) and self.cfg.max_duration.endswith("ep"):
            return int(self.cfg.max_duration[:-2].strip())
        else:
            return 1

    @property
    def max_steps(self) -> int:
        if isinstance(self.cfg.max_duration, int):
            return self.cfg.max_duration
        elif isinstance(self.cfg.max_duration, str):
            if self.cfg.max_duration.endswith("T"):
                # convert to float *first* to handle scientific notation
                max_tokens = int(float(self.cfg.max_duration[:-1].strip()))
                tokens_remaining = max(max_tokens - self.global_train_tokens_seen, 0)
                steps_remaining = tokens_remaining // self.tokens_per_batch
                return self.global_step + steps_remaining
            elif self.cfg.max_duration.endswith("ep"):
                max_epochs = int(self.cfg.max_duration[:-2].strip())
                return max_epochs * self.batches_per_epoch
            else:
                # convert to float *first* to handle scientific notation
                return int(float(self.cfg.max_duration))
        else:
            raise TypeError(f"expected int or str for 'max_duration', found {type(self.cfg.max_duration)}")

    @property
    def max_tokens(self) -> int:
        if isinstance(self.cfg.max_duration, int):
            return (
                self.global_train_tokens_seen
                + max(self.cfg.max_duration - self.global_step, 0) * self.tokens_per_batch
            )
        elif isinstance(self.cfg.max_duration, str):
            if self.cfg.max_duration.endswith("T"):
                # convert to float *first* to handle scientific notation
                return int(float(self.cfg.max_duration[:-1].strip()))
            elif self.cfg.max_duration.endswith("ep"):
                max_epochs = int(self.cfg.max_duration[:-2].strip())
                return max_epochs * self.batches_per_epoch * self.tokens_per_batch
            else:
                # convert to float *first* to handle scientific notation
                return (
                    self.global_train_tokens_seen
                    + max(int(float(self.cfg.max_duration)) - self.global_step, 0) * self.tokens_per_batch
                )
        else:
            raise TypeError(f"expected int or str for 'max_duration', found {type(self.cfg.max_duration)}")

    @property
    def scheduler_current(self) -> int:
        if self.cfg.scheduler.units == SchedulerUnits.steps:
            return self.global_step
        elif self.cfg.scheduler.units == SchedulerUnits.tokens:
            return self.global_train_tokens_seen
        else:
            raise NotImplementedError(self.cfg.scheduler.units)

    @property
    def scheduler_max(self) -> int:
        if self.cfg.scheduler.units == SchedulerUnits.steps:
            return self.max_steps
        elif self.cfg.scheduler.units == SchedulerUnits.tokens:
            return self.max_tokens
        else:
            raise NotImplementedError(self.cfg.scheduler.units)

    def trainer_state_dict(self) -> Dict[str, Any]:
        if self.cfg.data.packing and self.cfg.data.packing.track_packing_state:
            # Get a checkpoint that will preserve the packing state
            # We broadcast the dataloader checkpoint to all processes since it is
            # light-weight and that will ensure things will work if we resume from
            # a different world size
            worker_states = list(self._data_worker_states.values())
            if get_world_size() > 1:
                states = [None] * get_world_size()
                dist.all_gather_object(states, worker_states)
                worker_states = flatten_lists(states)
            num_workers = self.cfg.data.num_workers or 1
            data_checkpoint = IterableDataMixtureCheckpoint(
                worker_states, get_world_size(), num_workers, next_worker_id=self.global_step % num_workers)
        else:
            data_checkpoint = None

        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "global_train_examples_seen_this_epoch": self.global_train_examples_seen_this_epoch,
            "global_train_tokens_seen": self.global_train_tokens_seen,
            "world_size": get_world_size(),
            "num_workers": self.cfg.data.num_workers,
            "checkpoints": self.checkpoints,
            "unsharded_checkpoints": self.unsharded_checkpoints,
            "ephemeral_checkpoints": self.ephemeral_checkpoints,
            "data_checkpoint": data_checkpoint,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(),
            },
        }

    def load_trainer_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Checkpoint paths.
        normalized_save_folder = normalize_path(self.cfg.save_folder)

        def _is_ours(filename):
            return normalize_path(filename).startswith(normalized_save_folder)

        self.checkpoints = [
            path for path in state_dict["checkpoints"] if _is_ours(path)]
        self.unsharded_checkpoints = [
            path for path in state_dict["unsharded_checkpoints"] if _is_ours(path)]
        self.ephemeral_checkpoints = [
            path for path in state_dict.get("ephemeral_checkpoints", []) if _is_ours(path)]

        # Dataset / dataloader position.
        checkpoint_epoch = state_dict.get("epoch", 0)
        self.global_step = state_dict["global_step"]
        self.global_train_examples_seen_this_epoch = state_dict["global_train_examples_seen_this_epoch"]
        self.global_train_tokens_seen = state_dict["global_train_tokens_seen"]

        if not self.cfg.restore_dataloader:
            self.epoch = 0
            self.global_train_tokens_seen = 0
            self.global_train_examples_seen_this_epoch = 0
        elif self.epoch is None:
            self.epoch = checkpoint_epoch
        elif checkpoint_epoch != self.epoch:
            log.info(f"Starting new epoch (epoch = {self.epoch})")
            self.global_train_examples_seen_this_epoch = 0

        if self.cfg.fast_forward_batches:
            log.info(f"Fast-forwarding data loader by {self.cfg.fast_forward_batches:,d} steps")
            # Technically we don't "see" these batches that we fast-forward through, but we use
            # this variable to update the position of the dataset so we need to include them here.
            self.global_train_examples_seen_this_epoch += (
                self.cfg.fast_forward_batches * self.cfg.global_train_batch_size
            )
            # NOTE: on the other hand we don't add anything to 'self.global_train_tokens_seen' here because
            # that variable is meant to track the actual number of tokens trained on.

        if state_dict.get("data_checkpoint"):
            assert isinstance(self.dataset.dataset, IterableDatasetMixture)
            log.info(f"Data loader restoring from checkpoint...")
            checkpoint = state_dict["data_checkpoint"]
            self.dataset.dataset.resume_from = checkpoint
            if (
                checkpoint.world_size != get_world_size() or
                checkpoint.num_workers != (1 or self.cfg.data.num_workers)
            ):
                log.warning("Num workers/world size has changed, the order of "
                            "future examples will be changed")

        elif self.global_train_examples_seen_this_epoch > 0:
            assert isinstance(self.dataset.dataset, IterableDatasetMixture)
            log.info(f"Data loader will start at instance index {self.global_train_examples_seen_this_epoch:,d}")
            self.dataset.dataset.resume_from_index = self.global_train_examples_seen_this_epoch

        # RNG states.
        if "rng" in state_dict and state_dict.get("world_size", get_world_size()) == get_world_size():
            log.info("Restoring RNG states...")
            rng_state = state_dict["rng"]
            self.restore_rng_state(rng_state)
        else:
            log.warning(
                "Trainer will not restore RNG states since the RNG states in the checkpoint are missing or invalid. "
                "This typically happens when restoring from an unsharded checkpoint or a checkpoint that was saved "
                "with a different world size. If that's the case you can safely ignore this warning."
            )

    def restore_rng_state(self, rng_state: Dict[str, Any]) -> None:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        torch.cuda.set_rng_state(rng_state["cuda"])

    def save_checkpoint(self, checkpoint_type: CheckpointType, optim=True) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        if checkpoint_type == CheckpointType.sharded:
            suffix = ""
            current_checkpoints = self.checkpoints
            num_checkpoints_to_keep = self.cfg.save_num_checkpoints_to_keep
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            suffix = ""
            current_checkpoints = self.ephemeral_checkpoints
            num_checkpoints_to_keep = 1
        else:
            raise NotImplementedError(checkpoint_type)

        self.last_sharded_checkpoint_step = self.global_step

        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)

        checkpoint_dir = join(self.cfg.save_folder, f"step{self.global_step}{suffix}")
        current_checkpoints.append(checkpoint_dir)

        # torch.distributed.checkpointing can experience weird transients errors, where one
        # process will hit "800 operation not permitted"
        # barrier/synchronize/gc to try and fix the issue
        gc_cuda()
        barrier()
        torch.cuda.synchronize(self.device)

        self.checkpointer.save(
            checkpoint_dir,
            self.fsdp_model,
            self.optim if optim else None,
            self.trainer_state_dict(),
            config=self.cfg,
        )

        self.remove_checkpoints(current_checkpoints, num_checkpoints_to_keep)
        barrier()
        gc_cuda()
        return checkpoint_dir

    def restore_checkpoint(
        self,
        load_path: PathOrStr,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
        allow_missing_keys: bool = False
    ):
        trainer_state = self.checkpointer.load(
            load_path, self.fsdp_model, self.optim,
            load_optimizer_state=load_optimizer_state,
            load_trainer_state=load_trainer_state,
            # allow_missing_keys=allow_missing_keys
        )
        if load_trainer_state:
            self.load_trainer_state_dict(trainer_state)
            if self.global_step >= self.cfg.stop_at:
                raise ValueError(f"Checkpointed it at {self.global_step}, but stop_at is {self.cfg.stop_at}")
            if self.global_step >= self.max_steps:
                raise ValueError(f"Checkpointed it at {self.global_step}, but max steps is {self.max_steps}")
        gc_cuda()
        barrier()

    def _remove_sharded_checkpoint(self, idx: int, checkpoints: List[Path]):
        oldest_checkpoint = checkpoints.pop(idx)
        barrier()
        if get_fs_local_rank() == 0:
            clear_directory(oldest_checkpoint)
        barrier()

    def remove_checkpoints(self, current_checkpoints, num_checkpoints_to_keep):
        if num_checkpoints_to_keep > 0:
            while len(current_checkpoints) > num_checkpoints_to_keep:
                self._remove_sharded_checkpoint(0, current_checkpoints)

    def move_to_device(self, batch, device):
        return move_to_device(batch, device)

    def _log_timing(self, name: str) -> None:
        """Log the most recent timing for an operation."""
        elapsed = self._timer_manager.get_last(name)
        if elapsed is not None:
            log.info(f"[TIMING] {name}: {elapsed*1000:.2f}ms ({elapsed:.4f}s)")


    def get_labels(self, batch: Dict[str, Any]) -> torch.Tensor:
        # Labels are just input IDs shifted to the left (first item is ignored).
        labels, label_mask, attention_mask, instance_mask = (
            batch["input_ids"].clone(),
            batch.get("label_mask"),
            batch.get("attention_mask"),
            batch.get("instance_mask"),
        )
        if label_mask is not None:
            labels.masked_fill_(~label_mask, -100)
        if attention_mask is not None:
            labels.masked_fill_(attention_mask == 0.0, -100)
        if instance_mask is not None:
            labels.masked_fill_(~instance_mask.unsqueeze(-1), value=-100)
        return labels[..., 1:].contiguous()

    def model_forward(
        self, batch: Dict[str, Any], compute_z_loss: bool = False
    ) -> Tuple:
        # shape: (batch_size, seq_len, vocab_size)
        loss_masks = batch["loss_masks"]
        labels = batch["labels"]
        response_mask = (loss_masks > 0)

        # need to pass loss_masks and labels to model when using cp_degree > 1 in order to chunk them across cp ranks
        keys_to_exclude = ["loss_masks", "labels"] if not self.cp_enabled else []

        with torch.autocast("cuda", dtype=self.cfg.autocast_precision):
            model_out = self.fsdp_model(
                **{k: v for k, v in batch.items() if k not in keys_to_exclude},
                response_mask=response_mask,
                response_logits_only=self.cfg.response_logits_only
            )
            logits = model_out.logits
            # get the sharded loss masks from forward pass of the model instead
            loss_masks = model_out.loss_masks if model_out.loss_masks is not None else loss_masks
            loss_masks = (loss_masks * (loss_masks > 0)).view(-1)
            # get the sharded labels
            labels = model_out.labels if model_out.labels is not None else labels
            labels = labels.long().view(-1)
            response_mask = model_out.response_mask if model_out.response_mask is not None else response_mask

            logits_for_loss = logits.to(torch.float32).view(-1, logits.size(-1)) # for numerical stability
            if self.cfg.response_logits_only:
                loss_masks = loss_masks[response_mask.view(-1)]
                labels = labels[response_mask.view(-1)]
            else:
                labels.masked_fill_(~(loss_masks > 0), -100)
            ce_loss, z_loss = self.loss_fn(
                logits_for_loss,
                labels,
                ignore_index=-100,
                reduction="none",
                compute_z_loss=compute_z_loss,
                z_loss_scale=self.cfg.softmax_auxiliary_loss_scale,
            )

        loss_masks = loss_masks.to(dtype=ce_loss.dtype)
        ce_loss = torch.dot(ce_loss, loss_masks)
        z_loss = torch.dot(z_loss, loss_masks) if z_loss is not None else None

        return ce_loss, z_loss, model_out

    def train_batch(self, batch: Dict[str, Any], compute_metrics) -> Tuple[torch.Tensor, Optional[Dict]]:
        # Split into micro-batches.
        micro_batches = self.split_batch(batch)
        if self.cp_enabled:
            batch["loss_masks"] = batch["loss_masks"].to(self.device)
        loss_masks = batch["loss_masks"] * (batch["loss_masks"] > 0)
        if self.cfg.batch_divisor == BatchDivisor.global_batch:
            batch_size_in_tokens = loss_masks.sum()
            dist.all_reduce(batch_size_in_tokens)
            batch_size_in_tokens.div_(get_world_size())
        elif self.cfg.batch_divisor == BatchDivisor.global_batch_average:
            batch_size_in_tokens = loss_masks.sum()
            dist.all_reduce(batch_size_in_tokens)
            batch_size_in_tokens.div_(get_world_size())
            self._global_batch_size_average.append(batch_size_in_tokens.item())
            batch_size_in_tokens = np.mean(self._global_batch_size_average)
        elif self.cfg.batch_divisor == BatchDivisor.device_batch:
            batch_size_in_tokens = loss_masks.sum()
        else:
            raise ValueError()
        del batch  # in case this helps reduce memory
        assert batch_size_in_tokens > 0

        total_loss = torch.tensor(0.0, device=self.device)
        for m_b, micro_batch in enumerate(micro_batches):
            ce_loss, z_loss, model_out = self.model_forward(
                micro_batch, compute_z_loss=self.cfg.softmax_auxiliary_loss)
            if compute_metrics:
                self._train_metrics.update(micro_batch, model_out, ce_loss, z_loss)

            # In case this helps with memory utilization.
            del micro_batch

            ce_loss = ce_loss.sum() / batch_size_in_tokens

            # Get loss to optimize for.
            if self.cfg.softmax_auxiliary_loss:
                z_loss = z_loss.sum() / batch_size_in_tokens
                loss = ce_loss + z_loss
            else:
                loss = ce_loss
            loss = loss * self.cp_degree  # scale loss to account for gradient averaging in FSDP

            if model_out.metrics is not None:
                if "AuxLoss" in model_out.metrics:
                    loss += model_out.metrics["AuxLoss"] / len(micro_batches)
                if "token_losses" in model_out.metrics:
                    token_losses = model_out.metrics.pop("token_losses")
                    if torch.isnan(token_losses) or torch.isinf(token_losses):
                        raise ValueError()
                    loss += token_losses / batch_size_in_tokens

                if self.cfg.saliency_score_loss_wt is not None:
                    if model_out.metrics.get('saliency_difference', None) is not None:
                        loss += self.cfg.saliency_score_loss_wt * model_out.metrics['saliency_difference'] / len(micro_batches)

                    elif model_out.internal.get('pred_saliency', None) is not None and model_out.internal.get('gt_saliency', None) is not None:
                        pred_saliency = model_out.internal.pop('pred_saliency')
                        gt_saliency = model_out.internal.pop('gt_saliency')

                        valid = gt_saliency != -100
                        gt_saliency[~valid] = 0.0
                        pred_saliency[~valid] = 0.0

                        saliency_loss = F.binary_cross_entropy(pred_saliency, gt_saliency, reduction="sum") / torch.sum(valid * 1.0)
                        loss += self.cfg.saliency_score_loss_wt * saliency_loss / len(micro_batches)

                if self.cfg.frame_score_loss_wt is not None and model_out.metrics.get("embedding_scores", None) is not None:
                    assert self.cfg.frame_score_loss_target is not None
                    embedding_scores = model_out.metrics['embedding_scores']
                    target = torch.tensor(self.cfg.frame_score_loss_target, device=embedding_scores.device, dtype=embedding_scores.dtype)
                    if self.cfg.frame_score_loss_type == "l1":
                        micro_batch_loss = l1_loss(embedding_scores, target)
                    elif self.cfg.frame_score_loss_type == "mse":
                        micro_batch_loss = mse_loss(embedding_scores, target)
                    elif self.cfg.frame_score_loss_type == "rmse":
                        micro_batch_loss = torch.sqrt(mse_loss(embedding_scores, target) + 1e-8)
                    else:
                        raise ValueError(f"Unsupported frame_score_loss_type: {self.cfg.frame_score_loss_type}")
                    loss += self.cfg.frame_score_loss_wt * micro_batch_loss / len(micro_batches)

            del model_out

            # Run backward pass.
            loss.backward()
            total_loss += loss.detach()

        return total_loss

    def train_step(self, batch: Dict[str, Any], compute_metrics: bool = True) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # Zero-gradients.
        self.optim.zero_grad(set_to_none=True)

        if not self.cp_enabled:
            # Move tensors to the right device.
            batch = self.move_to_device(batch, self.device)

        # Run forward-backward pass
        if compute_metrics:
            self._train_metrics.reset()
        loss = self.train_batch(batch, compute_metrics)

        if torch.isnan(loss) or torch.isinf(loss):
            # Log the batch into a file for debugging
            # save_debug_batch(batch, self.cfg.save_folder, self.global_step, loss.item())
            raise RuntimeError(f"NaN or Inf loss detected: {loss.item()}")

        if compute_metrics:
            metrics = {f"train/{k}": v for k, v in self._train_metrics.compute().items()}
        else:
            metrics = {}

        should_log_optim_metrics_this_step = self.should_log_optim_metrics_this_step()
        if should_log_optim_metrics_this_step:
            # No current implementation of per-parameter metrics because I am not sure the
            # old very complex one makes sense anymore
            raise NotImplementedError()

        # Clip gradient norms, norms are clipped per group name
        # Note group name might have multiple optimizer param groups
        param_norm_groups = defaultdict(list)
        for group in self.optim.param_groups:
            param_norm_groups[group["group_name"]].append(group)
        optim_metrics = {}
        grad_norms = []

        max_grad_norm = self.scheduler.get_max_grad_norm(
            self.cfg.max_grad_norm, self.scheduler_current, self.scheduler_max)
        if self.cfg.max_grad_norm_ratio is not None:
            raise NotImplementedError()
        if max_grad_norm is not None:
            for group_name, groups in param_norm_groups.items():
                params = flatten_lists(group["params"] for group in groups)
                grad_norm = clip_grad_norm(params, max_grad_norm=max_grad_norm)
                grad_norms.append(grad_norm)
                optim_metrics[f"{group_name}_grad_norm"] = grad_norm

        # Adjust the learning rate.
        initial_lr_dict = {
            "connector": self.cfg.optimizer.connector_learning_rate,
            "vit": self.cfg.optimizer.vit_learning_rate,
            "llm": self.cfg.optimizer.llm_learning_rate,
            "frame_selector": self.cfg.optimizer.frame_selector_learning_rate,
        }
        for group in self.optim.param_groups:
            group["lr"] = self.scheduler.get_lr(
                initial_lr_dict[group["group_name"]],
                self.scheduler_current,
                self.scheduler_max,
                group["group_name"],
            )

        # Optimizer step.
        self.optim.step()

        # Collect metrics and check for NaN loss.
        # NOTE: this involves a bunch of host-device syncs so we wait until the last moment to do this.
        for key, value in optim_metrics.items():
            metrics[f"optim/{key}"] = value.item()
        self.cur_train_loss = loss.item()
        self.min_train_loss = min(self.min_train_loss, self.cur_train_loss)
        return metrics

    def split_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        microbatch_size = self.cfg.device_train_microbatch_size
        batch_size = batch["input_ids"].shape[0]
        if batch_size <= microbatch_size:
            return [batch]
        else:
            micro_batches = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    micro_batches[key] = value.split(microbatch_size, dim=0)
                elif isinstance(value, list):
                    micro_batches[key] = [
                        value[microbatch_size * i : microbatch_size * i + microbatch_size]
                        for i in range(math.ceil(batch_size / microbatch_size))
                    ]
                else:
                    raise ValueError(f"unexpected item in batch: '{key}={value}'")
            return [
                {key: value[i] for key, value in micro_batches.items()}  # type: ignore
                for i in range(len(micro_batches["input_ids"]))
            ]

    def system_metrics(self) -> Dict[str, float]:
        metrics = {}
        if self.global_step < 3 or self.global_step % 10 == 0:
            peak_gpu_mb = peak_gpu_memory()
            if peak_gpu_mb is not None:
                metrics["System/Peak GPU Memory (MB)"] = peak_gpu_mb
        return metrics

    def log_metrics_to_console(self, prefix: str, metrics: Dict[str, float]):
        def format_float(value: float) -> str:
            if value < 0.0001:
                return str(value)  # scientific notation
            elif value > 1000:
                return f"{int(value):,d}"
            elif value > 100:
                return f"{value:.1f}"
            elif value > 10:
                return f"{value:.2f}"
            elif value > 1:
                return f"{value:.3f}"
            else:
                return f"{value:.4f}"

        log.info(
            f"{prefix}\n"
            + "\n".join(
                [
                    f"    {name}={format_float(value)}"
                    for name, value in metrics.items()
                    # there's too many optimizer metrics
                    # also skip non-float wandb.Metrics from inference evaluators
                    if (
                    isinstance(value, (int, float)) and (
                    name == "optim/total_grad_norm"
                    or (not name.startswith("optim/") and not name.startswith("batch/"))
                ))
                ]
            )
        )

    def should_log_optim_metrics_this_step(self) -> bool:
        if self.cfg.wandb is None:
            # We only log optimizer-specific metrics to W&B, since there are usually too many metrics
            # to log to the console.
            return False
        optim_log_interval = self.cfg.optimizer.metrics_log_interval
        if optim_log_interval is None:
            optim_log_interval = self.cfg.wandb.log_interval
        elif optim_log_interval <= 0:
            return False
        else:
            optim_log_interval = max(optim_log_interval, self.cfg.wandb.log_interval)
        return self.global_step % optim_log_interval == 0

    def should_log_this_step(self) -> bool:
        if self.global_step % self.cfg.console_log_interval == 0:
            return True
        elif self.cfg.wandb is not None and self.global_step % self.cfg.wandb.log_interval == 0:
            return True
        else:
            return False

    def inference_eval(self) -> Dict[str, Union[float, WBValue]]:
        self.optim.zero_grad(set_to_none=True)

        all_metrics = {}
        all_eval_t0 = time.perf_counter()
        if not self.cfg.inf_eval_config.distributed:
            log.info(f"Setting up non-FSDP model...")
            if self.cfg.inf_eval_config.offload_optim:
                for state in self.optim.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cpu()
                for group in self.optim.param_groups:
                    for k, v in group.items():
                        if torch.is_tensor(v):
                            group[k] = v.cpu()

            state_dict = get_state_dict(self.fsdp_model, [],
                                        options=StateDictOptions(full_state_dict=True))[0]
            self.model.load_state_dict(state_dict, assign=True)
            model = self.model

            if self.cfg.inf_eval_config.offload_model:
                self.fsdp_model.cpu()
        else:
            model = self.fsdp_model
        model.eval()

        all_eval_t0 = time.perf_counter()
        for evaluator in self.inference_evaluators:
            t0 = time.perf_counter()
            log.info(f"Running evaluation for '{evaluator.label}'...")
            if self.cfg.save_folder and self.cfg.save_inloop_predictions:
                # Add a metric that will save our predictions
                output_dir = join(self.cfg.save_folder, "inloop-predictions", f"step{self.global_step}-{evaluator.label}")
                save_inloop_metric = SavePredictions(output_dir, save_tokens=False)
                evaluator = dataclasses.replace(evaluator, evaluator=dataclasses.replace(
                    evaluator.evaluator, metrics=evaluator.evaluator.metrics + [save_inloop_metric]
                ))
            dataset_metrics = evaluator.run(
                model,
                device=self.device,
                autocast_precision=self.cfg.autocast_precision,
                is_distributed=self.cfg.inf_eval_config.distributed,
                pbar=False,
            )
            self.log_metrics_to_console(f"{evaluator.label}", dataset_metrics)
            all_metrics.update({f"{evaluator.label}/{k}": v for k, v in dataset_metrics.items()})
            log.info(f"Eval for '{evaluator.label}' done in {time.perf_counter()-t0:0.1f} seconds")

        if not self.cfg.inf_eval_config.distributed:
            log.info(f"Offloading non-FSDP model...")
            self.model.to("meta")
            if self.cfg.inf_eval_config.offload_model:
                self.fsdp_model.cuda()
            if self.cfg.inf_eval_config.offload_optim:
                for state in self.optim.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
                for group in self.optim.param_groups:
                    for k, v in group.items():
                        if torch.is_tensor(v):
                            group[k] = v.cpu()
        if len(self.inference_evaluators) > 1:
            log.info(f"All evals took done in {time.perf_counter()-all_eval_t0:0.1f} seconds")
        return all_metrics

    def loss_eval(self) -> Dict[str, Union[float, WBValue]]:
        self.optim.zero_grad(set_to_none=True)
        self.fsdp_model.eval()
        eval_metrics = {}
        for evaluator in self.evaluators:
            t0 = time.perf_counter()
            log.info(f"Running evaluation for '{evaluator.label}'...")
            metrics = evaluator.run(
                self.fsdp_model, self.device,
                autocast_precision=self.cfg.autocast_precision,
                loss_fn=self.loss_fn,
                cp_enabled=self.cp_enabled,
            )
            eval_metrics.update({f"{evaluator.label}/{k}": v for k, v in metrics.items()})
            log.info(f"Eval for '{evaluator.label}' done in {time.perf_counter()-t0:0.1f} seconds")
            self.log_metrics_to_console(f"{evaluator.label}", metrics)
        return eval_metrics

    def _handle_interrupt(self, signalnum, stack_frame):
        del stack_frame

        signame: Optional[str] = None
        if signalnum == signal.SIGTERM:
            signame = "SIGTERM"
        elif signalnum == signal.SIGINT:
            signame = "SIGINT"

        if signame is not None:
            msg = f"{signame} received"
        else:
            msg = f"Sig({signalnum}) received"

        log.warning(msg)
        self._cancelled = True
        self._cancel_reason = msg

    def check_if_cancelled(self) -> Tuple[bool, int]:
        should_cancel = self._cancelled
        cancel_reason = self._cancel_reason
        extra_steps = self.cfg.extra_steps_after_cancel
        if get_global_rank() == 0 and not should_cancel:
            if self.cfg.time_limit is not None and time.time() - self._start_time >= self.cfg.time_limit:
                # First check if we've reached the training time limit.
                should_cancel = True
                cancel_reason = "time limit reached"

        run_canceled = synchronize_flag(should_cancel, self.device)
        if run_canceled:
            if cancel_reason is None:
                if extra_steps > 0:
                    log.warning(f"Run canceled, stopping in {extra_steps} more steps...")
                else:
                    log.warning("Run canceled")
            else:
                if extra_steps > 0:
                    log.warning(f"Run canceled due to {cancel_reason}, stopping in {extra_steps} more steps...")
                else:
                    log.warning(f"Run canceled due to {cancel_reason}")
        return run_canceled, extra_steps

    def get_eta(self) -> str:
        if self._train_start_time is None:
            return "???"
        if self.cfg.stop_at:
            steps_left = self.cfg.stop_at - self.global_step
        else:
            steps_left = self.max_steps - self.global_step
        time_passed = (time.monotonic() - self._train_start_time)
        steps_per_second = time_passed / (self.global_step - self._start_step)
        seconds_left = steps_per_second * steps_left
        # Round off to minutes to make it the string easier to parse
        minutes_left = 1 + seconds_left // 60
        return format_timedelta(timedelta(minutes=minutes_left))

    def fit(self):
        if self.cfg.stop_after is not None:
            if self.cfg.stop_at is None:
                self.cfg.stop_at = self.global_step + self.cfg.stop_after
            else:
                self.cfg.stop_at = min(self.cfg.stop_at, self.global_step + self.cfg.stop_after)

        self._start_time = time.time()
        self._gc_init_state = gc.isenabled()  # cache if garbage collection is enabled, reset on close.

        # Disable automatic garbage collection, FSDP doesn't work well with it.
        if self.cfg.gen1_gc_interval is not None:
            gc.disable()

        if self.cfg.load_path is not None and self.global_step > 0 and self.cfg.eval_on_load:
            eval_metrics = self.loss_eval()
            if wandb.run is not None:
                wandb.log(eval_metrics, step=self.global_step)

            eval_metrics = self.inference_eval()
            if wandb.run is not None:
                wandb.log(eval_metrics, step=self.global_step)
            torch.cuda.empty_cache()

        # Set model to 'train' mode.
        self.fsdp_model.train()

        # Initialize monitors.
        speed_monitor = SpeedMonitor(self.cfg.speed_monitor)
        lr_monitor = LRMonitor(self.optim)
        batch_monitor = BatchStatsMonitor()

        # Log system metrics at the start of training.
        sys_metrics = self.system_metrics()
        if sys_metrics:
            self.log_metrics_to_console("Pre-train system metrics", sys_metrics)
            if wandb.run is not None:
                wandb.log(sys_metrics, step=0)

        # Python Profiler stuff
        if self.cfg.python_profiling:
            python_profiler = cProfile.Profile()
        else:
            python_profiler = None

        # PyTorch Profiler stuff
        if self.cfg.torch_profiling and get_global_rank() == 0:
            from torch.profiler import schedule

            profiling_schedule = schedule(wait=1, warmup=5, active=3, repeat=1)

            def on_trace_ready(p):
                profiler_output_dir = Path(self.cfg.save_folder) / "profiler"
                profiler_output_dir.mkdir(exist_ok=True)

                output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=32)
                log.info(f"Profile by total GPU time at step {p.step_num}:\n{output}")
                output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=32)
                log.info(f"Profile by total CPU time at step {p.step_num}:\n{output}")

                p.export_chrome_trace(
                    str(trace_path := (profiler_output_dir / f"{p.step_num}.chrome_trace.json.gz"))
                )

            from torch.profiler import ProfilerActivity

            torch_profiler = torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=True,
                schedule=profiling_schedule,
                on_trace_ready=on_trace_ready,
            )
            del profiling_schedule
        else:
            import contextlib

            torch_profiler = contextlib.nullcontext()

        # Train.
        first_batch: bool = True
        cancel_initiated: bool = False
        stop_at: Optional[int] = self.cfg.stop_at
        save_checkpoints: bool = True
        self._start_step = self.global_step
        with torch_profiler as p:
            for epoch in range(self.epoch or 0, self.max_epochs):
                for batch in self.train_loader:
                    # Bookkeeping.
                    batch_size, seq_len = batch["input_ids"].shape
                    global_batch_size = batch_size * self.dp_world_size  # assumes batch size equal across ranks
                    self.global_step += 1
                    # log.info(f"{self.global_step}")
                    # continue
                    if hasattr(self.fsdp_model, "_global_step"):
                        self.fsdp_model._global_step = self.global_step
                    self.global_train_examples_seen_this_epoch += global_batch_size
                    self.global_train_tokens_seen += global_batch_size * seq_len

                    speed_monitor.batch_start(
                        self.global_train_tokens_seen,
                        (batch_size * seq_len) / self.cp_degree,  # num tokens in batch for this device
                        (batch["loss_masks"] > 0).sum() / self.cp_degree,  # approximate num loss tokens in batch for this device
                        # We start monitoring speed after the first batch since the first
                        # batch might be an outlier due to compiling and other initialization overhead.
                        record=not first_batch,
                        )
                    batch_monitor.log_batch(batch)

                    if self.cfg.data.packing and self.cfg.data.packing.track_packing_state:
                        # Update the data worker states for dataloader checkpointing
                        for state in batch.pop("data_worker_state"):
                            if state.worker_global_id not in self._data_worker_states:
                                self._data_worker_states[state.worker_global_id] = state
                            else:
                                cur_version = self._data_worker_states[state.worker_global_id].version
                                if cur_version < state.version:
                                    self._data_worker_states[state.worker_global_id] = state

                    should_log_this_step = self.should_log_this_step()

                    if self._train_start_time is None:
                        # Start timing after the first step so we don't count warm-up
                        self._train_start_time = time.monotonic()

                    # Run train step on batch.
                    metrics = self.train_step(batch, compute_metrics=should_log_this_step)

                    # Maybe collect other metrics.
                    if should_log_this_step:
                        metrics.update(speed_monitor.check())
                        metrics.update(self.system_metrics())
                        metrics.update(batch_monitor.check(self.device))
                        metrics.update(lr_monitor.check())

                    # Do beaker logging
                    if (
                        self.beaker_logger and
                        (
                            (self.global_step % self.beaker_logger.log_interval == 0) or
                            # Log on step 0 so we can tell the model is done initializing
                            (self.beaker_logger.log_interval == 0 and self.global_step == 1)
                        )
                    ):
                        self.beaker_logger.log_progress(self.global_step, stop_at, self.get_eta())

                    # Log metrics to console.
                    if self.global_step % self.cfg.console_log_interval == 0:
                        header = f"[step={self.global_step}/{self.max_steps}, eta={self.get_eta()}]"
                        if get_global_rank() == 0:
                            self.log_metrics_to_console(header, metrics)
                        else:
                            log.info(header)

                    # Log metrics to W&B.
                    if (
                        wandb.run is not None
                        and self.cfg.wandb is not None
                        and self.global_step % self.cfg.wandb.log_interval == 0
                    ):
                        wandb.log(metrics, step=self.global_step)

                    # Check if/when run should be canceled.
                    if not cancel_initiated and self.global_step % self.cfg.canceled_check_interval == 0:
                        cancel_initiated, extra_steps = self.check_if_cancelled()
                        if cancel_initiated:
                            stop_at = (
                                self.global_step + extra_steps
                                if stop_at is None
                                else min(self.global_step + extra_steps, stop_at)
                            )

                    # Maybe save sharded checkpoint.
                    done_training = stop_at is not None and self.global_step >= stop_at
                    if save_checkpoints and (
                        cancel_initiated
                        or (
                            (self.global_step == self.cfg.save_at
                             or (self.global_step % self.cfg.save_interval == 0)
                            )
                            and self.cfg.save_num_checkpoints_to_keep != 0
                        )
                    ):
                        log.info("Saving checkpoint...")
                        checkpoint_path = self.save_checkpoint(
                            CheckpointType.sharded,
                            optim=(not done_training or self.cfg.save_final_optim)
                        )
                        log.info(f"Checkpoint saved to {checkpoint_path}")

                        # Remove any ephemeral checkpoints.
                        self.remove_checkpoints(self.ephemeral_checkpoints, 0)

                        # Reset speed monitor so that we don't count the time taken to save checkpoints.
                        speed_monitor.reset()

                        # If the run was just canceled this will be the final checkpoint.
                        if cancel_initiated:
                            save_checkpoints = False
                    elif (
                        self.cfg.save_interval_ephemeral is not None
                        and self.global_step % self.cfg.save_interval_ephemeral == 0
                    ):
                        log.info("Saving ephemeral checkpoint...")
                        checkpoint_path= self.save_checkpoint(CheckpointType.sharded_ephemeral)
                        log.info(f"Checkpoint saved to {checkpoint_path}")

                        # Reset speed monitor so that we don't count the time taken to save checkpoints.
                        speed_monitor.reset()

                    # Maybe run evaluations.
                    last_step = stop_at and (self.global_step >= stop_at)
                    if not cancel_initiated and self.cfg.eval_interval > 0 and (
                        (self.global_step % self.cfg.eval_interval == 0) or
                        (last_step and self.cfg.eval_on_last_step) or
                        (self.global_step in self.cfg.eval_on)
                    ):
                        eval_metrics = self.loss_eval()

                        # Log metrics to W&B.
                        if wandb.run is not None:
                            wandb.log(eval_metrics, step=self.global_step)

                        # Reset speed monitor so that we don't count the time taken to run evaluations.
                        speed_monitor.reset()

                        # Reset model to 'train' mode.
                        self.fsdp_model.train()

                    if not cancel_initiated and (
                        self.inference_evaluators and
                        self.cfg.inf_eval_interval > 0 and
                        ((self.global_step % self.cfg.inf_eval_interval == 0) or
                         (self.cfg.eval_on_last_step and last_step) or
                         (self.global_step in self.cfg.eval_on))
                    ):
                        eval_metrics = self.inference_eval()

                        # Log metrics to W&B.
                        if wandb.run is not None:
                            wandb.log(eval_metrics, step=self.global_step)

                        # Reset speed monitor so that we don't count the time taken to run evaluations.
                        speed_monitor.reset()

                        # Reset model to 'train' mode.
                        self.fsdp_model.train()

                    # End of batch.
                    first_batch = False
                    if p is not None:
                        p.step()

                    if stop_at is not None and self.global_step >= stop_at:
                        break

                    # Run generation 1 garbage collection.
                    if self.cfg.gen1_gc_interval is not None and self.global_step % self.cfg.gen1_gc_interval == 0:
                        gc.collect(1)

                    # Python Profiler stuff
                    # We do this now, at the bottom of this loop, so we capture the work of getting the next batch.
                    if python_profiler is not None:
                        if self.global_step == 5:
                            python_profiler.enable()
                        elif self.global_step == 8:
                            python_profiler.disable()
                            python_profiler.print_stats(sort=SortKey.CUMULATIVE)
                            python_profiler = None
                else:
                    log.info("Training epoch complete")
                    self.epoch = epoch + 1
                    self.global_train_examples_seen_this_epoch = 0
                    if self.epoch < self.max_epochs:
                        self.dataset.reshuffle()
                    continue
                break

        # Save final checkpoint.
        if save_checkpoints:
            if (
                self.cfg.save_num_checkpoints_to_keep != 0
                and self.last_sharded_checkpoint_step != self.global_step
            ):
                log.info("Saving final checkpoint...")
                checkpoint_path = self.save_checkpoint(
                    CheckpointType.sharded, optim=self.cfg.save_final_optim)
                log.info(f"Checkpoint saved to {checkpoint_path}")
            if self.cfg.save_final_unsharded_checkpoint:
                log.info("Saving final unsharded checkpoint...")
                checkpoint_path = join(normalize_path(self.cfg.save_folder), f"step{self.global_step}-unsharded")
                save_unsharded(checkpoint_path, self.fsdp_model, None, self.cfg, self.cfg.save_overwrite)
                log.info(f"Checkpoint saved to {checkpoint_path}")

    def close(self, exit_code: int = 0) -> None:
        if wandb.run is not None:
            if exit_code != 0:
                log.info(f"Finishing wandb with exit code {exit_code}")
            wandb.finish(exit_code=exit_code, quiet=True)
        gc_cuda()
        if self._gc_init_state:
            gc.enable()
        else:
            gc.disable()
        if self.beaker_logger is not None and exit_code == 0:
            self.beaker_logger.finish()

    def __enter__(self) -> Trainer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        del exc_val, exc_tb
        self.close(0 if exc_type is None else 1)
