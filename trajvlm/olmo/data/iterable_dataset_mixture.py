import dataclasses
import logging
import sys
import time
from typing import List, Optional, Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Sampler
from torch.distributed.device_mesh import DeviceMesh

from olmo.data.dataset import DeterministicDataset
from olmo.data.utils import make_random_state
from olmo.torch_util import (
    get_world_size,
    get_global_rank,
    get_rank
)
from olmo.dist_util import get_dp_process_group, get_cp_mesh

log = logging.getLogger(__name__)


@dataclasses.dataclass
class WorkerState:
    """State of a DataLoader worker that is using packing"""

    worker_global_id: int
    """Global id for the worker"""

    on_example: int
    """Last example that was added to the packer"""

    skipped_example_ids: List[int]
    """Examples the packer is currently buffering"""

    @property
    def version(self):
        return self.on_example


@dataclasses.dataclass
class IterableDataMixtureCheckpoint:
    """Light-weight checkpoint that stores the state of a `IterableDatasetMixture`"""
    worker_states: List[WorkerState]
    world_size: int
    num_workers: int
    next_worker_id: int = 0

    def __post_init__(self):
        assert len(self.worker_states) == (self.world_size*self.num_workers)
        self.worker_states = sorted(self.worker_states, key=lambda x: x.worker_global_id)
        assert [x.worker_global_id for x in self.worker_states] == list(range(len(self.worker_states)))

    def seen_example(self, on_example: int) -> bool:
        """Whether the checkpointed dataloader already produced example number `on_example`"""
        state = self.worker_states[on_example % len(self.worker_states)]
        return (state.on_example >= on_example) and (on_example not in state.skipped_example_ids)


class IterableDatasetMixture(torch.utils.data.IterableDataset[Dict[str, Any]]):
    """Infinitely iterates over a mixture of datasets"""

    def __init__(
        self,
        datasets: List[DeterministicDataset],
        global_batch_size: Optional[int],
        packer: Any = None,
        mesh: DeviceMesh = None,
        mixture_rates: List[float]=None,
        seed: int = 0,
        start_index: int = 0,
        shuffle: bool = True,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        stratify: bool = False,
        worker_info=None,
        track_packing_state: bool = False,
        resume_from: Optional[IterableDataMixtureCheckpoint] = None,
    ):
        self.datasets = list(datasets)
        self.track_packing_state = track_packing_state
        if mixture_rates:
            self.mixture_rates = np.array(mixture_rates, dtype=np.float32)
        else:
            self.mixture_rates = None

        self.seed = seed
        assert seed is not None
        self.shuffle = shuffle
        self.global_batch_size = global_batch_size

        self.resume_from = resume_from
        self.resume_from_index: int = start_index

        if mesh is not None:
            dp_group = get_dp_process_group(mesh)
            self.rank = get_rank(dp_group)
            cp_pg = get_cp_mesh(mesh).get_group()
            self.cp_rank = get_rank(cp_pg)
            self.world_size = get_world_size(dp_group)
            assert self.global_batch_size % get_world_size(dp_group) == 0, \
                f"Global batch size {self.global_batch_size} not divisible by world size {get_world_size(dp_group)}"
        else:
            self.rank = rank if rank is not None else get_global_rank()
            self.world_size = world_size if world_size is not None else get_world_size()

        self.device_batch_size = global_batch_size // self.world_size
        self.stratify = stratify
        self.worker_info = worker_info  # For testing
        self.packer = packer

        self.global_batch_size = global_batch_size

    def _get_next_source(self, rng, counts) -> int:
        if len(self.datasets) == 1:
            return 0
        if self.stratify:
            return np.argmax(self.mixture_rates - counts/counts.sum())
        else:
            return rng.choice(len(self.datasets), p=self.mixture_rates)

    def _get_order(self, dataset_ix, epoch):
        shuffled_order = np.arange(len(self.datasets[dataset_ix]), dtype=np.int32)
        # Three seeds so this seed is not correlated with the seed used in dataset.get
        make_random_state(self.seed, epoch, 1).shuffle(shuffled_order)
        return shuffled_order

    def __iter__(self):
        worker_info = self.worker_info or torch.utils.data.get_worker_info()
        if worker_info:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            if (
                self.resume_from is not None and
                self.resume_from.num_workers == num_workers and
                self.world_size == self.resume_from.world_size
            ):
                # Offset the worker ids so that the first batch will be yielded by worker_id
                # `resume_first_worker_id`
                # This is needed if resuming from stateful workers since, for example, if
                # worker 1 was the last worker used before checkpointing we should start
                # with worker 2, not worker 0
                worker_id = (worker_id + self.resume_from.next_worker_id) % worker_info.num_workers
        else:
            num_workers = 1
            worker_id = 0

        global_worker_id = self.rank * num_workers + worker_id
        num_global_workers = self.world_size * num_workers

        rng = np.random.RandomState(self.seed)
        counts = np.zeros(len(self.datasets), dtype=np.int64)

        on_example = -1
        shuffled_ixs = [(None, None) for _ in self.datasets]
        while True:
            # Get the next dataset to sample, we do this in every worker, and even if we
            # are fast-forwarding, so `rng` and `counts` are synced with `on_example`
            dataset_ix = self._get_next_source(rng, counts)
            count = int(counts[dataset_ix])
            counts[dataset_ix] += 1
            on_example += 1

            # Maybe fast-forwarding past this example
            if self.resume_from:
                if self.resume_from.seen_example(on_example):
                    continue
            elif on_example < self.resume_from_index:
                continue

            # Check if the worker is assigned this example
            if self.global_batch_size and self.packer is None:
                # `DataLoader` will collect full batches one-at-a-time for each worker
                # (so global batch 0 is produced by all worker 0s on each rank)
                # To ensure each batch contains examples `on_example` to `on_example+global_batch_size`
                # we therefore have to assign example_ids->workers im blocks of size global_batch_size
                # This prevents data order changing w/num workers and edge cases where examples
                # from near the end of an epoch and the start of the next epoch get mixed together.
                on_batch = on_example // self.global_batch_size
                if on_batch % num_workers != worker_id:
                    continue
                batch_ix = on_example % self.global_batch_size
                if batch_ix % self.world_size != self.rank:
                    continue
            else:
                # if packing we can't make similar promises about the data order, or compute
                # what batch we are on in the same way, so we just keep it simple
                if on_example % num_global_workers != global_worker_id:
                    continue

            # Actually the load the example
            dataset = self.datasets[dataset_ix]
            epoch = count // len(dataset)
            if self.shuffle:
                shuffled_for, shuffled_order = shuffled_ixs[dataset_ix]
                if epoch != shuffled_for:
                    shuffled_order = self._get_order(dataset_ix, epoch)
                    shuffled_ixs[dataset_ix] = (epoch, shuffled_order)

                example_ix = int(shuffled_order[count % len(dataset)])
            else:
                example_ix = count % len(dataset)
            try:
                out = dataset.get(example_ix, epoch)
            except Exception as e:
                # Skip corrupt/unreadable examples (e.g., bad videos) so one bad sample
                # doesn't kill the whole rank — this is critical with 32+ rank training
                # where any rank's death causes NCCL-timeout cascade across the job.
                # Use logging instead of e.add_note() because add_note requires Python 3.11+
                # and we're on 3.10.
                import logging
                logging.getLogger(__name__).warning(
                    f"Skipping bad example {example_ix}/{epoch} from {dataset.dataset}: "
                    f"{type(e).__name__}: {str(e)[:200]}"
                )
                continue

            # Either yield it or add it to the packer
            if self.packer is None:
                yield out
            else:
                out = self.packer(on_example, out)
                if out is not None:
                    if self.track_packing_state:
                        # Snapshot the worker state to enable dataset checkpointing
                        out["data_worker_state"] = WorkerState(
                            global_worker_id, on_example, self.packer.get_buffered_example_ids())
                    yield out
