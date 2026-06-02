import dataclasses
from collections import defaultdict
import numpy as np

import pytest
import torch
from olmo.util import flatten_lists

from olmo.data.iterable_dataset_mixture import IterableDatasetMixture, IterableDataMixtureCheckpoint


@dataclasses.dataclass
class MockWorkerInfo:
    id: int
    num_workers: int


@dataclasses.dataclass
class MockItem:
    dataset: str
    idx: int
    epoch: int


@dataclasses.dataclass
class MockDataset:
    name: str
    n: int

    def __len__(self):
        return self.n

    def get(self, item, epoch=0):
        assert 0 <= item < self.n
        return MockItem(self.name, item, epoch)


def test_single_dataset():
    ds = IterableDatasetMixture(
        [MockDataset("", 10)], global_batch_size=2)
    it = iter(ds)
    for epoch in range(10):
        epoch_data = [next(it) for _ in range(10)]
        assert set(x.idx for x in epoch_data) == set(range(10))
        assert all(x.epoch == epoch for x in epoch_data)


def test_mixture():
    ds = IterableDatasetMixture(
        [MockDataset("a", 10), MockDataset("b", 3)],
        mixture_rates=[0.9, 0.1],
        seed=32,
        global_batch_size=2
    )
    it = iter(ds)
    for_a = []
    for_b = []
    for _ in range(200):
        item: MockItem = next(it)
        if item.dataset == "a":
            for_a.append(item)
        else:
            for_b.append(item)
    assert len(for_a) > (len(for_b) * 0.2)
    for items, ds_len in [(for_a, 10), (for_b, 3)]:
        for epoch in range(len(items)//ds_len):
            epoch_items = items[epoch*ds_len:(epoch+1)*ds_len]
            assert all(x.epoch == epoch for x in epoch_items)
            assert set(x.idx for x in epoch_items) == set(range(ds_len))


def _get_device_batch(_worker_its, device_batch_size):
    while True:
        for it in _worker_its:
            batch = []
            for _ in range(device_batch_size):
                batch.append(next(it))
            yield batch


@pytest.mark.parametrize("world_size,num_workers,device_batch_size", [
    (1, 1, 2),
    (3, 1, 4),
    (2, 2, 2),
    (2, 2, 8),
    (3, 6, 4),
])
def test_distributed(world_size, num_workers, device_batch_size):
    start = 0
    global_batch_size = device_batch_size*world_size
    iterators = []
    datasets = [MockDataset("a", 5), MockDataset("b", 11)]
    mixture_rates = [0.8, 0.2]
    device_iterators = []
    bk = torch.utils.data.get_worker_info
    for rank in range(world_size):
        worker_iterators = []
        for worker_id in range(num_workers):
            worker_iterators.append(iter(IterableDatasetMixture(
                datasets, global_batch_size, mixture_rates=mixture_rates,
                rank=rank, world_size=world_size,
                worker_info=MockWorkerInfo(worker_id, num_workers),
                seed=32, start_index=start)))

        device_iterators.append(_get_device_batch(worker_iterators, device_batch_size))
    torch.utils.data.get_worker_info = bk

    grouped_by_dataset = defaultdict(list)
    for i in range(4):
        global_batch = []
        for it in device_iterators:
            global_batch += next(it)
        global_batch.sort(key=lambda x: x.epoch)
        for ex in global_batch:
            grouped_by_dataset[ex.dataset].append(ex)

    for dataset in datasets:
        items = grouped_by_dataset[dataset.name]
        ds_len = dataset.n
        for epoch in range(len(items)//ds_len):
            epoch_items = items[epoch*ds_len:(epoch+1)*ds_len]
            assert all(x.epoch == epoch for x in epoch_items)
            assert set(x.idx for x in epoch_items) == set(range(ds_len))


@pytest.mark.parametrize("ns,start_index,world_size,rank", [
    ([11], 16, 1, 0),
    ([8], 201, 1, 0),
    ([17, 27], 50, 1, 0),
    ([5, 7], 20, 2, 1),
    ([5, 7, 3], 27, 3, 0),
])
def test_dataset_start_at(ns, start_index, world_size, rank):
    datasets = [MockDataset("", n) for n in ns]
    global_batch_size = world_size
    ff_ds = IterableDatasetMixture(
        datasets, start_index=start_index,
        rank=rank, world_size=world_size,
        global_batch_size=global_batch_size,
    )
    ff_it = iter(ff_ds)
    ds = IterableDatasetMixture(
        datasets, start_index=0,
        rank=rank, world_size=world_size,
        global_batch_size=global_batch_size,
    )
    it = iter(ds)
    for _ in range(start_index // global_batch_size):
        for _ in range(global_batch_size//world_size):
            next(it)
    for _ in range(30):
        assert next(it) == next(ff_it)


class MockPacker:
    def __init__(self, buffer_size, verbose=False):
        self.buffer_size = buffer_size
        self.verbose = verbose
        self._buffer = []

    def get_buffered_example_ids(self):
        return [x[0] for x in self._buffer]

    def __call__(self, example_id, example):
        # We use RNG state based on `example_id` to make the packer be deterministic
        if np.random.RandomState(example_id * 37).random() < 0.1:
            if self.verbose:
                print(f"Yield {example_id} from {[x[0] for x in self._buffer]}")
            return dict(items=[example])
        if self.verbose:
            print(f"Add {example_id}")
        self._buffer.append((example_id, example))

        if len(self._buffer) == self.buffer_size:
            rng = np.random.RandomState(sum(x[0]*17 for x in self._buffer))
            n_to_pack = rng.randint(1, 4)
            to_pack = rng.choice(len(self._buffer), n_to_pack, replace=False)
            to_pack = sorted(to_pack, reverse=True)
            if self.verbose:
                print(f"Yield {[self._buffer[i][0] for i in to_pack]} from {[x[0] for x in self._buffer]}")
            return dict(items=[self._buffer.pop(i)[1] for i in to_pack])


def test_restore_packing_simple():
    datasets = [MockDataset("a", 17), MockDataset("b", 12)]
    ds = IterableDatasetMixture(
        datasets, mixture_rates=[0.8, 0.2], global_batch_size=2, packer=MockPacker(4),
        track_packing_state=True)
    checkpoint_at = 4
    after_checkpoint = 4
    it = iter(ds)
    for _ in range(checkpoint_at):
        batch = next(it)
    checkpoint = IterableDataMixtureCheckpoint([batch["data_worker_state"]], 1, 1)

    ds = IterableDatasetMixture(
        datasets, mixture_rates=[0.8, 0.2], global_batch_size=2, packer=MockPacker(4),
        track_packing_state=True)
    ds.resume_from = checkpoint
    ff_it = iter(ds)
    for i in range(after_checkpoint):
        actual = next(ff_it)["items"]
        expected = next(it)["items"]
        assert actual == expected


@pytest.mark.parametrize("world_size,num_workers,checkpoint_at", [
    (1, 1, 19),
    (1, 2, 4),
    (3, 1, 19),
    (4, 8, 19),
    (4, 7, 16),
    (5, 3, 29),
])
def test_restore_packing_global(world_size, num_workers, checkpoint_at):
    device_batch_size = 2
    global_batch_size = device_batch_size*world_size
    iterators = []
    datasets = [MockDataset("a", 23), MockDataset("b", 17)]
    mixture_rates = [0.8, 0.2]
    device_iterators = []

    def _build_iterator(_resume_from=None):
        rank_iterators = []
        all_workers = []
        for rank in range(world_size):
            worker_iterators = []
            for worker_id in range(num_workers):
                ds = IterableDatasetMixture(
                    datasets, global_batch_size, mixture_rates=mixture_rates,
                    rank=rank, world_size=world_size,
                    packer=MockPacker(5), track_packing_state=True,
                    worker_info=MockWorkerInfo(worker_id, num_workers),
                    seed=32, resume_from=_resume_from
                )
                all_workers.append(ds)
                worker_iterators.append(iter(ds))
            rank_iterators.append(_get_device_batch(worker_iterators, device_batch_size))

        def get_batch():
            while True:
                yield flatten_lists(next(r) for r in rank_iterators)
        return get_batch(), all_workers

    it, workers = _build_iterator(None)
    _data_worker_states = {}
    for ix in range(checkpoint_at):
        batch = next(it)
        for ex in batch:
            state = ex["data_worker_state"]
            if state.worker_global_id not in _data_worker_states:
                _data_worker_states[state.worker_global_id] = state
            else:
                cur_version = _data_worker_states[state.worker_global_id].version
                if cur_version < state.version:
                    _data_worker_states[state.worker_global_id] = state

    checkpoint = IterableDataMixtureCheckpoint(
        list(_data_worker_states.values()), world_size, num_workers,
        next_worker_id=checkpoint_at % num_workers)
    ff_it, ff_workers = _build_iterator(checkpoint)

    for _ in range(10):
        actual = [x["items"] for x in next(ff_it)]
        expected = [x["items"] for x in next(it)]
        assert actual == expected


def test_stratify():
    datasets = [MockDataset("a", 17), MockDataset("b", 12)]
    ds = IterableDatasetMixture(
        datasets, mixture_rates=[0.8, 0.2], stratify=True, global_batch_size=2)
    it = iter(ds)
    grouped_by_dataset = defaultdict(list)
    for _ in range(10):
        ex = next(it)
        grouped_by_dataset[ex.dataset].append(ex)
    assert len(grouped_by_dataset["a"]) == 8
    assert len(grouped_by_dataset["b"]) == 2

    for _ in range(90):
        ex = next(it)
        grouped_by_dataset[ex.dataset].append(ex)
    assert len(grouped_by_dataset["a"]) == 80
    assert len(grouped_by_dataset["b"]) == 20
    for dataset in datasets:
        items = grouped_by_dataset[dataset.name]
        ds_len = dataset.n
        for epoch in range(len(items)//ds_len):
            epoch_items = items[epoch*ds_len:(epoch+1)*ds_len]
            assert all(x.epoch == epoch for x in epoch_items)
            assert set(x.idx for x in epoch_items) == set(range(ds_len))
