import torch
import torch.distributed as dist
from trajtok_segmenter.train.distributed import get_rank, is_dist_avail_and_initialized, is_main_process
import random
import logging

logger = logging.getLogger(__name__)


class MetaLoader(object):
    """ wraps multiple data loader """
    def __init__(self, name2loader):
        """Iterates over multiple dataloaders, it ensures all processes
        work on data from the same dataloader. This loader will end when
        the shorter dataloader raises StopIteration exception.

        loaders: Dict, {name: dataloader}
        """
        self.name2loader = name2loader
        self.name2iter = {name: iter(l) for name, l in name2loader.items()}
        name2index = {name: idx for idx, (name, l) in enumerate(name2loader.items())}
        index2name = {v: k for k, v in name2index.items()}

        iter_order = []
        for n, l in name2loader.items():
            iter_order.extend([name2index[n]]*len(l))

        random.shuffle(iter_order)
        iter_order = torch.Tensor(iter_order).to(torch.device("cuda")).to(torch.uint8)

        # sync
        if is_dist_avail_and_initialized():
            # make sure all processes have the same order so that
            # each step they will have data from the same loader
            dist.broadcast(iter_order, src=0)
        self.iter_order = [index2name[int(e.item())] for e in iter_order.cpu()]

        logger.info(str(self))

    def __str__(self):
        output = [f"MetaLoader has {len(self.name2loader)} dataloaders, {len(self)} batches in total"]
        for idx, (name, loader) in enumerate(self.name2loader.items()):
            output.append(
                f"dataloader index={idx} name={name}, batch-size={loader.batch_size} length(#batches)={len(loader)} "
            )
        return "\n".join(output)

    def __len__(self):
        return len(self.iter_order)

    def __iter__(self):
        """Iterate batches in `iter_order`. If a sub-loader raises StopIteration
        before iter_order is exhausted (common for IterableDatasets whose
        declared length is approximate), restart that loader's iterator and
        continue. This prevents the PEP-479 `RuntimeError: generator raised
        StopIteration` crash that would otherwise kill training mid-epoch.
        """
        for name in self.iter_order:
            _iter = self.name2iter[name]
            try:
                batch = next(_iter)
            except StopIteration:
                logger.warning(
                    f"MetaLoader: sub-loader '{name}' exhausted early; restarting iterator."
                )
                _iter = iter(self.name2loader[name])
                self.name2iter[name] = _iter
                try:
                    batch = next(_iter)
                except StopIteration:
                    logger.error(
                        f"MetaLoader: sub-loader '{name}' yielded zero batches on restart; ending epoch."
                    )
                    return
            yield name, batch
