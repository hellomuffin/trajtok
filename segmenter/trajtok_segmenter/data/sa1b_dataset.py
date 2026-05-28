"""SA-1B WebDataset wrapper that streams from tar files and outputs
the same (video, caption, match_id, masks, graphs, num_token) tuple
as ImgGraphTrainDataset, so it plugs into the existing collate and
training loop unchanged.

Supports two SA-1B tar formats:
  - SA-1B-merged (raw): json has {"image": {...}, "annotations": [{segmentation: RLE}, ...]}
  - SA-1B-small (pre-processed): json has {"image": {...}, "panoptic_mask": {mask_png: base64}}
"""
import base64
import cv2
import glob as _glob
import io
import json as _json
import logging
import random

import numpy as np
import torch
import webdataset as wds
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import IterableDataset
from torchvision.transforms import PILToTensor

from trajtok_segmenter.data.collate import PanopticPosAugmentation, pre_text

logger = logging.getLogger(__name__)


def _annotations_to_panoptic(annotations, height, width, max_objects=256, min_area_ratio=0.001):
    """Compose per-object RLE annotations into a panoptic mask (H, W).

    SA-1B-merged format: each annotation has 'segmentation' with COCO RLE.
    Objects are painted in order; later objects overwrite earlier ones at
    overlap pixels (same as the SA-1B preprocessing script).
    Small objects below min_area_ratio are skipped.
    """
    min_area = height * width * min_area_ratio
    panoptic = np.zeros((height, width), dtype=np.int32)
    obj_id = 0
    for ann in annotations:
        if obj_id >= max_objects:
            break
        area = ann.get("area", 0)
        if area < min_area:
            continue
        rle = ann["segmentation"]
        # pycocotools expects size as [h, w]
        if rle["size"] != [height, width]:
            continue  # skip malformed
        binary = mask_utils.decode(rle)  # (H, W) uint8
        obj_id += 1
        panoptic[binary > 0] = obj_id
    return panoptic


def _panoptic_png_to_mask(panoptic_dict, max_objects=256):
    """Decode base64 PNG panoptic mask -> (H, W) int32 with sequential IDs.
    SA-1B-small pre-processed format.
    """
    png_bytes = base64.b64decode(panoptic_dict["mask_png"])
    mask = np.array(Image.open(io.BytesIO(png_bytes)), dtype=np.int32)

    # remap sparse IDs to sequential [0, 1, 2, ..., N]
    unique_ids = np.unique(mask)
    obj_ids = unique_ids[unique_ids > 0]
    if len(obj_ids) > 0 and mask.max() != len(obj_ids):
        new_mask = np.zeros_like(mask)
        for new_id, old_id in enumerate(obj_ids, start=1):
            new_mask[mask == old_id] = new_id
        mask = new_mask

    mask[mask > max_objects] = 0
    return mask


def _make_sample(sample):
    """webdataset map: raw tar sample -> decoded dict with image, json."""
    result = {}
    for key, value in sample.items():
        if key == "__key__":
            result[key] = value
        elif key in ("jpg", "png", "jpeg", "webp"):
            result["image"] = Image.open(io.BytesIO(value)).convert("RGB")
        elif key == "json":
            result["json"] = _json.loads(value)
    return result


class SA1BDataset(IterableDataset):
    """Streams SA-1B webdataset tars and yields the same tuple format as
    ImgGraphTrainDataset so it works with graph_custom_collate_fn.

    Supports both SA-1B-merged (raw RLE annotations) and SA-1B-small
    (pre-composed panoptic PNG) tar formats automatically.

    Args:
        tar_paths:  glob pattern or brace-expansion for tar shards
        image_res:  target image resolution (e.g. 224)
        mask_down_factor: downsample factor for masks
        max_objects: maximum number of object IDs to keep
        shuffle_buffer: webdataset shuffle buffer size
        num_samples: approximate total number of samples (for len / scheduler)
    """
    media_type = "sa1b"

    def __init__(
        self,
        tar_paths,
        image_res=224,
        mask_down_factor=1,
        max_objects=256,
        shuffle_buffer=5000,
        num_samples=11_000_000,
    ):
        super().__init__()
        self.tar_paths = tar_paths
        self.image_res = int(image_res)
        self.mask_down_factor = mask_down_factor
        self.max_objects = max_objects
        self.shuffle_buffer = shuffle_buffer
        self.num_samples = num_samples
        self.image_mask_augmentation = PanopticPosAugmentation(size=image_res)

        self._pil_to_tensor = PILToTensor()
        self._dataset = self._build_pipeline()

    def _build_pipeline(self):
        if '*' in self.tar_paths:
            shards = sorted(_glob.glob(self.tar_paths))
        elif isinstance(self.tar_paths, str):
            shards = wds.shardlists.expand_urls(self.tar_paths)
        else:
            shards = self.tar_paths
        # Filter out empty/corrupted tar files (< 1KB)
        import os
        valid_shards = [s for s in shards if os.path.getsize(s) > 1024]
        if len(valid_shards) < len(shards):
            logger.warning(f"SA1BDataset: filtered out {len(shards) - len(valid_shards)} "
                           f"empty/corrupt shards, {len(valid_shards)} remaining")
        else:
            logger.info(f"SA1BDataset: found {len(valid_shards)} tar shards")
        return (
            wds.WebDataset(valid_shards, shardshuffle=True, nodesplitter=wds.split_by_node,
                           handler=wds.warn_and_continue)
            .shuffle(self.shuffle_buffer)
            .map(_make_sample)
        )

    def __len__(self):
        # webdataset shards by node (split_by_node), so each rank only sees
        # ~num_samples/world_size. DataLoader.__len__ for IterableDatasets is
        # dataset.__len__()//batch_size, *not* further divided by world_size,
        # so we must do the per-rank division here. Otherwise MetaLoader
        # over-counts iterations for this loader and crashes mid-epoch with
        # `RuntimeError: generator raised StopIteration` (PEP 479).
        try:
            import torch.distributed as dist
            world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        except Exception:
            world_size = 1
        return self.num_samples // max(world_size, 1)

    def __iter__(self):
        for sample in self._dataset:
            try:
                item = self._process_sample(sample)
                if item is not None:
                    yield item
            except Exception as e:
                logger.warning(f"Skipping sample {sample.get('__key__', '?')}: {e}")
                continue

    def _process_sample(self, sample):
        pil_image = sample["image"]
        meta = sample["json"]
        img_w, img_h = pil_image.size  # PIL size is (w, h)

        # --- caption ---
        image_meta = meta.get("image", {})
        caption = image_meta.get("caption", "") or image_meta.get("global_caption", "")
        if isinstance(caption, list):
            caption = random.choice(caption)
        caption = pre_text(caption)

        # --- build panoptic mask (auto-detect format) ---
        if "panoptic_mask" in meta:
            # SA-1B-small pre-processed format
            mask_np = _panoptic_png_to_mask(meta["panoptic_mask"], max_objects=self.max_objects)
        elif "annotations" in meta:
            # SA-1B-merged raw format
            mask_np = _annotations_to_panoptic(
                meta["annotations"], img_h, img_w, max_objects=self.max_objects
            )
        else:
            return None

        if mask_np.max() == 0:
            return None  # no valid objects

        # --- image -> tensor (1, C, H, W) uint8 ---
        image_tensor = self._pil_to_tensor(pil_image).unsqueeze(0)

        # --- mask -> (1, H, W), resize to match image if needed ---
        mask_h, mask_w = mask_np.shape
        if mask_h != img_h or mask_w != img_w:
            mask_np = cv2.resize(mask_np, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        # --- synchronized augmentation ---
        image_aug, mask_aug = self.image_mask_augmentation(image_tensor, mask_tensor)

        # --- downsample mask ---
        target_h = self.image_res // self.mask_down_factor
        target_w = self.image_res // self.mask_down_factor
        if mask_aug.shape[1] != target_h or mask_aug.shape[2] != target_w:
            mask_np_aug = mask_aug.numpy()
            resized = np.empty((1, target_h, target_w), dtype=mask_np_aug.dtype)
            resized[0] = cv2.resize(
                mask_np_aug[0], (target_w, target_h), interpolation=cv2.INTER_NEAREST,
            )
            mask_aug = torch.from_numpy(resized)

        # --- build graph from mask (same as ImgGraphTrainDataset) ---
        num_objects = int(mask_aug.max())
        if num_objects == 0:
            graphs = torch.zeros(1, 1, dtype=torch.long)
            num_token = 0
        else:
            graphs = torch.arange(1, num_objects + 1).unsqueeze(1)  # (N, 1)
            num_token = graphs.shape[0]

        match_id = hash(sample.get("__key__", "")) % (2**31)
        return image_aug, caption, match_id, mask_aug, graphs, num_token
