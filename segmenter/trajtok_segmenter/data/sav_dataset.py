"""SA-V (Segment Anything Video) dataset wrapper that streams video + instance
masks and outputs the same (video, caption, match_id, masks, graphs, num_token)
tuple as VidGraphTrainDataset.

Data layout expected:
  videos_dir/   : {video_id}.mp4  (fps6 pre-processed)
  instances_dir/: {video_id}_instances.json

Instance JSON format per object:
  { "segmentations": [{"frame_id": "000", "frame_number": 0,
                        "rle": {"size": [H,W], "counts": "..."}}, ...],
    "start_frame": int, "category_id": int, "category_name": str }
"""
import cv2
import glob
import json
import logging
import os
import random

import decord
import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import IterableDataset

from trajtok_segmenter.data.collate import PanopticPosAugmentation, pre_text
from trajtok_segmenter.data.video_utils import get_frame_indices

logger = logging.getLogger(__name__)


def _decode_rle_to_panoptic(instance_data, frame_indices, video_h, video_w, max_objects=256):
    """Decode per-object RLE masks at given frame indices into a panoptic mask.

    Returns:
        masks:  (T, H, W) int32 array, pixel value = object_id (1-indexed), 0 = bg
        graphs: (N, T) int64 array, graphs[m, t] = object_id if present else 0
    """
    T = len(frame_indices)
    obj_ids = sorted(instance_data.keys(), key=int)

    # Build a lookup: for each object, map frame_number -> rle
    obj_frame_rles = {}
    for obj_id_str in obj_ids:
        obj = instance_data[obj_id_str]
        frame_map = {}
        for seg in obj["segmentations"]:
            frame_map[seg["frame_number"]] = seg["rle"]
        obj_frame_rles[int(obj_id_str)] = frame_map

    # For fps6 videos, annotation frame_number = original_frame * 4
    # fps6 frame index i corresponds to annotation frame_number = i * 4
    panoptic = np.zeros((T, video_h, video_w), dtype=np.int32)
    n_objects = min(len(obj_ids), max_objects)
    graphs = np.zeros((n_objects, T), dtype=np.int64)

    for t_idx, frame_idx in enumerate(frame_indices):
        ann_frame = frame_idx * 4  # fps6 frame -> original frame number
        for m, obj_id_str in enumerate(obj_ids[:max_objects]):
            obj_id = int(obj_id_str)
            rle = obj_frame_rles[obj_id].get(ann_frame)
            if rle is None:
                # Try nearest annotated frame (annotations may not cover every frame)
                continue
            binary_mask = mask_utils.decode(rle)  # (H, W) uint8
            obj_label = m + 1  # 1-indexed
            panoptic[t_idx][binary_mask > 0] = obj_label
            graphs[m, t_idx] = obj_label

    return panoptic, graphs


class SAVDataset(IterableDataset):
    """Streams SA-V videos with instance segmentation masks.

    Args:
        videos_dir:    path to fps6 videos, e.g. ".../videos_fps6/"
        instances_dir: path to instance JSONs, e.g. ".../sav_instances/"
        num_frames:    number of frames to sample per video
        sample_type:   frame sampling strategy ("rand" or "middle")
        image_res:     target spatial resolution
        mask_down_factor: downsample factor for masks
        max_objects:   maximum number of objects per video
        num_samples:   approximate dataset length (for scheduler)
    """
    media_type = "sav"

    def __init__(
        self,
        videos_dir,
        instances_dir,
        num_frames=8,
        sample_type="rand",
        image_res=224,
        mask_down_factor=1,
        max_objects=256,
        num_samples=48_000,
    ):
        super().__init__()
        self.videos_dir = videos_dir
        self.instances_dir = instances_dir
        self.num_frames = num_frames
        self.sample_type = sample_type
        self.image_res = int(image_res)
        self.mask_down_factor = mask_down_factor
        self.max_objects = max_objects
        self.num_samples = num_samples
        self.image_mask_augmentation = PanopticPosAugmentation(size=image_res)

        # Build index: list of (video_path, instance_json_path)
        self.samples = self._build_index()
        logger.info(f"SAVDataset: found {len(self.samples)} videos with instance annotations")

    def _build_index(self):
        inst_files = sorted(glob.glob(os.path.join(self.instances_dir, "*_instances.json")))
        samples = []
        for inst_path in inst_files:
            video_id = os.path.basename(inst_path).replace("_instances.json", "")
            video_path = os.path.join(self.videos_dir, f"{video_id}.mp4")
            if os.path.exists(video_path):
                samples.append((video_path, inst_path))
        return samples

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        # Shard across workers
        worker_info = torch.utils.data.get_worker_info()
        samples = self.samples
        if worker_info is not None:
            per_worker = len(samples) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(samples)
            samples = samples[start:end]

        # Shuffle
        samples = list(samples)
        random.shuffle(samples)

        for video_path, inst_path in samples:
            try:
                item = self._process_sample(video_path, inst_path)
                if item is not None:
                    yield item
            except Exception as e:
                logger.warning(f"Skipping {video_path}: {e}")
                continue

    def _process_sample(self, video_path, inst_path):
        # --- load video frames ---
        vr = decord.VideoReader(video_path, num_threads=1)
        vlen = len(vr)
        if vlen < self.num_frames:
            return None

        frame_indices = get_frame_indices(
            self.num_frames, vlen, sample=self.sample_type
        )
        frames = vr.get_batch(frame_indices)  # (T, H, W, C) torch.uint8
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)
        T, C, H, W = frames.shape

        # --- load instance annotations and build panoptic masks ---
        with open(inst_path) as f:
            instance_data = json.load(f)

        panoptic, graphs = _decode_rle_to_panoptic(
            instance_data, frame_indices, H, W, max_objects=self.max_objects
        )
        # panoptic: (T, H, W) int32, graphs: (N, T) int64
        masks = torch.from_numpy(panoptic)  # (T, H, W)

        # --- synchronized augmentation (same crop/flip for all frames + masks) ---
        frames_aug, masks_aug = self.image_mask_augmentation(frames, masks)

        # --- downsample masks ---
        target_h = self.image_res // self.mask_down_factor
        target_w = self.image_res // self.mask_down_factor
        if masks_aug.shape[1] != target_h or masks_aug.shape[2] != target_w:
            masks_np = masks_aug.numpy()
            resized = np.empty((T, target_h, target_w), dtype=masks_np.dtype)
            for t in range(T):
                resized[t] = cv2.resize(
                    masks_np[t], (target_w, target_h), interpolation=cv2.INTER_NEAREST
                )
            masks_aug = torch.from_numpy(resized)

        # --- rebuild graphs from augmented masks (crop may drop objects) ---
        max_id = int(masks_aug.max())
        if max_id == 0:
            graphs_tensor = torch.zeros(1, T, dtype=torch.long)
            num_token = 0
        else:
            graphs_tensor = torch.zeros(max_id, T, dtype=torch.long)
            for m in range(max_id):
                obj_id = m + 1
                for t in range(T):
                    if (masks_aug[t] == obj_id).any():
                        graphs_tensor[m, t] = obj_id
            # Filter out objects that disappeared entirely after crop
            valid = (graphs_tensor > 0).any(dim=1)
            if valid.any():
                graphs_tensor = graphs_tensor[valid]
                num_token = graphs_tensor.shape[0]
            else:
                graphs_tensor = torch.zeros(1, T, dtype=torch.long)
                num_token = 0

        # --- caption (SAV has no text annotations) ---
        caption = "a video"

        match_id = hash(video_path) % (2**31)
        return frames_aug, caption, match_id, masks_aug, graphs_tensor, num_token
