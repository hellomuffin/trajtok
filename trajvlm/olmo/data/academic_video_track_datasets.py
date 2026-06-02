"""
Academic video object tracking datasets.

Directory layout under VIDEO_TRACK_DATA_HOME:

    VIDEO_TRACK_DATA_HOME/
    ├── MeViS/                              # raw frames + encoded videos
    │   ├── train/
    │   │   ├── JPEGImages.tar              # raw download from HF
    │   │   ├── JPEGImages/                 # extracted frames
    │   │   │   ├── 05ce52751f5f/
    │   │   │   │   ├── 00000.jpg
    │   │   │   │   └── ...
    │   │   │   └── ...
    │   │   └── videos/                     # encoded .mp4s (created by download())
    │   │       ├── 05ce52751f5f.mp4
    │   │       └── ...
    │   ├── valid_u/                        # same structure as train/
    │   │   └── ...
    │   └── valid/                          # same structure as train/
    │       └── ...
    │
    └── tracking/                           # HF annotation cache (auto-created)
        ├── mevis/                          # MOT dataset
        │   ├── track/
        │   │   ├── train/
        │   │   ├── valid/
        │   │   └── valid_u/
        │   ├── ground/
        │   │   └── ...
        │   └── single_point_track/
        │       └── ...
        └── lasot/                          # SOT dataset (same layout)
            └── single_point_track/
                └── train/

HF repo structure:
  Multi-object tracking (task_type as HF config, standard splits):
    allenai/molmo2-mevis          -> configs: track, ground, single_point_track
    allenai/molmo2-burst          -> configs: track, ground, single_point_track
    allenai/molmo2-ref-yt-vos     -> configs: track
    allenai/molmo2-lv-vis         -> configs: track, ground, single_point_track
    allenai/molmo2-vicas          -> configs: track, ground, single_point_track
    allenai/molmo2-revos          -> configs: track, ground, single_point_track
    allenai/molmo2-ref-davis17    -> configs: track
    allenai/molmo2-yt-vis         -> configs: track
    allenai/molmo2-moca           -> configs: track, ground

  Single object tracking (dataset_name as HF config, task: single_point_track):
    allenai/molmo2-single-object-track -> configs: all (default), lasot, webuav, trackingnet, ...
"""

import os
import json
import logging
from os.path import join, exists
from glob import glob
try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
import multiprocessing as mp
from typing import List, Dict, Optional
from typing_extensions import TypedDict
from tqdm import tqdm
import subprocess
import re
from huggingface_hub import snapshot_download, hf_hub_download
import datasets
import zipfile

from olmo.data.dataset import Dataset, VIDEO_DATA_HOME
from olmo.data.utils import maybe_download_and_unzip, maybe_download_file

log = logging.getLogger(__name__)

MAX_VIDEO_FPS = 10

VIDEO_TRACK_DATA_HOME = join(VIDEO_DATA_HOME, "video_track")

TRACKING_TASKS = ["track", "ground", "single_point_track"]

class Point(TypedDict):
    point: List[float] # [x, y] coordinates; None if empty point
    occluded: Optional[bool] = False

class PointTrack(TypedDict):
    frame: int # frame index
    time: float # time in seconds
    points: Dict[int, Point] # object_id -> {'point': [x, y], 'occluded': bool}

def _load_hf_dataset(hf_source, split, local_name=None, config=None, overwrite_cache=False):
    local_dir = join(VIDEO_TRACK_DATA_HOME, local_name) if VIDEO_TRACK_DATA_HOME and local_name else None

    if local_dir and exists(local_dir) and not overwrite_cache:
        log.info(f"Loading {hf_source} config={config} split={split} from {local_dir}")
        return datasets.load_from_disk(local_dir)

    log.info(f"Downloading {hf_source} config={config} split={split}")
    ds = datasets.load_dataset(hf_source, config, split=split) if config else datasets.load_dataset(hf_source, split=split)

    if local_dir:
        log.info(f"Caching to {local_dir}")
        ds.save_to_disk(local_dir)

    return ds

def get_image_files(image_folder):
    exts = ['jpg', 'jpeg', 'png', 'gif']
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(image_folder, f'*.{ext}')))
        files.extend(glob(os.path.join(image_folder, f'*.{ext.upper()}')))
    return sorted(files)

def encode_frames_to_video(frames_dir, output_path, fps, native_fps:int=None,
                           start_frame=None, end_frame=None):
    """Convert a directory of image frames to an H.264 .mp4 using ffmpeg.

    Frames are first subsampled for FPS conversion (if native_fps differs from
    fps), then sliced to [start_frame, end_frame]. Frame indices refer to
    positions *after* subsampling.

    Args:
        frames_dir: Directory containing image frames (jpg/png).
        output_path: Path for the output .mp4 file.
        fps: Target framerate for the output video.
        native_fps: Original framerate of the frame sequence. When provided
            and different from fps, every round(native_fps / fps)-th frame
            is kept (e.g. native_fps=24, fps=6 keeps every 4th frame).
            None means all frames are used as-is.
        start_frame: First frame index to include (inclusive, 0-based,
            after subsampling). None means start from the beginning.
        end_frame: Last frame index to include (inclusive, 0-based,
            after subsampling). None means include through the last frame.
    """
    image_files = get_image_files(frames_dir)
    if not image_files:
        log.warning(f"No frames in {frames_dir}")
        return None

    if native_fps is not None and native_fps != fps:
        subsample_factor = max(1, round(native_fps / fps))
        image_files = image_files[::subsample_factor]
    
    # pro
    end_frame = end_frame + 1 if end_frame is not None else None
    image_files = image_files[start_frame:end_frame]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write a concat demuxer file list for the selected frames
    filelist_path = output_path + ".filelist.txt"
    with open(filelist_path, 'w') as f:
        for img in image_files:
            f.write(f"file '{img}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-r", str(fps),
        "-i", filelist_path,
        "-c:v", "libopenh264",
        "-b:v", "4M",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_path
    except subprocess.CalledProcessError as e:
        log.error(f"ffmpeg failed for {frames_dir}: {e.stderr}")
        return None
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found")
    finally:
        if exists(filelist_path):
            os.remove(filelist_path)

def _encode_frames_to_video_worker(kwargs):
    """Multiprocessing worker for encode_frames_to_video."""
    output_path = kwargs.get('output_path', '')
    try:
        result = encode_frames_to_video(**kwargs)
        return (output_path, result is not None, None)
    except Exception as e:
        return (output_path, False, str(e))

def extract_frames(video_path, out_folder):
    """Extract all frames from a video to JPEG files using ffmpeg."""
    os.makedirs(out_folder, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        os.path.join(out_folder, "frame_%05d.jpg"),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def _extract_frames_worker(kwargs):
    """Unpacks tuple for multiprocessing. Returns (video_path, success, error)."""
    video_path = kwargs.get('video_path', '')
    try:
        extract_frames(**kwargs)
        return (video_path, True, None)
    except Exception as e:
        return (video_path, False, str(e))

def get_candidate_sampling_fps(video_fps, sampling_fps, max_fps=MAX_VIDEO_FPS):
    video_fps, sampling_fps = int(video_fps), int(sampling_fps)
    if video_fps <= 0 or sampling_fps <= 0:
        raise ValueError(f"Must be positive (got {video_fps}, {sampling_fps})")
    if video_fps % sampling_fps != 0:
        raise ValueError(f"sampling_fps={sampling_fps} must divide video_fps={video_fps}")
    candidates = []
    for c in range(sampling_fps, video_fps + 1, sampling_fps):
        if c > max_fps:
            break
        if video_fps % c == 0:
            candidates.append(c)
    return candidates


class TrackingDataset(Dataset):
    HF_SOURCE = None
    DATASET_NAME = None
    VIDEO_HOME = VIDEO_TRACK_DATA_HOME   # Root dir for this dataset's raw data
    LOCAL_DIR = join(VIDEO_TRACK_DATA_HOME, "tracking")  # Root dir for caching HF datasets
    VIDEO_FPS = 6       # Default fall back; actual fps comes from annotation per-example but academic tracking data uses 6 fps
    TASKS = []
    SPLIT_MAP = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }

    @classmethod
    def _get_hf_config(cls, task):
        """Return the HF config name for loading a given task.

        MOT repos use task as config (e.g. "track", "ground").
        SOT repo overrides to use DATASET_NAME as config.
        """
        return task

    @classmethod
    def _get_local_dir(cls, task, data_split):
        """Local cache path: tracking/{dataset_name}/{task}/{split}/"""
        return join(cls.LOCAL_DIR, cls.DATASET_NAME, task, data_split)

    @classmethod
    def _get_video_dir(cls, data_split):
        """Default: VIDEO_HOME/{data_split}/videos/"""
        return join(cls.VIDEO_HOME, data_split, "videos")

    @classmethod
    def _load_all_dataset_and_fps(cls, overwrite_cache=True):
        """Load/cache annotations across all tasks/splits, return {(data_split, video_name): fps}.

        video_name is the video identifier without file extension.
        overwrite_cache: re-download from HF even if local cache exists (pass True from download()).
        """
        video_fps = {}
        seen = set()
        for task in cls.TASKS:
            for data_split in set(cls.SPLIT_MAP.values()):
                if (task, data_split) in seen:
                    continue
                seen.add((task, data_split))
                try:
                    local_dir = cls._get_local_dir(task, data_split)
                    config = cls._get_hf_config(task)
                    ds = _load_hf_dataset(cls.HF_SOURCE, data_split, local_name=local_dir,
                                          config=config, overwrite_cache=overwrite_cache)
                    ds = ds.select_columns(["video", "fps"])
                except Exception as e:
                    log.warning(f"Could not load {cls.HF_SOURCE}/{config}/{data_split}: {e}")
                    continue
                for ex in ds:
                    video_name = os.path.splitext(ex['video'])[0]
                    key = (data_split, video_name)
                    fps = ex['fps']
                    if key in video_fps:
                        assert video_fps[key] == fps, f"Conflicting FPS for {key}: {video_fps[key]} vs {fps}"
                    video_fps[key] = fps
        return video_fps
    
    @classmethod
    def _create_videos(cls, work_items, n_procs=1):
        """Convert frame dirs to videos, optionally in parallel."""
        if not work_items:
            log.info(f"[{cls.DATASET_NAME}] No videos to create (frames missing or already processed).")
            return

        n = min(n_procs, len(work_items))
        log.info(f"[{cls.DATASET_NAME}] Creating {len(work_items)} videos with {n} workers...")
        log.info(f"[{cls.DATASET_NAME}] Examples:")
        for item in work_items[:5]:
            log.info(f"  Frames dir: {item['frames_dir']}, Output: {item['output_path']}, FPS: {item['fps']}")

        failed = []
        with mp.Pool(n) as pool:
            for output_path, success, error in tqdm(
                pool.imap_unordered(_encode_frames_to_video_worker, work_items),
                total=len(work_items), desc=f"[{cls.DATASET_NAME}] Encoding videos",
            ):
                if not success:
                    failed.append((output_path, error))
        if failed:
            log.warning(f"[{cls.DATASET_NAME}] {len(failed)}/{len(work_items)} videos failed, e.g.: {failed[:5]}")
        else:
            log.info(f"[{cls.DATASET_NAME}] All {len(work_items)} videos created successfully.")

    @classmethod
    def _check_videos(cls, video_fps):
        """After download, check that all videos referenced in annotations exist."""
        missing = []
        for (data_split, video_name), fps in video_fps.items():
            video_dir = cls._get_video_dir(data_split)
            video_path = join(video_dir, f"{video_name}.mp4")
            if not exists(video_path):
                missing.append(video_path)
        if missing:
            log.warning(f"[{cls.DATASET_NAME}] {len(missing)} missing videos, e.g.: {missing[:5]}")
        else:
            log.info(f"[{cls.DATASET_NAME}] ✅ All {len(video_fps)} unique instances for all splits verified")
        return missing

    # ── Download pipeline ──────────────────────────────────────────────────
    #
    # Generic download flow (subclasses override _get_frames_dir and
    # _prepare_annotation_dir; everything else is inherited):
    #
    #   1. _load_all_dataset_and_fps()        -> {(split, video): fps}
    #   2. _check_videos()                    -> return early if none missing
    #   3. _prepare_annotation_dir()          -> download + extract raw frames for all splits/videos
    #   4. _build_video_work_items()          -> [{frames_dir, output_path, fps}]
    #   5. _create_videos() + _check_videos() -> encode & verify
    #   6. _precompute_gt_masks()             -> RLE JSON for eval (dataset decides which splits)
    #

    @classmethod
    def _get_frames_dir(cls, data_split, video_name):
        """Return path to the frame directory for a given video.

        Default: VIDEO_HOME/{data_split}/JPEGImages/{video_name}
        Override for datasets with different layouts.
        """
        return join(cls.VIDEO_HOME, data_split, "JPEGImages", video_name)

    @classmethod
    def _prepare_annotation_dir(cls, n_procs=1):
        """Download and extract raw data so that frames are available.

        After this returns, _get_frames_dir(data_split, video_name) should
        point to a valid directory of images for every video in every split.

        Override per dataset. Default is a no-op (assumes frames already exist).
        """
        pass

    @classmethod
    def _precompute_gt_masks(cls):
        """Precompute GT masks as RLE JSON for eval. Override if dataset has GT masks."""
        pass

    @classmethod
    def _build_video_work_items(cls, video_fps_map):
        """Build work items for _create_videos from the fps map.

        Returns list of dicts: [{frames_dir, output_path, fps, ...}].
        Override for datasets needing extra kwargs (e.g. native_fps).
        """
        work_items = []
        for (data_split, video_name), fps in video_fps_map.items():
            frames_dir = cls._get_frames_dir(data_split, video_name)
            output_path = join(cls._get_video_dir(data_split), f"{video_name}.mp4")
            if not exists(output_path) and exists(frames_dir):
                work_items.append({'frames_dir': frames_dir, 'output_path': output_path, 'fps': fps})
        return work_items

    @classmethod
    def download(cls, n_procs=1):
        """Generic download pipeline. Override _prepare_annotation_dir and
        _get_frames_dir to customize; override download() entirely for
        non-standard flows (e.g. pre-existing videos, chunked downloads)."""
        # Step 1: What videos do we need?
        video_fps_map = cls._load_all_dataset_and_fps()

        # Step 2: Which are missing?
        missing = cls._check_videos(video_fps_map)
        if missing:
            log.info(f"[{cls.DATASET_NAME}] {len(missing)}/{len(video_fps_map)} videos missing.")

            # Step 3: Download + extract raw frames
            cls._prepare_annotation_dir(n_procs)

            # Step 4-5: Build work items, encode frames -> mp4, verify
            work_items = cls._build_video_work_items(video_fps_map)
            cls._create_videos(work_items, n_procs)
            cls._check_videos(video_fps_map)

        # Step 6: Always precompute GT masks (checks internally if already done)
        cls._precompute_gt_masks()

    def __init__(self, split, task, sampling_fps=None, use_fps_sampling=True):
        assert task in TRACKING_TASKS, f"Invalid task: {task}"
        assert task in self.TASKS, f"Task '{task}' not supported for {self.DATASET_NAME}. Available: {self.TASKS}"
        assert split in self.SPLIT_MAP, f"Invalid split: {split}. Available: {list(self.SPLIT_MAP.keys())}"
        self.split = split
        self.task = task
        self.sampling_fps = sampling_fps
        self.use_fps_sampling = use_fps_sampling
        self.is_eval = split not in ["train"]
        self.data_split = self.SPLIT_MAP[split]
        self.video_dir = self._get_video_dir(self.data_split)
        self.data_lookup = {}
        self.data = self.load()

    def load(self):
        local_dir = self._get_local_dir(self.task, self.data_split)
        config = self._get_hf_config(self.task)
        data = _load_hf_dataset(self.HF_SOURCE, self.data_split, local_name=local_dir, config=config)

        self.data_lookup = {ex_id: i for i, ex_id in enumerate(data["id"])}

        if self.use_fps_sampling:
            n_pre = len(data)
            data = data.filter(self._try_get_fps, input_columns="fps")
            if n_pre != len(data):
                log.warning(f"Filtered {n_pre - len(data)}/{n_pre} examples due to FPS mismatch")

        # Filter rows where fps is not divisible by the row's sampling_fps
        if "sampling_fps" in data.column_names:
            n_pre = len(data)
            data = data.filter(lambda fps, sfps: fps % sfps == 0, input_columns=["fps", "sampling_fps"])
            if n_pre != len(data):
                log.warning(f"Filtered {n_pre - len(data)}/{n_pre} examples where fps is not divisible by sampling_fps")

        if self.sampling_fps is not None:
            n_pre = len(data)
            data = data.filter(lambda sampling_fps: sampling_fps == self.sampling_fps, input_columns="sampling_fps")
            log.info(f"Filtered to sampling_fps={self.sampling_fps}: {len(data)}/{n_pre}")

        return data

    def __len__(self):
        return len(self.data)

    def _try_get_fps(self, video_fps):
        try:
            self._get_candidate_fps(video_fps or self.VIDEO_FPS)
            return True
        except ValueError:
            return False

    def _get_candidate_fps(self, video_fps):
        return get_candidate_sampling_fps(video_fps, self.sampling_fps or 1)
    
    def _get_style(self):
        if self.task == "track":
            return "video_point_track_per_frame"
        elif self.task == "ground":
            return "video_point_ground_start_end"
        elif self.task == "single_point_track":
            return "video_single_point_track_per_frame"
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def _create_message_list(self, ex):
        """ Create message list with points organized by frame. """
        style = self._get_style()

        object_id_to_idx = {obj_id: idx for idx, obj_id in enumerate(ex['mask_id'])}

        message_list = [{
            "style": style,
            "label": ex['expression'],
            "sampling_fps": ex['sampling_fps'],
            "width": ex['width'],
            "height": ex['height'],
        }]

        if 'frame_trajectories' in ex:
            point_frames = []
            for frame_data in ex['frame_trajectories']:
                points_out = {}
                for p in frame_data['points']:
                    obj_key = str(p['id'])
                    if obj_key in object_id_to_idx:
                        points_out[object_id_to_idx[obj_key]] = {
                            'point': p['point'],
                            'occluded': p['occluded'],
                        }
                point_frames.append({
                    'frame': frame_data['frame'],
                    'time': frame_data['time'],
                    'points': points_out,
                })
            point_frames.sort(key=lambda x: x['frame'])

            message_list[0]['points'] = point_frames or None
        
        return message_list


    def get(self, idx, rng):
        ex = self.data[idx]
        video_fps = ex.get("fps", self.VIDEO_FPS)

        video_rel_path = ex['video'] + '.mp4'
        video_path = join(self.video_dir, video_rel_path)
        message_list = self._create_message_list(ex)

        metadata = {
            'example_id': ex['id'],
            'task': self.task,
            'expression': ex['expression'],
            'w': ex['width'],
            'h': ex['height'],
            'video_fps': video_fps,
            'video': ex['video'],
        }

        if self.use_fps_sampling:
            metadata['sampler_overrides'] = {
                'frame_sample_mode': 'fps',
                'candidate_sampling_fps': self._get_candidate_fps(video_fps),
                'min_fps': ex['sampling_fps'],
            }

        return {
            'video': video_path,
            'message_list': message_list,
            'sampling_fps': ex['sampling_fps'],
            'metadata': metadata,
        }

    def get_by_example_id(self, example_id):
        idx = self.data_lookup.get(example_id)
        if idx is not None:
            return self.get(idx, None)
        log.warning(f"Example ID '{example_id}' not found.")
        return None



# ── Dataset subclasses ──────────────────────────────────────────────────────


class Mevis(TrackingDataset):
    """MeViS: https://github.com/henghuiding/MeViS"""
    HF_SOURCE = "allenai/molmo2-mevis"
    DATASET_NAME = "mevis"
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, "MeViS")
    VIDEO_FPS = 6
    TASKS = ["track", "ground", "single_point_track"]
    SPLIT_MAP = {
        "train": "train",
        "validation": "valid_u",
        "test": "valid_u",
    }
    MEVIS_DIR = join(VIDEO_TRACK_DATA_HOME, "MeViS", "MeViS_release")

    @classmethod
    def _get_frames_dir(cls, data_split, video_name):
        return join(cls.MEVIS_DIR, data_split, "JPEGImages", video_name)

    @classmethod
    def _prepare_annotation_dir(cls, n_procs=1):
        """Download and extract MeViS frames for all splits including "valid" (for MevisChallenge).

        Fallback chain per split:
        1. frames_root exists -> skip
        2. JPEGImages.tar exists -> extract
        3. Download FudanCVL/MeViS from HF -> unzip MeViS_release.zip -> extract tar
        """

        # All splits including "valid" which MevisChallenge uses
        ALL_SPLITS = ["train", "valid_u", "valid"]
        for data_split in ALL_SPLITS:
            frames_root = join(cls.MEVIS_DIR, data_split, "JPEGImages")
            if exists(frames_root):
                log.info(f"[{cls.DATASET_NAME}] Frames for '{data_split}' found at {frames_root}")
                continue

            tar_path = join(cls.MEVIS_DIR, data_split, "JPEGImages.tar")
            if not exists(tar_path):
                zip_path = join(cls.VIDEO_HOME, "MeViS_release.zip")
                if not exists(zip_path):
                    log.info(f"[{cls.DATASET_NAME}] Downloading from FudanCVL/MeViS...")
                    snapshot_download(
                        repo_id="FudanCVL/MeViS",
                        repo_type="dataset",
                        local_dir=cls.VIDEO_HOME,
                        local_dir_use_symlinks=False,
                        max_workers=n_procs,
                    )

                if exists(zip_path) and not exists(cls.MEVIS_DIR):
                    log.info(f"[{cls.DATASET_NAME}] Extracting {zip_path}...")
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        for member in tqdm(zf.infolist(), desc="MeViS_release.zip", leave=False):
                            zf.extract(member, cls.VIDEO_HOME)

            assert exists(tar_path), f"[{cls.DATASET_NAME}] Expected {tar_path} after download+unzip, not found."

            log.info(f"[{cls.DATASET_NAME}] Extracting {tar_path}...")
            subprocess.run(["tar", "-xf", tar_path, "-C", join(cls.MEVIS_DIR, data_split),
                            "--checkpoint=1000", "--checkpoint-action=echo=#%u files extracted"], check=True)

            assert exists(frames_root), f"[{cls.DATASET_NAME}] Expected {frames_root} after tar extraction, not found."

    @classmethod
    def _precompute_gt_masks(cls):
        cls._precompute_gt_masks_for_split("valid_u")

    @classmethod
    def _precompute_gt_masks_for_split(cls, data_split):
        """Encode GT mask PNGs to COCO RLE and save as individual JSON files per query.

        Saves to: VIDEO_HOME/{data_split}/MasksRLE/{video_id}/{query_id}.json
        """
        VALID_U_N_QUERIES = 793

        mevis_dir = join(cls.VIDEO_HOME, "MeViS_release")

        # Load HF dataset to get (video_id, qid, ano_id) for each example
        local_dir = cls._get_local_dir("track", data_split)
        config = cls._get_hf_config("track")
        ds = _load_hf_dataset(cls.HF_SOURCE, data_split, local_name=local_dir, config=config)
        hf_queries = [(ex['video'], ex['qid'], ex['anno_id']) for ex in ds]
        assert len(hf_queries) == VALID_U_N_QUERIES, \
            f"Expected {VALID_U_N_QUERIES} queries but found {len(hf_queries)} in HF dataset"

        # Quick check: if all expected mask files exist, skip
        output_dir = join(cls.VIDEO_HOME, data_split, "MasksRLE")
        if exists(output_dir):
            missing_masks = [
                (vid, qid) for (vid, qid, _) in hf_queries
                if not exists(join(output_dir, vid, f"{qid}.json"))
            ]
            if not missing_masks:
                log.info(f"[{cls.DATASET_NAME}] All {len(hf_queries)} GT masks present, skipping.")
                return
            log.info(f"[{cls.DATASET_NAME}] {len(missing_masks)}/{len(hf_queries)} GT masks missing, will precompute.")

        # Save RLE masks for each query
        # total_query = sum(len(v['expressions']) for v in meta_expressions['videos'].values())
        mask_annotation_path = join(mevis_dir, data_split, "mask_dict.json")
        mask_dict = json.load(open(mask_annotation_path, 'r'))
        n_encoded = 0
        n_skipped = 0
        for (video_id, query_id, anno_ids) in tqdm(hf_queries, desc=f"Encoding GT masks ({data_split})"):
            query_output = join(output_dir, video_id, f"{query_id}.json")
            if exists(query_output):
                n_skipped += 1
                continue

            mask_annot = {}
            for mask_idx, anno_id in enumerate(anno_ids):
                anno_id = str(anno_id) # mask_dict keys are strings
                mask_annot[mask_idx] = mask_dict[anno_id]

            os.makedirs(join(output_dir, video_id), exist_ok=True)
            with open(query_output, 'w') as f:
                json.dump(mask_annot, f)
            n_encoded += 1

        log.info(f"Precomputed GT masks: {n_encoded} new, {n_skipped} already exist, out of {len(hf_queries)} queries.")

    def get(self, idx, rng):
        item = super().get(idx, rng)
        if self.is_eval:
            ex = self.data[idx]
            masks_dir = join(self.VIDEO_HOME, self.data_split, "MasksRLE")
            masks_path = join(masks_dir, ex['video'], f"{ex['qid']}.json")
            with open(masks_path, 'r') as f:
                masks = json.load(f)
            if masks is not None:
                item['metadata']['masks'] = masks
                item['metadata']['mask_id'] = ex['mask_id']
                if "points" in item['message_list'][0]:
                    item['metadata']['points'] = item['message_list'][0]['points']
        return item

class MevisChallenge(Mevis):
    """MeViS Challenge set: MeViS valid split (no GT annotations).

    Inherits _prepare_annotation_dir and _get_frames_dir from Mevis.
    Uses base TrackingDataset.download() pipeline. No GT masks.
    """
    HF_SOURCE = "allenai/molmo2-mevis-valid"
    DATASET_NAME = "mevis-valid"
    TASKS = ["track"]
    SPLIT_MAP = {
        "test": "valid"
    }

    @classmethod
    def _prepare_annotation_dir(cls, n_procs=1):
        """Download and extract MeViS frames for all splits including "valid" (for MevisChallenge).

        Fallback chain per split:
        1. frames_root exists -> skip
        2. JPEGImages.tar exists -> extract
        3. Download FudanCVL/MeViS from HF -> unzip MeViS_release.zip -> extract tar
        """

        # All splits including "valid" which MevisChallenge uses
        ALL_SPLITS = ["valid"]
        for data_split in ALL_SPLITS:
            frames_root = join(cls.MEVIS_DIR, data_split, "JPEGImages")
            if exists(frames_root):
                log.info(f"[{cls.DATASET_NAME}] Frames for '{data_split}' found at {frames_root}")
                continue

            tar_path = join(cls.MEVIS_DIR, data_split, "JPEGImages.tar")
            if not exists(tar_path):
                zip_path = join(cls.VIDEO_HOME, "MeViS_release.zip")
                if not exists(zip_path):
                    log.info(f"[{cls.DATASET_NAME}] Downloading from FudanCVL/MeViS...")
                    snapshot_download(
                        repo_id="FudanCVL/MeViS",
                        repo_type="dataset",
                        local_dir=cls.VIDEO_HOME,
                        local_dir_use_symlinks=False,
                        max_workers=n_procs,
                    )

                if exists(zip_path) and not exists(cls.MEVIS_DIR):
                    log.info(f"[{cls.DATASET_NAME}] Extracting {zip_path}...")
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        for member in tqdm(zf.infolist(), desc="MeViS_release.zip", leave=False):
                            zf.extract(member, cls.VIDEO_HOME)

            assert exists(tar_path), f"[{cls.DATASET_NAME}] Expected {tar_path} after download+unzip, not found."

            log.info(f"[{cls.DATASET_NAME}] Extracting {tar_path}...")
            subprocess.run(["tar", "-xf", tar_path, "-C", join(cls.MEVIS_DIR, data_split),
                            "--checkpoint=1000", "--checkpoint-action=echo=#%u files extracted"], check=True)

            assert exists(frames_root), f"[{cls.DATASET_NAME}] Expected {frames_root} after tar extraction, not found."

    @classmethod
    def _precompute_gt_masks(cls):
        """No GT masks for MeViS Challenge set."""
        pass

    def get(self, idx, rng):
        # Skip Mevis.get() mask loading — challenge set has no GT annotations
        return TrackingDataset.get(self, idx, rng)


class RefYoutubeVOS(TrackingDataset):
    """Refer-YouTube-VOS: Video Object Segmentation with Referring Expressions."""
    HF_SOURCE = "allenai/molmo2-ref-yt-vos"
    DATASET_NAME = "ref-yt-vos"
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, "Ref-YT-VOS")
    VIDEO_FPS = 6
    TASKS = ["track"]
    SPLIT_MAP = {
        "train": "train",
        "validation": "valid",
        "test": "valid",
    }
    MANUAL_DOWNLOAD_INSTRUCTION = """
1. Participate in Codalab competitions to download train.zip and valid.zip: https://competitions.codalab.org/competitions/29139#participate-get-data"
2. Place the downloaded zip files in the VIDEO_HOME directory (e.g. VIDEO_TRACK_DATA_HOME/Ref-YT-VOS/)
"""

    @classmethod
    def _prepare_annotation_dir(cls, n_procs=1):
        for split in set(cls.SPLIT_MAP.values()):
            zip_path = join(cls.VIDEO_HOME, f"{split}.zip")
            extract_path = join(cls.VIDEO_HOME, split)
            if exists(extract_path):
                log.info(f"{extract_path} already exists, skipping.")
                continue
            if not exists(zip_path):
                log.warning(f"Expected {zip_path} not found. Please download manually. {cls.MANUAL_DOWNLOAD_INSTRUCTION}")
                raise RuntimeError(f"Missing {zip_path}")
            log.info(f"Unzipping {zip_path} to {extract_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for member in tqdm(zf.infolist(), desc=split, leave=False):
                    zf.extract(member, cls.VIDEO_HOME)

    @classmethod
    def _precompute_gt_masks(cls):
        cls._precompute_gt_masks_for_split("valid")

    @classmethod
    def _precompute_gt_masks_for_split(cls, data_split):
        """Encode GT mask PNGs to COCO RLE and save as individual JSON files per query.

        Ref-YT-VOS masks: {split}/Annotations/{video_id}/{query_id}/*.png
        Saves to: VIDEO_HOME/{data_split}/MasksRLE/{video_id}/{query_id}.json
        """
        from pycocotools import mask as mask_utils

        VALID_N_QUERIES = 834

        annotations_dir = join(cls.VIDEO_HOME, data_split, "Annotations")
        if not exists(annotations_dir):
            log.warning(f"Annotations directory not found: {annotations_dir}")
            return

        # Load HF dataset to get (video_id, qid, w, h) for each example
        local_dir = cls._get_local_dir("track", data_split)
        config = cls._get_hf_config("track")
        ds = _load_hf_dataset(cls.HF_SOURCE, data_split, local_name=local_dir, config=config)

        hf_queries = {}
        for ex in ds:
            hf_queries[(ex['video'], ex['qid'])] = (ex['mask_id'], ex['width'], ex['height'])
        assert len(hf_queries) == VALID_N_QUERIES, \
            f"Expected {VALID_N_QUERIES} queries but found {len(hf_queries)} in HF dataset"

        # Quick check: if all expected mask files exist, skip
        output_dir = join(cls.VIDEO_HOME, data_split, "MasksRLE")
        if exists(output_dir):
            missing_masks = [
                (vid, qid) for (vid, qid) in hf_queries
                if not exists(join(output_dir, vid, f"{qid}.json"))
            ]
            if not missing_masks:
                log.info(f"[{cls.DATASET_NAME}] All {len(hf_queries)} GT masks present, skipping.")
                return

        n_encoded = 0
        n_skipped = 0
        for (video_id, query_id), (mask_id, w, h) in tqdm(hf_queries.items(), desc=f"Encoding GT masks ({data_split})"):
            query_output = join(output_dir, video_id, f"{query_id}.json")
            if exists(query_output):
                n_skipped += 1
                continue

            mask_dir = join(annotations_dir, video_id, query_id)
            png_paths = sorted(glob(join(mask_dir, "*.png")))
            assert png_paths, f"No mask PNGs found at {mask_dir}"

            rle_masks = []
            for png_path in png_paths:
                img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    rle_masks.append(None)
                    continue
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
                binary = (img >= 255).astype(np.uint8, order='F')
                rle = mask_utils.encode(binary)
                rle['counts'] = rle['counts'].decode('utf-8')
                rle_masks.append(rle)
            
            assert len(mask_id) == 1
            mask_annot = {mask_id[0]: rle_masks}
            os.makedirs(join(output_dir, video_id), exist_ok=True)
            with open(query_output, 'w') as f:
                json.dump(mask_annot, f)
            n_encoded += 1

        log.info(f"Precomputed GT masks: {n_encoded} new, {n_skipped} already exist, out of {len(hf_queries)} queries.")

    def get(self, idx, rng):
        item = super().get(idx, rng)
        if self.is_eval:
            ex = self.data[idx]
            masks_dir = join(self.VIDEO_HOME, self.data_split, "MasksRLE")
            masks_path = join(masks_dir, ex['video'], f"{ex['qid']}.json")
            with open(masks_path, 'r') as f:
                masks = json.load(f)
            if masks is not None:
                assert len(ex['mask_id']) == 1
                item['metadata']['masks'] = masks
                item['metadata']['mask_id'] = ex['mask_id']
                if "points" in item['message_list'][0]:
                    item['metadata']['points'] = item['message_list'][0]['points']
        return item


class RefDavis17(TrackingDataset):
    """Ref-DAVIS17: Referring Video Object Segmentation."""
    HF_SOURCE = "allenai/molmo2-ref-davis17"
    DATASET_NAME = "ref-davis17"
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, "Ref-DAVIS17")
    VIDEO_URL = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip'
    VIDEO_FPS = 6
    TASKS = ["track"]
    SPLIT_MAP = {
        "train": "train",
        "validation": "valid",
        "test": "valid",
    }

    @classmethod
    def _get_frames_dir(cls, data_split, video_name):
        return join(cls.VIDEO_HOME, 'DAVIS', 'JPEGImages', '480p', video_name)

    @classmethod
    def _prepare_annotation_dir(cls, n_procs=1):
        maybe_download_and_unzip(cls.VIDEO_HOME, cls.VIDEO_URL)

    @classmethod
    def _precompute_gt_masks(cls):
        cls._precompute_gt_masks_for_split("valid")

    @classmethod
    def _precompute_gt_masks_for_split(cls, data_split):
        """Encode GT mask PNGs to COCO RLE and save as individual JSON files per query.

        Davis17 masks are multi-object palette PNGs at DAVIS/Annotations/480p/{video_name}/{frame}.png
        where pixel value == obj_id gives the binary mask for that object.
        Saves to: VIDEO_HOME/{data_split}/MasksRLE/{video_name}/{query_id}.json
        """
        VALID_N_QUERIES = 244

        from pycocotools import mask as mask_utils
        from PIL import Image

        annotations_dir = join(cls.VIDEO_HOME, "DAVIS", "Annotations", "480p")
        if not exists(annotations_dir):
            log.warning(f"Annotations directory not found: {annotations_dir}")
            return

        local_dir = cls._get_local_dir("track", data_split)
        config = cls._get_hf_config("track")
        ds = _load_hf_dataset(cls.HF_SOURCE, data_split, local_name=local_dir, config=config)

        hf_queries = {}
        for ex in ds:
            hf_queries[(ex['video'], ex['qid'])] = (ex['width'], ex['height'], ex['obj_id'], ex['mask_id'])
        assert len(hf_queries) == VALID_N_QUERIES, \
            f"Expected {VALID_N_QUERIES} queries but found {len(hf_queries)} in HF dataset"

        # Quick check: if all expected mask files exist, skip
        output_dir = join(cls.VIDEO_HOME, data_split, "MasksRLE")
        if exists(output_dir):
            missing_masks = [
                (vid, qid) for (vid, qid) in hf_queries
                if not exists(join(output_dir, vid, f"{qid}.json"))
            ]
            if not missing_masks:
                log.info(f"[{cls.DATASET_NAME}] All {len(hf_queries)} GT masks present, skipping.")
                return

        n_encoded = 0
        n_skipped = 0
        for (video_id, query_id), (w, h, obj_ids, mask_id) in tqdm(hf_queries.items(), desc=f"Encoding GT masks ({data_split})"):
            video_name = os.path.splitext(video_id)[0]
            query_output = join(output_dir, video_name, f"{query_id}.json")
            if exists(query_output):
                n_skipped += 1
                continue

            mask_dir = join(annotations_dir, video_name)
            png_paths = sorted(glob(join(mask_dir, "*.png")))
            assert png_paths, f"No mask PNGs found at {mask_dir}"

            # Encode per-object masks keyed by mask_id
            mask_annot = {}
            for mid, obj_id in zip(mask_id, obj_ids):
                rle_masks = []
                for png_path in png_paths:
                    mask_img = np.array(Image.open(png_path))
                    if mask_img.shape[:2] != (h, w):
                        mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)
                    binary = (mask_img == obj_id).astype(np.uint8)
                    binary = np.asfortranarray(binary)
                    rle = mask_utils.encode(binary)
                    rle['counts'] = rle['counts'].decode('utf-8')
                    rle_masks.append(rle)
                mask_annot[mid] = rle_masks

            os.makedirs(join(output_dir, video_name), exist_ok=True)
            with open(query_output, 'w') as f:
                json.dump(mask_annot, f)
            n_encoded += 1

        log.info(f"Precomputed GT masks: {n_encoded} new, {n_skipped} already exist, out of {len(hf_queries)} queries.")

    def get(self, idx, rng):
        result = super().get(idx, rng)
        if self.is_eval:
            ex = self.data[idx]
            masks_dir = join(self.VIDEO_HOME, self.data_split, "MasksRLE")
            masks_path = join(masks_dir, ex['video'], f"{ex['qid']}.json")
            with open(masks_path, 'r') as f:
                masks = json.load(f)
            if masks is not None:
                result['metadata']['masks'] = masks
                result['metadata']['mask_id'] = ex['mask_id']
                if "points" in result['message_list'][0]:
                    result['metadata']['points'] = result['message_list'][0]['points']
        return result

class ReasonVOS(TrackingDataset):
    """
    [VideoLISA] One Token to Seg Them All: Language Instructed Reasoning Segmentation in Videos
    We use their ReasonVOS benchmark as only evaluation set.
    """
    HF_SOURCE = "allenai/molmo2-reasonvos"
    DATASET_NAME = "reasonvos"
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, "ReasonVOS")
    VIDEO_URL = "https://drive.google.com/file/d/1QMggGR91AE-iUUub5YwzMYiFt0TeV5_8/view?usp=sharing"
    REASONVOS_DIR = join(VIDEO_TRACK_DATA_HOME, "ReasonVOS", "ReasonVOS")
    TASKS = ["track"]
    SPLIT_MAP = {
        "test": "test",
    }

    @classmethod
    def _get_video_dir(cls, data_split):
        return join(cls.VIDEO_HOME, "videos")

    @classmethod
    def _get_frames_dir(cls, data_split, video_name):
        return join(cls.REASONVOS_DIR, "JPEGImages", video_name)

    @classmethod
    def _prepare_annotation_dir(cls, n_procs=1):
        maybe_download_and_unzip(
            cls.VIDEO_HOME,
            cls.VIDEO_URL,
            expected_dir="ReasonVOS/JPEGImages",
        )

    @classmethod
    def _precompute_gt_masks(cls):
        cls._precompute_gt_masks_for_split("test")

    @classmethod
    def _precompute_gt_masks_for_split(cls, data_split):
        """Encode GT mask PNGs to COCO RLE and save as individual JSON files per query.

        ReasonVOS annotations: ReasonVOS/Annotations/{source}_{video_id}_{obj_id}/*.png
        Saves to: VIDEO_HOME/MasksRLE/{video_id}/{query_id}.json
        """
        TEST_N_QUERIES = 458

        from pycocotools import mask as mask_utils

        annotations_dir = join(cls.REASONVOS_DIR, "Annotations")
        if not exists(annotations_dir):
            log.warning(f"Annotations directory not found: {annotations_dir}")
            return

        local_dir = cls._get_local_dir("track", data_split)
        config = cls._get_hf_config("track")
        ds = _load_hf_dataset(cls.HF_SOURCE, data_split, local_name=local_dir, config=config)

        assert len(ds) == TEST_N_QUERIES, \
            f"Expected {TEST_N_QUERIES} queries but found {len(ds)} in HF dataset"

        # Quick check: if all expected mask files exist, skip
        output_dir = join(cls.VIDEO_HOME, "MasksRLE")
        if exists(output_dir):
            missing_masks = [
                (vid, qid) for (vid, qid) in zip(ds['video'], ds['qid'])
                if not exists(join(output_dir, vid, f"{qid}.json"))
            ]
            if not missing_masks:
                log.info(f"[{cls.DATASET_NAME}] All {len(ds)} GT masks present, skipping.")
                return

        n_encoded = 0
        n_skipped = 0
        for ex in tqdm(ds, desc=f"Encoding GT masks ({data_split})"):
            video_id = ex['video']
            query_id = ex['qid']
            obj_ids = ex['obj_id']
            
            query_output = join(output_dir, video_id, f"{query_id}.json")
            if exists(query_output):
                n_skipped += 1
                continue

            # Annotation dir: {source}_{video_id}_{obj_id}
            anno_ids = ex['anno_id']
            assert len(obj_ids) == 1, f"Expected single obj_id for ReasonVOS, got {obj_ids}"
            assert len(anno_ids) == 1, f"Expected single anno_id for ReasonVOS, got {anno_ids}"
            mask_dir = join(annotations_dir, anno_ids[0])
            png_paths = sorted(glob(join(mask_dir, "*.png")))
            assert png_paths, f"No mask PNGs found at {mask_dir}"

            rle_masks = []
            for png_path in png_paths:
                img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    rle_masks.append(None)
                    continue
                img = cv2.resize(img, (ex['width'], ex['height']), interpolation=cv2.INTER_NEAREST)
                binary = (img >= 255).astype(np.uint8, order='F')
                rle = mask_utils.encode(binary)
                rle['counts'] = rle['counts'].decode('utf-8')
                rle_masks.append(rle)

            assert len(obj_ids) == 1, f"Expected single obj_id for ReasonVOS, got {obj_ids}"
            mask_annot = {obj_ids[0]: rle_masks}
            os.makedirs(join(output_dir, video_id), exist_ok=True)
            with open(query_output, 'w') as f:
                json.dump(mask_annot, f)
            n_encoded += 1

        log.info(f"Precomputed GT masks: {n_encoded} new, {n_skipped} already exist, out of {len(ds)} queries.")

    def get(self, idx, rng):
        item = super().get(idx, rng)
        if self.is_eval:
            ex = self.data[idx]
            masks_dir = join(self.VIDEO_HOME, "MasksRLE")
            masks_path = join(masks_dir, ex['video'], f"{ex['qid']}.json")
            with open(masks_path, 'r') as f:
                masks = json.load(f)
            if masks is not None:
                item['metadata']['masks'] = masks
                item['metadata']['mask_id'] = ex['mask_id']
                if "points" in item['message_list'][0]:
                    item['metadata']['points'] = item['message_list'][0]['points']
        return item


class Burst(TrackingDataset):
    """
    Annotations from Tracking Any Object Amodally (TAO-Amodal) dataset: https://github.com/WesleyHsieh0806/TAO-Amodal
    BURST uses the same videos as TAO but have segmentation mask instead of bbox.
    Make sure to sign agreement before downloading from TAO-Amodal hf repo.
    """
    HF_SOURCE = "allenai/molmo2-burst"
    DATASET_NAME = "burst"
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, "TAO-Amodal")
    VIDEO_FPS = 6
    TASKS = ["track", "ground", "single_point_track"]
    SPLIT_MAP = {
        "train": "train",
        # "validation": "val",
        # "test": "test",
    }

    @classmethod
    def download(cls, n_procs=1):
        if not cls.VIDEO_HOME:
            raise RuntimeError("VIDEO_TRACK_DATA_HOME not set")

        # Download raw frames
        log.info(f"Downloading  TAO-Amodal frames to {cls.VIDEO_HOME}...")
        snapshot_download(
            repo_id="chengyenhsieh/TAO-Amodal",
            repo_type="dataset",
            local_dir=cls.VIDEO_HOME,
            max_workers=n_procs,
        )

        # Unzip frame files
        for data_split in set(cls.SPLIT_MAP.values()):
            for data_source in ['ArgoVerse', 'AVA', 'BDD', 'Charades', 'HACS', 'LaSOT', 'YFCC100M']:
                zip_path = join(cls.VIDEO_HOME, 'frames', data_split,  f"{data_source}.zip")
                extract_path = join(cls.VIDEO_HOME, 'frames', data_split, data_source)
                if not exists(zip_path):
                    log.warning(f"Expected {zip_path} not found. Please download manually from https://huggingface.co/datasets/chengyenhsieh/TAO-Amodal/tree/main/{data_split}")
                elif not exists(extract_path):
                    log.info(f"Unzipping {zip_path} to {extract_path}...")
                    extract_dir = join(cls.VIDEO_HOME, 'frames', data_split, data_source)
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        for member in tqdm(zf.infolist(), desc=data_source, leave=False):
                            zf.extract(member, extract_dir)
                else:
                    log.info(f"{extract_path} already exists, skipping.")

        # Create videos from frames with desired FPS
        # NOTE: For BURST, we re-encode video original fps to target fps=6 to match the annotation fps.
        # We subsample frames every (original_fps / target_fps) frames when creating videos.
        local_dir = join(cls.LOCAL_DIR, cls.DATASET_NAME)
        hf_hub_download(
            repo_id=cls.HF_SOURCE, 
            filename="video_original_fps.json", 
            repo_type="dataset",
            local_dir=local_dir,
        )
        with open(join(local_dir, "video_original_fps.json"), 'r') as f:
            original_video_fps: dict = json.load(f)
        
        # Create videos from frames with desired FPS for all splits
        video_fps = cls._load_all_dataset_and_fps()
        work_items = []
        for (data_split, video_name), fps in video_fps.items():
            original_fps = original_video_fps[video_name]
            frames_dir = join(cls.VIDEO_HOME, 'frames', data_split, video_name)
            video_dir = cls._get_video_dir(data_split)
            output_path = join(video_dir, f"{video_name}.mp4")
            if not exists(output_path) and exists(frames_dir):
                work_items.append({'frames_dir': frames_dir, 'output_path': output_path, 'fps': fps, 'native_fps': original_fps})
        cls._create_videos(work_items, n_procs)

        # Verify videos
        cls._check_videos(video_fps)

class LVVIS(TrackingDataset):
    """LV-VIS: Large Vocabulary Video Instance Segmentation."""
    HF_SOURCE = "allenai/molmo2-lv-vis"
    DATASET_NAME = "lv-vis"
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, "LV-VIS")
    VIDEO_URL = "https://drive.google.com/file/d/1er2lBQLF75TI5O4wzGyur0YYoohMK6C3/view"
    VIDEO_FPS = 4
    TASKS = ["track", "ground", "single_point_track"]
    SPLIT_MAP = {
        "train": "train",
    }

    @classmethod
    def download(cls, n_procs=1):

        maybe_download_file( 
            cls.VIDEO_URL,
            join(cls.VIDEO_HOME, "train.zip"),
        )

        # Unzip train.zip
        zip_path = join(cls.VIDEO_HOME, "train.zip")
        extract_path = join(cls.VIDEO_HOME, "train")
        if not exists(extract_path):
            log.info(f"Unzipping {zip_path} to {extract_path}...")
            if not exists(zip_path):
                log.warning(f"Expected {zip_path} not found. Please download manually from {cls.VIDEO_URL}")
                raise RuntimeError(f"Missing {zip_path}")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for member in tqdm(zf.infolist(), desc="train.zip", leave=False):
                    zf.extract(member, cls.VIDEO_HOME)
        else:
            log.info(f"{extract_path} already exists, skipping.")
        
        # Create videos from frames with desired FPS for all splits
        video_fps = cls._load_all_dataset_and_fps()
        work_items = []
        for (data_split, video_name), fps in video_fps.items():
            frames_dir = join(cls.VIDEO_HOME, data_split, 'JPEGImages', video_name)
            video_dir = cls._get_video_dir(data_split)
            output_path = join(video_dir, f"{video_name}.mp4")
            if not exists(output_path) and exists(frames_dir):
                work_items.append({'frames_dir': frames_dir, 'output_path': output_path, 'fps': fps})
        cls._create_videos(work_items, n_procs)

        # Verify videos
        cls._check_videos(video_fps)

class YTVIS(TrackingDataset):
    """YouTube-VIS: Video Instance Segmentation."""
    HF_SOURCE = "allenai/molmo2-yt-vis"
    DATASET_NAME = "yt-vis"
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, "YT-VIS")
    VIDEO_URL = "https://drive.google.com/file/d/1U8c65FVL3wifZdk7qhNbBI1m2cn2fPHB/view"
    TASKS = ["track"]
    SPLIT_MAP = {
        "train": "train",
    }
    MANUAL_DOWNLOAD_INSTRUCTION = """
1. Participate in Codalab competitions to download train.zip: https://codalab.lisn.upsaclay.fr/competitions/3410#participate"
2. Place the downloaded zip files in the VIDEO_HOME directory (e.g. VIDEO_TRACK_DATA_HOME/YT-VIS/)
"""

    @classmethod
    def download(cls, n_procs=1):
        
        # Check and unzip train.zip in VIDEO_HOME
        zip_path = join(cls.VIDEO_HOME, "train.zip")
        extract_path = join(cls.VIDEO_HOME, "train")
        if not exists(zip_path):
            log.warning(f"Expected {zip_path} not found. Please download manually. {cls.MANUAL_DOWNLOAD_INSTRUCTION}")
            raise RuntimeError(f"Missing {zip_path}")
        elif not exists(extract_path):
            log.info(f"Unzipping {zip_path} to {extract_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for member in tqdm(zf.infolist(), desc="train.zip", leave=False):
                    zf.extract(member, cls.VIDEO_HOME)
        else:
            log.info(f"{extract_path} already exists, skipping.")
        
        # Create videos from frames with desired FPS for all splits
        video_fps = cls._load_all_dataset_and_fps()
        work_items = []
        for (data_split, video_name), fps in video_fps.items():
            frames_dir = join(cls.VIDEO_HOME, data_split, 'JPEGImages', video_name)
            video_dir = cls._get_video_dir(data_split)
            output_path = join(video_dir, f"{video_name}.mp4")
            if not exists(output_path) and exists(frames_dir):
                work_items.append({'frames_dir': frames_dir, 'output_path': output_path, 'fps': fps})
        cls._create_videos(work_items, n_procs)

        # Verify videos
        cls._check_videos(video_fps)


class ViCaS(TrackingDataset):
    """ViCaS: Video Camouflaged Animal Segmentation."""
    HF_SOURCE = "allenai/molmo2-vicas"
    DATASET_NAME = "vicas"
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, "ViCaS")
    VIDEO_FPS = 6
    TASKS = ["track", "ground", "single_point_track"]
    SPLIT_MAP = {
        "train": "train",
    }

    @classmethod
    def _extract_videos_to_frames(cls, work_items, n_procs=1):
        """Extract videos to frames, optionally in parallel."""
        if not work_items:
            log.info("No videos to extract.")
            return

        n = min(n_procs, len(work_items))
        log.info(f"Extracting frames for {len(work_items)} videos with {n} workers...")
        log.info(f"Example items: {work_items[:5]}")

        failed = []
        with mp.Pool(n) as pool:
            for video_path, success, error in tqdm(
                pool.imap_unordered(_extract_frames_worker, work_items),
                total=len(work_items), desc="Extracting frames",
            ):
                if not success:
                    failed.append((video_path, error))
        if failed:
            log.warning(f"{len(failed)}/{len(work_items)} videos failed to extract, e.g.: {failed[:5]}")
        else:
            log.info(f"All {len(work_items)} videos extracted successfully.")
    
    @classmethod
    def download(cls, n_procs=1):

        # Download videos
        log.info(f"Downloading ViCaS videos to {cls.VIDEO_HOME}...")

        if not exists(join(cls.VIDEO_HOME, 'videos')):
            snapshot_download(
                repo_id="kumuji/ViCaS",
                repo_type="dataset",
                local_dir=cls.VIDEO_HOME,
                local_dir_use_symlinks=False,
                max_workers=n_procs,
            )

        # Decode videos to frames (if not already done) to ensure consistent fps and re-encode at target fps=6.
        extract_items = []
        for video_dir in ['videos', 'videos2', 'videos3']:
            video_dir_path = join(cls.VIDEO_HOME, video_dir)
            if not exists(video_dir_path):
                log.warning(f"Expected {video_dir_path} not found. Please check if videos are downloaded correctly.")
                continue
            for video_file in tqdm(os.listdir(video_dir_path), desc=f"Decoding {video_dir}", leave=False):
                if video_file.endswith(".mp4"):
                    video_path = join(video_dir_path, video_file)
                    video_name = video_file.split('_')[0]
                    frames_dir = join(cls.VIDEO_HOME, "JPEGImages", video_name)
                    if not exists(frames_dir):
                        extract_items.append({'video_path': video_path, 'out_folder': frames_dir})
        cls._extract_videos_to_frames(extract_items, n_procs)

        # Create videos from frames with desired FPS
        # NOTE: For ViCaS, we re-encode video original fps to target fps=6 to match the annotation fps.
        # We subsample frames every (original_fps / target_fps) frames when creating videos.
        local_dir = join(cls.LOCAL_DIR, cls.DATASET_NAME)
        hf_hub_download(
            repo_id=cls.HF_SOURCE, 
            filename="video_original_fps.json", 
            repo_type="dataset",
            local_dir=local_dir,
        )
        with open(join(local_dir, "video_original_fps.json"), 'r') as f:
            original_video_fps: dict = json.load(f)

        # Create videos from frames with desired FPS
        video_fps = cls._load_all_dataset_and_fps()
        work_items = []
        for (data_split, video_name), fps in video_fps.items():
            original_fps = original_video_fps[video_name]
            frames_dir = join(cls.VIDEO_HOME, 'JPEGImages', video_name)
            video_dir = cls._get_video_dir(data_split)
            output_path = join(video_dir, f"{video_name}.mp4")
            if not exists(output_path) and exists(frames_dir):
                work_items.append({'frames_dir': frames_dir, 'output_path': output_path, 'fps': fps, 'native_fps': original_fps})
        cls._create_videos(work_items, n_procs)

        # Verify videos
        cls._check_videos(video_fps)


class ReVOS(TrackingDataset):
    """ReVOS: Reasoning Video Object Segmentation."""
    HF_SOURCE = "allenai/molmo2-revos"
    DATASET_NAME = "revos"
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, "ReVOS")
    VIDEO_FPS = 6
    TASKS = ["track", "ground", "single_point_track"]
    SPLIT_MAP = {
        "train": "train",
    }
    MANUAL_DOWNLOAD_INSTRUCTION = """
    Videos must be downloaded manually from OneDrive.
    1. Visit https://github.com/cilinyan/ReVOS-api and click the OneDrive link to request access.
    2. Download the "JPEGImages.zip" file and place it in the VIDEO_HOME directory (e.g. VIDEO_TRACK_DATA_HOME/ReVOS/)
"""
    
    @classmethod
    def download(cls, n_procs=1):

        # Check and unzip JPEGImages.zip in VIDEO_HOME
        zip_path = join(cls.VIDEO_HOME, "JPEGImages.zip")
        extract_path = join(cls.VIDEO_HOME, "JPEGImages")
        if not exists(extract_path):
            if not exists(zip_path):
                log.warning(f"Expected {zip_path} not found. Please download manually from OneDrive. {cls.MANUAL_DOWNLOAD_INSTRUCTION}")
                raise RuntimeError(f"Missing {zip_path}")
            log.info(f"Unzipping {zip_path} to {extract_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for member in tqdm(zf.infolist(), desc="JPEGImages.zip", leave=False):
                    zf.extract(member, extract_path) # Actually unzips to VIDEO_HOME/JPEGImages/{video_id} directly
        else:
            log.info(f"Videos already exist, skipping extraction.")

        # Create videos from frames with desired FPS
        video_fps = cls._load_all_dataset_and_fps()
        work_items = []
        for (data_split, video_name), fps in video_fps.items():
            frames_dir = join(cls.VIDEO_HOME, "JPEGImages", video_name)
            video_dir = cls._get_video_dir(data_split)
            output_path = join(video_dir, f"{video_name}.mp4")
            if not exists(output_path) and exists(frames_dir):
                work_items.append({'frames_dir': frames_dir, 'output_path': output_path, 'fps': fps})
        cls._create_videos(work_items, n_procs)


class MoCA(TrackingDataset):
    """MoCA: Moving Camouflaged Animals Dataset."""
    HF_SOURCE = "allenai/molmo2-moca"
    DATASET_NAME = "moca"
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, "MoCA")
    VIDEO_URL = 'https://thor.robots.ox.ac.uk/datasets/MoCA/MoCA.zip'
    TASKS = ["track", "ground"]
    SPLIT_MAP = {
        "train": "train",
    }

    @classmethod
    def _get_video_dir(cls, data_split):
        return join(cls.VIDEO_HOME, "videos")
    
    @classmethod
    def download(cls, n_procs=1):

        # Download file
        maybe_download_file( 
            cls.VIDEO_URL,
            join(cls.VIDEO_HOME, "MoCA.zip"),
        )

        # Unzip MoCA.zip
        zip_path = join(cls.VIDEO_HOME, "MoCA.zip")
        extract_path = cls.VIDEO_HOME
        if not exists(join(extract_path, "MoCA", "JPEGImages")): # MoCA.zip has a MoCA/ root inside
            if not exists(zip_path):
                log.warning(f"Expected {zip_path} not found. Please download manually from {cls.VIDEO_URL}")
                raise RuntimeError(f"Missing {zip_path}")
            log.info(f"Unzipping {zip_path} to {extract_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for member in tqdm(zf.infolist(), desc="MoCA.zip", leave=False):
                    zf.extract(member, extract_path)
        else:
            log.info(f"Videos already exist, skipping extraction.")
        
        # Create videos from frames with desired FPS for all splits
        video_fps = cls._load_all_dataset_and_fps()
        work_items = []
        for (data_split, video_name), fps in video_fps.items():
            frames_dir = join(cls.VIDEO_HOME, "MoCA", "JPEGImages", video_name)
            video_dir = cls._get_video_dir(data_split)
            output_path = join(video_dir, f"{video_name}.mp4")
            if not exists(output_path) and exists(frames_dir):
                work_items.append({'frames_dir': frames_dir, 'output_path': output_path, 'fps': fps})
        cls._create_videos(work_items, n_procs)

        # Verify videos
        cls._check_videos(video_fps)

# ── Single Object Tracking Datasets ────────────────────────────────────────
# Source: SOT benchmarks (bbox annotations -> segmentation masks -> point tracks).
# All share a single HF repo with dataset_name as config.
# Supports only one task: single_point_track (given one starting point, track the object).
class SingleObjectTrackingDataset(TrackingDataset):
    HF_SOURCE = "allenai/molmo2-single-object-track"
    DATASET_NAME = None
    VIDEO_HOME = None
    TASKS = ["single_point_track"]
    SPLIT_MAP = {
        "train": "train",
    }

    @classmethod
    def _get_hf_config(cls, task):
        """SOT repo uses dataset_name as config, not task."""
        return cls.DATASET_NAME

class LVOSv1(SingleObjectTrackingDataset):
    """LVOSv1: Large Vocabulary Object Tracking."""
    DATASET_NAME = "lvosv1"
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, "LVOSv1")
    VIDEO_URL = "https://drive.google.com/file/d/1pdA1Y7-VE4coj6yacya-kolZs6hKuQpS/view"

    @classmethod
    def download(cls, n_procs=1):
        # Download and extract frames
        maybe_download_and_unzip(cls.VIDEO_HOME, cls.VIDEO_URL, expected_dir="train")

        # Create videos from frames
        video_fps = cls._load_all_dataset_and_fps()
        work_items = []
        for (data_split, video_name), fps in video_fps.items():
            frames_dir = join(cls.VIDEO_HOME, data_split, 'JPEGImages', video_name)
            video_dir = cls._get_video_dir(data_split)
            output_path = join(video_dir, f"{video_name}.mp4")
            if not exists(output_path) and exists(frames_dir):
                work_items.append({'frames_dir': frames_dir, 'output_path': output_path, 'fps': fps})
        cls._create_videos(work_items, n_procs)
        cls._check_videos(video_fps)

class LVOSv2(SingleObjectTrackingDataset):
    """LVOSv2: Large Vocabulary Object Tracking."""
    DATASET_NAME = "lvosv2"
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, "LVOSv2")
    VIDEO_URL = "https://drive.google.com/file/d/1-ehpl5s0Fd14WwtT-GmWtIWa_BxZl9D6/view"

    @classmethod
    def download(cls, n_procs=1):
        # Download and extract frames
        maybe_download_and_unzip(cls.VIDEO_HOME, cls.VIDEO_URL, expected_dir="train")

        # Create videos from frames
        video_fps = cls._load_all_dataset_and_fps()
        work_items = []
        for (data_split, video_name), fps in video_fps.items():
            frames_dir = join(cls.VIDEO_HOME, data_split, 'JPEGImages', video_name)
            video_dir = cls._get_video_dir(data_split)
            output_path = join(video_dir, f"{video_name}.mp4")
            if not exists(output_path) and exists(frames_dir):
                work_items.append({'frames_dir': frames_dir, 'output_path': output_path, 'fps': fps})
        cls._create_videos(work_items, n_procs)
        cls._check_videos(video_fps)


class LaSOT(SingleObjectTrackingDataset):
    """LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking.

    Auto-downloads from HuggingFace: https://huggingface.co/datasets/l-lt/LaSOT
    HF structure: annotation/{class}.zip -> annotation/{class}-{idx}/img/*.jpg
    """
    DATASET_NAME = "lasot"
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, "LaSOT")
    HF_FRAMES_REPO = "l-lt/LaSOT"

    @classmethod
    def download(cls, n_procs=1):
        annotation_dir = join(cls.VIDEO_HOME, "annotation")

        # Download all per-class zips from HF
        if not exists(annotation_dir):
            log.info(f"[LaSOT] Downloading from {cls.HF_FRAMES_REPO}...")
            snapshot_download(
                repo_id=cls.HF_FRAMES_REPO,
                repo_type="dataset",
                local_dir=annotation_dir,
                local_dir_use_symlinks=False,
                max_workers=n_procs,
            )
        else:
            log.info(f"[LaSOT] annotation_dir already exists, skipping download: {annotation_dir}")

        # Extract each per-class zip: {class}.zip -> annotation/{class}-1/, {class}-2/, ...
        # Skip a zip if any {class}-* directory already exists (already extracted).
        class_zips = sorted(glob(join(annotation_dir, "*.zip")))
        to_extract = [z for z in class_zips
                      if not glob(join(annotation_dir,
                                       f"{os.path.splitext(os.path.basename(z))[0]}-*"))]
        if to_extract:
            log.info(f"[LaSOT] Extracting {len(to_extract)}/{len(class_zips)} class zips "
                     f"to {annotation_dir}...")
            for zip_path in tqdm(to_extract, desc="LaSOT extract classes", leave=False):
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(annotation_dir)
        else:
            log.info(f"[LaSOT] All {len(class_zips)} class zips already extracted.")

        # Create videos: frames at annotation/{video_name}/img/*.jpg
        video_fps = cls._load_all_dataset_and_fps()
        work_items = []
        for (data_split, video_name), fps in video_fps.items():
            frames_dir = join(annotation_dir, video_name, "img")
            video_dir = cls._get_video_dir(data_split)
            output_path = join(video_dir, f"{video_name}.mp4")
            if not exists(output_path) and exists(frames_dir):
                work_items.append({'frames_dir': frames_dir, 'output_path': output_path, 'fps': fps})
        cls._create_videos(work_items, n_procs)
        cls._check_videos(video_fps)


class UWCOT(SingleObjectTrackingDataset):
    """UW-COT220: Underwater Camouflaged Object Tracking Meets Vision-Language SAM2.

    Github repo: https://github.com/983632847/Awesome-Multimodal-Object-Tracking/tree/main/UW-COT220
    Downloads UW-COT220.zip and extract to VIDEO_TRACK_DATA_HOME/UW-COT220/
    """
    DATASET_NAME = "uwcot"
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, "UW-COT220")
    VIDEO_URL = "https://drive.google.com/file/d/1WlzOujduTPcoSSh8XujwMk-GGtMMTTx1/view?usp=drive_link"

    @classmethod
    def _get_video_dir(cls, data_split):
        # UW-COT220 already has videos decoded, so use the video as it is.
        return join(cls.VIDEO_HOME, "UW-COT220") # VIDEO_HOME/UW-COT220/UW-COT220/{video_id}/{video_id}.mp4


    @classmethod
    def download(cls, n_procs=1):

        maybe_download_and_unzip(cls.VIDEO_HOME, cls.VIDEO_URL, expected_dir="UW-COT220")

        # zip file is already in {video_id}/{video_id}.mp4 format
        video_fps = cls._load_all_dataset_and_fps()
        cls._check_videos(video_fps)


class WebUOT(SingleObjectTrackingDataset):
    """WebUOT: Web-collected Underwater Object Tracking.

    Github repo: https://github.com/983632847/Awesome-Multimodal-Object-Tracking/tree/main/WebUOT-1M
    Downloads Train.zip and extract to VIDEO_TRACK_DATA_HOME/webuot/
    """
    DATASET_NAME = "webuot"
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, "webuot")
    VIDEO_URL = "https://drive.google.com/file/d/1IFXUY04H6xyisBdwEqw-9CZ4MJ_cfX82/view?usp=drive_link"

    @classmethod
    def _get_video_dir(cls, data_split):
        # WebUOT already has videos decoded, so use the video as it is.
        return join(cls.VIDEO_HOME, "Train") # VIDEO_HOME/webuot/Train/{video_id}/{video_id}.mp4

    @classmethod
    def download(cls, n_procs=1):
        maybe_download_and_unzip(cls.VIDEO_HOME, cls.VIDEO_URL, expected_dir="Train")

        video_fps = cls._load_all_dataset_and_fps()
        cls._check_videos(video_fps)

class LaTOT(SingleObjectTrackingDataset):
    """LaTOT: Large-scale Tiny Object Tracking.

    Manual download (Google Drive, 2 parts):
    1. Part 1: https://drive.google.com/drive/folders/1eMKQnRPLTC9URhiW0eRQ1z6tqwh9YB-k
    2. Part 2: https://drive.google.com/drive/folders/1u5PHyt20RmwvsgNsUf7TVFfwZ3Mc3pko
    3. Extract both to VIDEO_TRACK_DATA_HOME/LaTOT/
    """
    DATASET_NAME = "latot"
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, "LaTOT")
    MANUAL_DOWNLOAD_INSTRUCTION = """
        1. "LaTOT requires manual download (2 parts)."
            - "Download Part 1: https://drive.google.com/drive/folders/1eMKQnRPLTC9URhiW0eRQ1z6tqwh9YB-k"
            - "Download Part 2: https://drive.google.com/drive/folders/1u5PHyt20RmwvsgNsUf7TVFfwZ3Mc3pko"
        2. Unrar both parts and place the extracted data in VIDEO_TRACK_DATA_HOME/LaTOT/LaTOT/
            - unrar x 'LATOT_part1/*.rar' . (notice the 'LATOT' typo)
            - unrar x 'LaTOT_part2/*.rar' .
        3. Frames should be in VIDEO_TRACK_DATA_HOME/LaTOT/LaTOT/{video_id}/img/*.jpg and videos will be created from these frames.
"""

    @classmethod
    def _get_video_dir(cls, data_split):
        # LaTOT already has videos decoded, so use the video as it is.
        return join(cls.VIDEO_HOME, 'videos') # VIDEO_HOME/LaTOT/videos/{video_id}.mp4

    @classmethod
    def download(cls, n_procs=1):
        if not exists(join(cls.VIDEO_HOME, 'LaTOT')):
            raise RuntimeError(cls.MANUAL_DOWNLOAD_INSTRUCTION)

        video_fps = cls._load_all_dataset_and_fps()
        work_items = []
        for (data_split, video_name), fps in video_fps.items():
            frames_dir = join(cls.VIDEO_HOME, 'LaTOT', video_name, 'img')
            video_dir = cls._get_video_dir(data_split)
            output_path = join(video_dir, f"{video_name}.mp4")
            if not exists(output_path) and exists(frames_dir):
                work_items.append({'frames_dir': frames_dir, 'output_path': output_path, 'fps': fps})
        cls._create_videos(work_items, n_procs)
        cls._check_videos(video_fps)


class TNL2K(SingleObjectTrackingDataset):
    """TNL2K: Towards More Flexible and Accurate Object Tracking with Natural Language.

    Manual download from https://github.com/wangxiao5791509/TNL2K_evaluation_toolkit
    1. Use this onedrive link: https://github.com/wangxiao5791509/TNL2K_evaluation_toolkit
    2. Download the TNL2K_train_subset/TNL2k_train_subset_*.zip to VIDEO_TRACK_DATA_HOME/TNL2K
    """
    DATASET_NAME = "tnl2k"
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, "TNL2K")
    MANUAL_DOWNLOAD_INSTRUCTION = (
        "TNL2K requires manual download from GitHub.\n"
        "Download from: https://github.com/wangxiao5791509/TNL2K_evaluation_toolkit"
        "Download the TNL2K_train_subset/TNL2k_train_subset_*.zip to VIDEO_TRACK_DATA_HOME/TNL2K/zip_files"
    )

    @classmethod
    def _get_video_dir(cls, data_split):
        return join(cls.VIDEO_HOME, "videos") # VIDEO_HOME/TNL2K/videos/{video_id}.mp4

    @classmethod
    def download(cls, n_procs=1):
        if not exists(cls.VIDEO_HOME):
            raise RuntimeError(cls.MANUAL_DOWNLOAD_INSTRUCTION)
        
        # unzip
        subset_ids = ['p1', 'p2_1', 'p2_2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13']
        for subset_id in subset_ids:
            zip_path = join(cls.VIDEO_HOME, f"zip_files/TNL2k_train_subset_{subset_id}.zip")
            extract_path = join(cls.VIDEO_HOME, "train")
            if not exists(extract_path):
                log.info(f"Unzipping {zip_path} to {extract_path}...")
                if not exists(zip_path):
                    log.warning(f"Expected {zip_path} not found. Please download manually from {cls.VIDEO_URL}")
                    raise RuntimeError(f"Missing {zip_path}")
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for member in tqdm(zf.infolist(), desc=f"zip_files/TNL2k_train_subset_{subset_id}.zip", leave=False):
                        zf.extract(member, extract_path)
            else:
                log.info(f"{extract_path} already exists, skipping.")

        # create videos
        video_fps = cls._load_all_dataset_and_fps()
        work_items = []
        for (data_split, video_name), fps in video_fps.items():
            frames_dir = join(cls.VIDEO_HOME, video_name, 'imgs')
            video_dir = cls._get_video_dir(data_split)
            output_path = join(video_dir, f"{video_name}.mp4")
            if not exists(output_path) and exists(frames_dir):
                work_items.append({'frames_dir': frames_dir, 'output_path': output_path, 'fps': fps})
        cls._create_videos(work_items, n_procs)
        cls._check_videos(video_fps)

class TNLLT(SingleObjectTrackingDataset):
    """TNLLT: Tracking with Natural Language and Long-Term.

    Manual download (Dropbox, multi-part):
    1. Download all parts from https://www.dropbox.com/scl/fo/yr5avjhdvgn4btev5a2wg/AIAA3H31e_s8pWtGA7EK14M?rlkey=ny3tw5uttdzqrvs36mp1wc2j8&st=mtd9mx6n&dl=0
    2. Reassemble and extract:
       cat TNLLT_part_* > TNLLT_restored.tar.gz
       gunzip TNLLT_restored.tar.gz
       tar -xvf TNLLT_restored.tar
    3. Place extracted data in VIDEO_TRACK_DATA_HOME/TNLLT/
    """
    DATASET_NAME = "tnnlt"
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, "TNLLT")
    MANUAL_DOWNLOAD_INSTRUCTION = (
        "TNLLT requires manual download (Dropbox, multi-part).\n"
        "1. Download from: https://www.dropbox.com/scl/fo/yr5avjhdvgn4btev5a2wg/AIAA3H31e_s8pWtGA7EK14M?rlkey=ny3tw5uttdzqrvs36mp1wc2j8&st=mtd9mx6n&dl=0\n"
        "2. Reassemble:\n"
        "   cat TNLLT_part_* > TNLLT_restored.tar.gz\n"
        "   gunzip TNLLT_restored.tar.gz\n"
        "   tar -xvf TNLLT_restored.tar\n"
        "3. Place in VIDEO_TRACK_DATA_HOME/TNLLT/frames"
    )

    @classmethod
    def _get_video_dir(cls, data_split):
        return join(cls.VIDEO_HOME, "videos") # VIDEO_HOME/TNLLT/videos/{video_id}.mp4

    @classmethod
    def download(cls, n_procs=1):
        if not exists(cls.VIDEO_HOME):
            raise RuntimeError(cls.MANUAL_DOWNLOAD_INSTRUCTION)
        # create videos
        video_fps = cls._load_all_dataset_and_fps()
        work_items = []
        for (data_split, video_name), fps in video_fps.items():
            frames_dir = join(cls.VIDEO_HOME, 'frames', video_name, 'imgs')
            video_dir = cls._get_video_dir(data_split)
            output_path = join(video_dir, f"{video_name}.mp4")
            if not exists(output_path) and exists(frames_dir):
                work_items.append({'frames_dir': frames_dir, 'output_path': output_path, 'fps': fps})
        cls._create_videos(work_items, n_procs)
        cls._check_videos(video_fps)

class WebUAV(SingleObjectTrackingDataset):
    """WebUAV: Web Unmanned Aerial Vehicle Tracking.

    Requires access request form:
    1. Fill out: https://docs.google.com/forms/d/e/1FAIpQLSe5Usq9VUSGjKollBCI1heln_o6u4SuiMcBRn_FNqp4v2d0Kw/viewform
    2. Download and run `zip -s 0 zip/Train/Train.zip --out Train_Single.zip` inside the WebUAV-3M/Dataset directory
    3. Copy Train_Single.zip to VIDEO_TRACK_DATA_HOME/WebUAV/
    """
    DATASET_NAME = "webuav"
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, "WebUAV")
    MANUAL_DOWNLOAD_INSTRUCTION = (
        "WebUAV requires an access request.\n"
        "1. Fill out: https://docs.google.com/forms/d/e/1FAIpQLSe5Usq9VUSGjKollBCI1heln_o6u4SuiMcBRn_FNqp4v2d0Kw/viewform\n"
        "2. Download and run `zip -s 0 zip/Train/Train.zip --out Train_Single.zip` inside the WebUAV-3M/Dataset directory"
        "3. Copy Train_Single.zip to VIDEO_TRACK_DATA_HOME/WebUAV/"
    )

    @classmethod
    def _get_video_dir(cls, data_split):
        return join(cls.VIDEO_HOME, "videos") # VIDEO_HOME/TNLLT/videos/{video_id}.mp4

    @classmethod
    def download(cls, n_procs=1):
        if not exists(cls.VIDEO_HOME):
            raise RuntimeError(cls.MANUAL_DOWNLOAD_INSTRUCTION)
        
        # unzip
        zip_path = join(cls.VIDEO_HOME, f"Train_Single.zip")
        extract_path = join(cls.VIDEO_HOME, "Train")
        if not exists(extract_path):
            log.info(f"Unzipping {zip_path} to {extract_path}...")
            if not exists(zip_path):
                log.warning(f"Expected {zip_path} not found. Please download manually from {cls.VIDEO_URL}")
                raise RuntimeError(f"Missing {zip_path}")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for member in tqdm(zf.infolist(), desc=f"Train_Single.zip", leave=False):
                    zf.extract(member, extract_path)
        else:
            log.info(f"{extract_path} already exists, skipping.")

        # create videos
        video_fps = cls._load_all_dataset_and_fps()
        work_items = []
        for (data_split, video_name), fps in video_fps.items():
            frames_dir = join(cls.VIDEO_HOME, 'Train', video_name, 'img')
            video_dir = cls._get_video_dir(data_split)
            output_path = join(video_dir, f"{video_name}.mp4")
            if not exists(output_path) and exists(frames_dir):
                work_items.append({'frames_dir': frames_dir, 'output_path': output_path, 'fps': fps})
        cls._create_videos(work_items, n_procs)
        cls._check_videos(video_fps)


class GOT10k(SingleObjectTrackingDataset):
    """GOT-10k: Generic Object Tracking Benchmark.

    Requires account registration:
    1. Register at http://got-10k.aitestunion.com/downloads
    2. Download the full_data.zip and put it under VIDEO_TRACK_DATA_HOME/GOT-10k"
    """
    DATASET_NAME = "got10k"
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, "GOT-10k")
    MANUAL_DOWNLOAD_INSTRUCTION = (
        "GOT-10k requires account registration.\n"
        "1. Register at http://got-10k.aitestunion.com/downloads\n"
        "2. Download the full_data.zip and put it under VIDEO_TRACK_DATA_HOME/GOT-10k"
    )

    @classmethod
    def _get_video_dir(cls, data_split):
        return join(cls.VIDEO_HOME, "videos") # VIDEO_HOME/TNLLT/videos/{video_id}.mp4

    @classmethod
    def download(cls, n_procs=1):
        if not exists(cls.VIDEO_HOME):
            raise RuntimeError(cls.MANUAL_DOWNLOAD_INSTRUCTION)
        
        # unzip
        zip_path = join(cls.VIDEO_HOME, f"full_data.zip")
        extract_path = join(cls.VIDEO_HOME, "train")
        if not exists(extract_path):
            log.info(f"Unzipping {zip_path} to {extract_path}...")
            if not exists(zip_path):
                log.warning(f"Expected {zip_path} not found. Please download manually from {cls.VIDEO_URL}")
                raise RuntimeError(f"Missing {zip_path}")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for member in tqdm(zf.infolist(), desc=f"full_data.zip", leave=False):
                    zf.extract(member, extract_path)
        else:
            log.info(f"{extract_path} already exists, skipping.")
        
        # create videos
        video_fps = cls._load_all_dataset_and_fps()
        work_items = []
        for (data_split, video_name), fps in video_fps.items():
            frames_dir = join(cls.VIDEO_HOME, 'train',video_name)
            video_dir = cls._get_video_dir(data_split)
            output_path = join(video_dir, f"{video_name}.mp4")
            if not exists(output_path) and exists(frames_dir):
                work_items.append({'frames_dir': frames_dir, 'output_path': output_path, 'fps': fps})
        cls._create_videos(work_items, n_procs)
        cls._check_videos(video_fps)


class VastTrack(SingleObjectTrackingDataset):
    """VastTrack: Vast Category Visual Object Tracking.

    Manual download (OneDrive):
    1. Download from https://1drv.ms/f/s!AnWdA-LZ-BEt5W9kQtMU8nB19qpy?e=IYm3eF
    2. Put all zip files under VIDEO_TRACK_DATA_HOME/VastTrack/zip_files
    """
    DATASET_NAME = "vasttrack"
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, "VastTrack")
    MANUAL_DOWNLOAD_INSTRUCTION = (
        "VastTrack requires manual download from OneDrive.\n"
        "Download from: https://1drv.ms/f/s!AnWdA-LZ-BEt5W9kQtMU8nB19qpy?e=IYm3eF\n"
        "Put all zip files under VIDEO_TRACK_DATA_HOME/VastTrack/zip_files"
    )

    @classmethod
    def _get_video_dir(cls, data_split):
        return join(cls.VIDEO_HOME, "videos") # VIDEO_HOME/TNLLT/videos/{video_id}.mp4

    @classmethod
    def download(cls, n_procs=1):
        if not exists(cls.VIDEO_HOME):
            raise RuntimeError(cls.MANUAL_DOWNLOAD_INSTRUCTION)

        # unzip
        for subset_id in range(1, 83):
            zip_path = join(cls.VIDEO_HOME, f"zip_files/part-{subset_id}.zip")
            extract_path = join(cls.VIDEO_HOME, "train")
            if not exists(extract_path):
                log.info(f"Unzipping {zip_path} to {extract_path}...")
                if not exists(zip_path):
                    log.warning(f"Expected {zip_path} not found. Please download manually from {cls.VIDEO_URL}")
                    raise RuntimeError(f"Missing {zip_path}")
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for member in tqdm(zf.infolist(), desc=f"zip_files/part-{subset_id}.zip", leave=False):
                        zf.extract(member, extract_path)
            else:
                log.info(f"{extract_path} already exists, skipping.")

        video_fps = cls._load_all_dataset_and_fps()
        work_items = []
        for (data_split, video_name), fps in video_fps.items():
            frames_dir = join(cls.VIDEO_HOME, 'train', re.sub(r'-\d+$', '', video_name), video_name, 'imgs')
            video_dir = cls._get_video_dir(data_split)
            output_path = join(video_dir, f"{video_name}.mp4")
            if not exists(output_path) and exists(frames_dir):
                work_items.append({'frames_dir': frames_dir, 'output_path': output_path, 'fps': fps})
        cls._create_videos(work_items, n_procs)
        cls._check_videos(video_fps)


class TrackingNet(SingleObjectTrackingDataset):
    """TrackingNet: A Large-Scale Dataset and Benchmark for Object Tracking in the Wild.

    Auto-downloads from HuggingFace: https://huggingface.co/datasets/SilvioGiancola/TrackingNet
    Frames are downloaded per-chunk (TRAIN_0 through TRAIN_11), ~90GB each (~1TB total).
    Use chunks=[0,1,...,11] or chunks="all" to download everything.

    Example:
        TrackingNet.download()                  # All 12 chunks (~1TB)
        TrackingNet.download(chunks=[0])        # TRAIN_0 only (~90GB)
        TrackingNet.download(chunks=[0, 3, 7])  # Specific chunks
    """
    DATASET_NAME = "trackingnet"
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, "TrackingNet")
    HF_FRAMES_REPO = "SilvioGiancola/TrackingNet"
    ALL_CHUNKS = [f"TRAIN_{i}" for i in range(12)]

    @classmethod
    def _get_video_dir(cls, data_split):
        return join(cls.VIDEO_HOME, "videos") # VIDEO_HOME/TNLLT/videos/{video_id}.mp4

    @classmethod
    def download(cls, n_procs=1, chunks=None):
        """Download TrackingNet frame chunks and create videos.

        Args:
            n_procs: Number of parallel workers for video creation.
            chunks: Which chunks to download. Options:
                - None (default): all 12 chunks (~1TB total)
                - list of ints, e.g. [0, 3, 7]: specific chunk indices
        """
        if chunks is None:
            selected = cls.ALL_CHUNKS
            log.warning(f"[TrackingNet] Downloading ALL 12 chunks (~1TB total).")
        else:
            selected = [f"TRAIN_{i}" for i in chunks]
            log.info(f"[TrackingNet] Downloading chunks: {selected}")

        # HF structure: TrackingNet/{chunk}/zips/{video}.zip  (one zip per video)
        #
        # Skip chunks whose frames/ dir already exists, then:
        #   - All chunks remaining -> snapshot_download the whole repo (max parallelism)
        #   - Subset remaining     -> one snapshot_download per chunk with allow_patterns
        to_download = [c for c in selected
                       if not exists(join(cls.VIDEO_HOME, c, "frames"))]
        already_done = [c for c in selected if c not in to_download]
        if already_done:
            log.info(f"[TrackingNet] Skipping {len(already_done)} chunk(s) with frames already present: {already_done}")

        if not to_download:
            log.info(f"[TrackingNet] All selected chunks already have frames, skipping download.")
        elif to_download == cls.ALL_CHUNKS:
            log.info(f"[TrackingNet] Downloading full repo from {cls.HF_FRAMES_REPO} "
                     f"with {n_procs} workers...")
            snapshot_download(
                repo_id=cls.HF_FRAMES_REPO,
                repo_type="dataset",
                local_dir=cls.VIDEO_HOME,
                local_dir_use_symlinks=False,
                max_workers=n_procs,
            )
        else:
            for chunk in to_download:
                log.info(f"[TrackingNet] Downloading {chunk} from {cls.HF_FRAMES_REPO}...")
                snapshot_download(
                    repo_id=cls.HF_FRAMES_REPO,
                    repo_type="dataset",
                    local_dir=cls.VIDEO_HOME,
                    allow_patterns=[f"{chunk}/zips/*.zip"],
                    local_dir_use_symlinks=False,
                    max_workers=n_procs,
                )

        # Extract each per-video zip to {chunk}/frames/
        for chunk in selected:
            zips_dir = join(cls.VIDEO_HOME, chunk, "zips")
            frames_dir = join(cls.VIDEO_HOME, chunk, "frames")

            if exists(frames_dir):
                log.info(f"[TrackingNet] {chunk}: frames already exist at {frames_dir}, skipping extraction.")
                continue

            video_zips = sorted(glob(join(zips_dir, "*.zip")))
            if not video_zips:
                log.warning(f"[TrackingNet] No zips found at {zips_dir}, skipping extraction.")
                continue

            os.makedirs(frames_dir, exist_ok=True)
            to_extract = [z for z in video_zips
                          if not exists(join(frames_dir, os.path.splitext(os.path.basename(z))[0]))]
            log.info(f"[TrackingNet] {chunk}: extracting {len(to_extract)}/{len(video_zips)} video zips...")
            for zip_path in tqdm(to_extract, desc=f"{chunk} extract", leave=False):
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(frames_dir)

        # Create videos from frames (only for available chunks)
        video_fps = cls._load_all_dataset_and_fps()
        work_items = []
        for (data_split, video_name), fps in video_fps.items():
            # TrackingNet frames are at TrackingNet/TRAIN_X/frames/{video_name}/*.jpg
            for chunk in cls.ALL_CHUNKS:
                frames_dir = join(cls.VIDEO_HOME, chunk, "frames", video_name)
                if exists(frames_dir):
                    break
            else:
                log.warning(f"Frames for {video_name} not found in any chunk, skipping video creation.")
                continue
            video_dir = cls._get_video_dir(data_split)
            output_path = join(video_dir, f"{video_name}.mp4")
            if not exists(output_path):
                work_items.append({'frames_dir': frames_dir, 'output_path': output_path, 'fps': fps})
        cls._create_videos(work_items, n_procs)
        cls._check_videos(video_fps)

if __name__ == "__main__":
    '''
    python -m olmo.data.academic_video_track_datasets --dataset ref-yt-vos --split validation
    '''
    import argparse
    logging.basicConfig(level=logging.INFO)


    parser = argparse.ArgumentParser(description="Download and prepare tracking datasets.")
    parser.add_argument("--dataset", required=True, help="Which dataset(s) to process.")
    parser.add_argument("--download", action="store_true",
                        help="Download raw data and create videos. Without this flag, only loads the HF annotations.")
    parser.add_argument("--split", default="train", help="Which data split to process (default: train).")
    parser.add_argument("--tasks", nargs="+", default=["track"], help="Which tasks to process (default: [track]).")
    parser.add_argument("--n_procs", type=int, default=4,
                        help="Number of parallel workers for video creation.")
    args = parser.parse_args()

    def get_dataset_registry():
        return {cls.DATASET_NAME: cls for cls in TrackingDataset.__subclasses__()
            if cls.DATASET_NAME is not None}
    dataset_registry = get_dataset_registry()

    dataset_name = args.dataset
    dataset_cls = dataset_registry.get(dataset_name)
    if dataset_cls is None:
        log.error(f"Unknown dataset: {dataset_name}. Available: {list(dataset_registry.keys())}")
    log.info(f"Processing dataset: {dataset_name}")
    if args.download:
        dataset_cls.download(n_procs=args.n_procs)
    else:
        # Load-only: just load the HF annotations (downloads from HF if not cached locally)
        for task in args.tasks:
            log.info(f"Loading {dataset_name}/{task}/{args.split}...")
            ds = dataset_cls(split=args.split, task=task)
            log.info(f"  Loaded {dataset_name}/{task}/{args.split} successfully.")
            log.info(f"  Got {len(ds)} samples. Iterating through them...")
            for item in tqdm(ds, total=len(ds), desc=f"{dataset_name}/{task}/{args.split}"):
                assert exists(item['video']), f"Video file not found: {item['video']}"
                pass
