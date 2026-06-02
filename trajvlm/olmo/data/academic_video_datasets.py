import json
import logging
import os
import random
import re
import requests
import tarfile
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from os.path import join, exists, isfile
from typing import Literal, Optional
import ast
import unicodedata

import pandas as pd
import datasets
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError


import numpy as np
from tqdm import tqdm
import imageio.v3 as iio
from pathlib import Path
from olmo.io import list_directory, file_exists, read_file, glob as olmo_glob

from olmo.util import resource_path, flatten_lists, split_into_groups

from olmo.data.dataset import DatasetBase, Dataset, DATA_HOME, VIDEO_DATA_HOME, VIDEO_DATA_HOME_RELEASE
from olmo.data.molmo2_datasets import sample_random_clip
from olmo.data.utils import maybe_download_and_unzip, maybe_download_file, maybe_download_and_untar
from olmo.util import set_example_style

from olmo import tokenizer
from olmo.torch_util import get_global_rank

log = logging.getLogger(__name__)


def _create_video_from_frame_range(frames_dir, start_frame, end_frame, fps=3, pad_frames=True):
    """Create an MP4 from numbered frames (e.g., 00001.jpg to 00051.jpg) in a directory."""
    output_path = os.path.join(frames_dir, f"video_{start_frame:05d}_{end_frame:05d}.mp4")
    if exists(output_path):
        return output_path
    frame_files = []
    for i in range(start_frame, end_frame + 1):
        fp = os.path.join(frames_dir, f"{i:05d}.jpg")
        if exists(fp):
            frame_files.append(fp)
    if not frame_files:
        raise ValueError(f"No frames in range {start_frame}-{end_frame} in {frames_dir}")
    frames = []
    for f in frame_files:
        frame = iio.imread(f)
        h, w = frame.shape[:2]
        if pad_frames:
            new_h = ((h + 15) // 16) * 16
            new_w = ((w + 15) // 16) * 16
            if h != new_h or w != new_w:
                if len(frame.shape) == 3:
                    padded = np.zeros((new_h, new_w, frame.shape[2]), dtype=frame.dtype)
                else:
                    padded = np.zeros((new_h, new_w), dtype=frame.dtype)
                padded[:h, :w] = frame
                frame = padded
        frames.append(frame)
    iio.imwrite(output_path, frames, fps=fps, codec='libx264')
    return output_path


def _load_hf_dataset(hf_source, split, local_name):
    """Load an HF dataset, caching locally under VIDEO_DATA_HOME."""
    local_dir = join(VIDEO_DATA_HOME, local_name) if VIDEO_DATA_HOME else None
    if local_dir and exists(local_dir):
        log.info(f"Loading {hf_source} split={split} from {local_dir}")
        return datasets.load_from_disk(local_dir)
    log.info(f"Downloading {hf_source} split={split} from HuggingFace")
    ds = datasets.load_dataset(hf_source, split=split)
    if local_dir:
        log.info(f"Saving to {local_dir}")
        ds.save_to_disk(local_dir)
    return ds


if DATA_HOME:
    VIDEO_HOME = join(DATA_HOME, "videos")
else:
    VIDEO_HOME = None


class Tomato(Dataset):
    SRC = "https://raw.githubusercontent.com/yale-nlp/TOMATO/refs/heads/main/data/"
    HOME = join(VIDEO_DATA_HOME, "TOMATO")
    VIDEOS_URL = "https://drive.google.com/file/d/1-dNt9bZcp6C3RXuGoAO3EBgWkAHg8NWR/view"

    reasoning_type_choices = [
        "count",
        "direction",
        "rotation",
        "shape&trend",
        "velocity&frequency",
        "visual_cues"
    ]
    demonstration_type_choices = [
        "human",
        "object",
        "simulated"
    ]

    @classmethod
    def download(cls, n_procs=None):
        for k in cls.reasoning_type_choices:
            maybe_download_file(cls.SRC + k + ".json", join(cls.HOME, "data", k + ".json"))
        maybe_download_and_unzip(cls.HOME, cls.VIDEOS_URL)

    @staticmethod
    def validate_choices(input_value, all_choices, input_name):
        if input_value == 'ALL':
            return all_choices
        else:
            selected_values = [item.strip() for item in input_value.split(",")]
            invalid_values = [item for item in selected_values if item not in all_choices]
            if invalid_values:
                raise ValueError(f"Invalid {input_name} type(s): {', '.join(invalid_values)}. "
                                 f"Valid choices are: {', '.join(all_choices + ['ALL'])}")
            return selected_values

    def __init__(self, split, reasoning_type="ALL", demonstration_type="ALL"):
        assert split in ["test"]
        self.reasoning_type = reasoning_type
        self.demonstration_type = demonstration_type
        queries = defaultdict(list)
        existing_paths = list()
        reasoning_type = self.validate_choices(self.reasoning_type, self.reasoning_type_choices, "reasoning")
        demonstration_type = self.validate_choices(self.demonstration_type, self.demonstration_type_choices, "demonstration")
        queries = []
        for rt in reasoning_type:
            dataset_path = resource_path(join(self.HOME, f"data/{rt}.json"))
            with open(dataset_path, "r") as f:
                qas = json.load(f)
            for id_, qa in qas.items():
                if qa['demonstration_type'] in demonstration_type:
                    if (qa["demonstration_type"], qa["key"]) == ("object", "0390-01"):
                        # This is a super special snowflake video the causes decord to hang
                        decode_method = "av_noseek"
                    else:
                        decode_method = None
                    queries.append(dict(
                        video=join(self.HOME, "videos", qa["demonstration_type"], qa["key"] + ".mp4"),
                        question=qa['question'],
                        options=qa["options"],
                        answer_idx=qa["answer"],
                        metadata=dict(
                            example_id=id_,
                            key=qa["key"],
                            reasoning_type=rt,
                            demonstration_type=qa["demonstration_type"],
                            motion_type=qa["motion_type"],
                        ),
                        style="video_multiple_choice",
                    ))
        self.data = queries

    def __len__(self):
        return len(self.data)

    def get(self, idx, rng):
        return self.data[idx]


class TemporalBenchQa(DatasetBase):
    HOME = join(VIDEO_DATA_HOME, "TemporalBench")
    # HOME = "/data/chrisc/temporal_bench"

    @classmethod
    def download(cls, n_procs=None):
        if exists(join(cls.HOME, "temporalbench_long_qa.json")):
            return
        snapshot_download(
            repo_id="microsoft/TemporalBench",
            repo_type="dataset",
            local_dir=cls.HOME,
            local_dir_use_symlinks=False
        )
        for filename in os.listdir(cls.HOME):
            if filename.endswith(".zip"):
                log.info(f"Extracting...")
                with zipfile.ZipFile(join(cls.HOME, filename), 'r') as zip_ref:
                    zip_ref.extractall(cls.HOME)
                log.info("Extraction complete!")

    def __init__(self, split, format="original"):
        assert split in ["test"]
        self.format = format
        super().__init__(split)

    def load(self):
        examples = []
        for src in ["long_qa", "short_qa"]:
            file = resource_path(join(self.HOME, f"temporalbench_{src}.json"))
            with open(file, "r") as f:
                data = json.load(f)
            for example in data:
                video = join(self.HOME, example["video_name"])
                metadata = dict(example_id=example["idx"], type=src, video=example["video_name"])
                if self.format == "original":
                    examples.append(dict(
                        video=video,
                        question=example['question'],
                        style="video_short_answer",
                        answer=example["GT"],
                        metadata=metadata
                    ))
                elif self.format == "mc":
                    parts = [x.strip() for x in example['question'].split("\n") if x.strip()]
                    question = parts[0]
                    instructions = parts[-1]
                    assert instructions.startswith("Answer with the option'")
                    options = []
                    answer_idx = None
                    for option_ix, option_part in enumerate(parts[1:-1]):
                        group = re.match("([A-Z]).(.*)", option_part)
                        options.append(group.group(2).strip())
                        if group.group(1) == example["GT"]:
                            assert answer_idx is None, "Multiple option matched the ground truth"
                            answer_idx = option_ix
                    assert answer_idx is not None, "No option matched the ground truth"
                    assert len(options) > 1, "<=1 options"
                    examples.append(dict(
                        video=video,
                        question=question,
                        options=options,
                        answer_idx=answer_idx,
                        style="video_multiple_choice",
                        metadata=metadata
                    ))
                else:
                    raise NotImplementedError(self.format)
        return examples

    def __len__(self):
        return len(self.data)

    def get(self, idx, rng):
        return self.data[idx]


MOTION_BENCH_HOME = join(VIDEO_DATA_HOME, "MotionBench")


def _download_motion_bench(max_workers):
    if exists(join(MOTION_BENCH_HOME, "README.md")):
        return
    log.info(f"Downloading motion bench...")
    snapshot_download(
        repo_id="zai-org/MotionBench",
        repo_type="dataset",
        local_dir=MOTION_BENCH_HOME,
        local_dir_use_symlinks=False,
        max_workers=max_workers
    )


class MotionBench(DatasetBase):
    PATH = "zai-org/MotionBench"
    HOME = join(MOTION_BENCH_HOME, "MotionBench")

    @classmethod
    def download(cls, n_procs=1):
        _download_motion_bench(max_workers=n_procs)

    def __init__(self, split):
        if split == "val":
            split = "validation"
        assert split in ["validation", "test"]
        self.split = split
        super().__init__(split)

    def find_video_path(self, video_filename):
        self_collected_video_path = join(self.HOME, "self-collected", video_filename)
        if file_exists(self_collected_video_path):
            return self_collected_video_path

        public_dataset_video_path = join(self.HOME, "public-dataset", video_filename)
        if file_exists(public_dataset_video_path):
            return public_dataset_video_path
        raise ValueError(f"Missing video {video_filename}")

    def load(self):
        data = []
        with open(resource_path(join(self.HOME, "video_info.meta.jsonl"))) as f:
            data_entries = f.readlines()
        for line in data_entries:
            entry = json.loads(line.strip().strip("\n"))
            video_path = self.find_video_path(entry["video_path"])
            for qa_instance in entry["qa"]:
                answer = qa_instance["answer"]
                if self.split == "validation" and answer == "NA":  # Samples in the test set don't have answers - https://github.com/zai-org/MotionBench/issues/9
                    continue

                data.append(
                    dict(
                        question=qa_instance["question"],
                        answer=answer,
                        video=video_path,
                        metadata=dict(
                            example_id=f"{qa_instance['uid']}_{entry['key']}",
                            qa_uid=qa_instance["uid"],
                            video_info=entry.get("video_info", {}),
                            task_type=entry.get("question_type", ""),
                            video_type=entry.get("video_type", "")
                        ),
                    ))
        return data

    def __len__(self):
        return len(self.data)

    def get(self, idx, rng):
        return dict(**self.data[idx], style="video_eval_multiple_choice")


class MotionBenchCaption(DatasetBase):
    """Video Localized Narratives dataset"""
    video_path = join(MOTION_BENCH_HOME, "MotionBenchCaption-train/videos")
    caption_path = join(MOTION_BENCH_HOME, "MotionBenchCaption-train/train.jsonl")
    qa_path = join(MOTION_BENCH_HOME, "MotionBenchCaption-train/motionbench_qa.parquet")

    @classmethod
    def download(cls, n_procs=1):
        _download_motion_bench(max_workers=n_procs)

    def __init__(self, split, flat: bool = False):
        assert split in ["train"], f"Invalid split: {split}"
        self.flat = flat
        super().__init__(split)

    def load(self):
        data = [json.loads(line) for line in read_file(self.caption_path).split("\n") if line]
        data_list = []
        for i, row in enumerate(data):
            video_path = join(self.video_path, row['video_path'])
            msg = dict(
                text=row['motion_caption'],
                style="video_motion_caption"
            )
            formatted_ex = {
                "video"       : video_path,
                "message_list": [msg]
            }
            data_list.append(formatted_ex)
        return data_list

    def get(self, item, rng):
        return self.data[item]


class QVHighlights(DatasetBase):
    data_path = join(VIDEO_DATA_HOME, "QVHighlights")
    dataset_url = "https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz"
    val_json = "https://raw.githubusercontent.com/jayleicn/moment_detr/refs/heads/main/data/highlight_val_release.jsonl"
    train_json = "https://raw.githubusercontent.com/jayleicn/moment_detr/refs/heads/main/data/highlight_train_release.jsonl"

    @classmethod
    def download(cls, n_procs=1):
        maybe_download_and_untar(cls.data_path, cls.dataset_url)

        if not exists(os.path.join(cls.data_path, "highlight_val_release.jsonl")):
            response = requests.get(cls.val_json, stream=True)
            response.raise_for_status()
            with open(os.path.join(cls.data_path, "highlight_val_release.jsonl"), "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        if not exists(os.path.join(cls.data_path, "highlight_train_release.jsonl")):
            response = requests.get(cls.train_json, stream=True)
            response.raise_for_status()
            with open(os.path.join(cls.data_path, "highlight_train_release.jsonl"), "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

    def __init__(self, split, minimum=0.0):
        self.download()

        if split == "val":
            split = "validation"
        assert split in ["validation", "train"]
        self.max_detected_windows = 5
        self.minimum = minimum
        assert 0 <= self.minimum < 1.0
        super().__init__(split)

    @staticmethod
    def qa_template(data):
        question = f"Here is a query: {data['query'].rstrip('.')}. Where are all the segments for the query? Report the segments in seconds."

        answer_parts = []
        for start, end in data['relevant_windows']:
            answer_parts.append(f"[{start}-{end}]")

        return question, f"Segments: {', '.join(answer_parts)}", data['query'].rstrip('.')

    def load(self):
        data = []
        if self.split == "validation":
            metadata_file_path = resource_path(join(self.data_path, "highlight_val_release.jsonl"))
        else:
            metadata_file_path = resource_path(join(self.data_path, "highlight_train_release.jsonl"))

        with open(metadata_file_path, "r") as f:
            lines = f.readlines()

        skipped = 0
        for line in lines:
            data_info = json.loads(line.strip())
            if len(data_info['relevant_windows']) > self.max_detected_windows:
                # Skipping example with more than self.max_detected_windows relevant windows
                continue

            video_path = join(self.data_path, "videos", f"{data_info['vid']}.mp4")
            if not file_exists(video_path):
                skipped += 1
                continue

            example_id = f"{data_info['qid']}_{data_info['vid']}"

            question, answer, frame_sel_input = self.qa_template(data_info)

            # scale from 0 to 4 to 0 to (1 - self.minimum)
            scaled_scored_id_to_score = {}
            for index, clip_id in enumerate(data_info['relevant_clip_ids']):
                scaled_scored_id_to_score[clip_id] = (sum(data_info['saliency_scores'][index]) / 3.0) / 4.0 * (1 - self.minimum)

            # Outputs scores between self.minimum and 1.0
            scaled_avg_scores = [self.minimum for _ in range(data_info['duration'] // 2)]
            for clip_id in range(len(scaled_avg_scores)):
                if clip_id in scaled_scored_id_to_score:
                    scaled_avg_scores[clip_id] += scaled_scored_id_to_score[clip_id]

            data.append({
                'question': question,
                'answer': answer,
                'video': video_path,
                "metadata": dict(
                    example_id=example_id,
                    video_path=video_path,
                    query=data_info['query'],
                    duration=data_info['duration'],
                    relevant_windows=data_info['relevant_windows'],
                    relevant_clip_ids=data_info['relevant_clip_ids'],
                    saliency_scores=data_info['saliency_scores'],
                    scaled_avg_scores=scaled_avg_scores,
                    frame_sel_input=frame_sel_input
                )
            })

        return data

    def __len__(self):
        return len(self.data)

    def get(self, idx, rng):
        return dict(**self.data[idx], style="video_short_answer")


class LLaVAVideoAcademic(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "LLaVA-Video-178K")
    # Added as a class object because the one on the README on huggingface is wrong for activitynet!
    data_subset_config = {
        "configs": [
            {
                "config_name": "0_30_s_activitynetqa",
                "data_files": [
                    {"split": "open_ended", "path": "0_30_s_activitynetqa/*oe*.json"}
                ]
            },
            {
                "config_name": "0_30_s_perceptiontest",
                "data_files": [
                    {"split": "multi_choice", "path": "0_30_s_perceptiontest/*mc*.json"}
                ]
            },
            {
                "config_name": "0_30_s_nextqa",
                "data_files": [
                    {"split": "open_ended", "path": "0_30_s_nextqa/*oe*.json"},
                    {"split": "multi_choice", "path": "0_30_s_nextqa/*mc*.json"}
                ]
            },
            {
                "config_name": "30_60_s_activitynetqa",
                "data_files": [
                    {"split": "open_ended", "path": "30_60_s_activitynetqa/*oe*.json"}
                ]
            },
            {
                "config_name": "30_60_s_perceptiontest",
                "data_files": [
                    {"split": "multi_choice", "path": "30_60_s_perceptiontest/*mc*.json"}
                ]
            },
            {
                "config_name": "30_60_s_nextqa",
                "data_files": [
                    {"split": "open_ended", "path": "30_60_s_nextqa/*oe*.json"},
                    {"split": "multi_choice", "path": "30_60_s_nextqa/*mc*.json"}
                ]
            },
            {
                "config_name": "1_2_m_activitynetqa",
                "data_files": [
                    {"split": "open_ended", "path": "1_2_m_activitynetqa/*oe*.json"}
                ]
            },
            {
                "config_name": "1_2_m_nextqa",
                "data_files": [
                    {"split": "open_ended", "path": "1_2_m_nextqa/*oe*.json"},
                    {"split": "multi_choice", "path": "1_2_m_nextqa/*mc*.json"}
                ]
            },
            {
                "config_name": "2_3_m_activitynetqa",
                "data_files": [
                    {"split": "open_ended", "path": "2_3_m_activitynetqa/*oe*.json"}
                ]
            },
            {
                "config_name": "2_3_m_nextqa",
                "data_files": [
                    {"split": "open_ended", "path": "2_3_m_nextqa/*oe*.json"},
                    {"split": "multi_choice", "path": "2_3_m_nextqa/*mc*.json"}
                ]
            }
        ]
    }

    @classmethod
    def download(cls, n_procs=1):
        if not exists(os.path.join(cls.data_path, "README.md")):
            log.info(f"Downloading LLaVAVideo178K for its academic subsets...")
            snapshot_download(
                repo_id="lmms-lab/LLaVA-Video-178K",
                repo_type="dataset",
                local_dir=cls.data_path,
                local_dir_use_symlinks=False,
                max_workers=n_procs
            )

            for config_item in cls.data_subset_config.get('configs', []):
                dir_name = config_item['config_name']
                dir_path = join(cls.data_path, dir_name)

                for target_file in os.listdir(dir_path):
                    if "tar.gz" in target_file:
                        target_file_path = join(dir_path, target_file)

                        with tarfile.open(target_file_path, 'r:*') as tar_ref:
                            # Get list of members to extract for progress tracking
                            members = tar_ref.getmembers()

                            # Extract with progress bar
                            with tqdm(total=len(members), desc="  Files", unit="file", leave=False) as extract_pbar:
                                for member in members:
                                    tar_ref.extract(member, dir_path)
                                    extract_pbar.update(1)


    def __init__(self, split, answer_type="multi_choice", flat=False, max_per_video=None):
        assert split == "train"
        assert answer_type in ["multi_choice", "open_ended"]
        self.answer_type = answer_type
        self.flat = flat
        self.max_per_video = max_per_video

        super().__init__(split)

    def load(self):
        data = {}
        data_list_format = []
        for config_item in self.data_subset_config.get('configs', []):
            for data_file in config_item['data_files']:
                question_type = data_file['split']
                if question_type != self.answer_type:
                    continue

                style = "video_" + ("short_answer" if question_type == "open_ended" else "multiple_choice")
                config_path = os.path.join(self.data_path, data_file['path'])
                first_file_data = None
                for file in olmo_glob(config_path):
                    first_file_data = json.loads(read_file(file))
                    break

                for qa_data in first_file_data:
                    video_path = os.path.join(self.data_path, qa_data['data_source'], qa_data['video'])
                    example_id = f"{qa_data['id']}_{qa_data['data_source']}_{qa_data['video']}_{question_type}"

                    conversations = qa_data['conversations']
                    if example_id not in data:
                        messages = []
                    else:
                        messages = data[example_id]['message_list']

                    for conv_idx in range(0, len(conversations), 2):
                        question = conversations[conv_idx]['value']
                        if tokenizer.IMAGE_PROMPT in question:
                            raise ValueError()
                        if question.startswith("<image>\n"):
                            question = question[len("<image>\n"):]

                        answer = conversations[conv_idx + 1]['value']
                        answer = answer.lstrip().strip()

                        msg = dict(answer=answer, question=question, style=style)
                        messages.append(msg)

                    data[example_id] = {
                        'video': video_path,
                        'prefix': data_file['path'],
                        'message_list': messages
                    }

        for example_id, example in data.items():
            data_list_format.append({
                "video": example["video"],
                "metadata": dict(
                    example_id=example_id,
                    prefix=example["prefix"],
                ),
                "message_list": example["message_list"],
            })

        if self.flat:
            data_list_format = flatten_lists(
                [dict(ex, message_list=[message]) for message in ex["message_list"]]
                for ex in data_list_format
            )
        elif self.max_per_video:
            flat = []
            for ex in tqdm(data_list_format):
                for msg in split_into_groups(ex["message_list"], self.max_per_video):
                    flat.append(dict(ex, message_list=msg))
            logging.info(f"Split {len(data_list_format)} in {len(flat)} examples")
            data_list_format = flat

        return data_list_format

    def __len__(self):
        return len(self.data)

    def get(self, idx, rng):
        return self.data[idx]


class MVBench(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "MVBench")
    data_list = {
        "Action Sequence": ("action_sequence.json", "star/Charades_v1_480/"),
        "Action Prediction": ("action_prediction.json", "star/Charades_v1_480/"),
        "Action Antonym": ("action_antonym.json", "ssv2_video"),
        "Fine-grained Action": ("fine_grained_action.json", "Moments_in_Time_Raw/videos/"),
        "Unexpected Action": ("unexpected_action.json", "FunQA_test/test/"),
        "Object Existence": ("object_existence.json", "clevrer/video_validation/"),
        "Object Interaction": ("object_interaction.json", "star/Charades_v1_480/"),
        "Object Shuffle": ("object_shuffle.json", "perception/videos/"),
        "Moving Direction": ("moving_direction.json", "clevrer/video_validation/"),
        "Action Localization": ("action_localization.json", "sta/sta_video/"),
        "Scene Transition": ("scene_transition.json", "scene_qa/video/"),
        "Action Count": ("action_count.json", "perception/videos/"),
        "Moving Count": ("moving_count.json", "clevrer/video_validation/"),
        "Moving Attribute": ("moving_attribute.json", "clevrer/video_validation/"),
        "State Change": ("state_change.json", "perception/videos/"),
        "Fine-grained Pose": ("fine_grained_pose.json", "nturgbd/"),
        "Character Order": ("character_order.json", "perception/videos/"),
        "Egocentric Navigation": ("egocentric_navigation.json", "vlnqa/"),
        "Episodic Reasoning": ("episodic_reasoning.json", "tvqa/frames_fps3_hq/"),
        "Counterfactual Inference": ("counterfactual_inference.json", "clevrer/video_validation/"),
    }
    data_types_with_bound = {"Action Sequence", "Action Prediction", "Object Interaction", "Action Localization", "Episodic Reasoning"}


    @staticmethod
    def create_video_from_frames(frames_dir, fps, pad_frames=True):
        """
        Creates a video file from a sequence of frames in a directory.

        Args:
            frames_dir (str): Directory containing the frames
            fps (int): Frames per second for the output video
            pad_frames (bool): Whether to pad frames to make them divisible by 16

        Returns:
            str: Path to the created video file
        """
        output_path = os.path.join(frames_dir, f"video.mp4")
        if file_exists(output_path):
            return output_path
        from natsort import natsorted
        frame_files = natsorted(Path(frames_dir).iterdir())
        if not frame_files:
            raise ValueError(f"No frames found in {frames_dir}")

        frames = []
        for f in frame_files:
            frame = iio.imread(f)
            h, w = frame.shape[:2]
            if pad_frames:
                # Calculate padded dimensions (divisible by 16)
                new_h = ((h + 15) // 16) * 16
                new_w = ((w + 15) // 16) * 16

                if h != new_h or w != new_w:
                    # Pad with black pixels
                    if len(frame.shape) == 3:
                        padded = np.zeros((new_h, new_w, frame.shape[2]), dtype=frame.dtype)
                    else:
                        padded = np.zeros((new_h, new_w), dtype=frame.dtype)
                    padded[:h, :w] = frame
                    frame = padded

            frames.append(frame)

        iio.imwrite(output_path, frames, fps=fps, codec='libx264')
        return output_path

    @classmethod
    def download(cls, n_procs=1):
        if not exists(join(cls.data_path, "README.md")):
            log.info(f"Downloading MVBench...")
            snapshot_download(
                repo_id="OpenGVLab/MVBench",
                repo_type="dataset",
                local_dir=cls.data_path,
                local_dir_use_symlinks=False,
                max_workers=n_procs
            )
            video_home = join(cls.data_path, "video")
            for zip_file in list_directory(video_home):
                if zip_file.endswith(".zip"):
                    log.info(f"Extracting {join(video_home, zip_file)}...")
                    with zipfile.ZipFile(join(video_home, zip_file), 'r') as zip_ref:
                        zip_ref.extractall(video_home)
                    log.info("Extraction complete!")
                    os.remove(join(video_home, zip_file))

            # Files in data0613 need to be moved to the other folders
            for file in list_directory(join(video_home, "data0613"), recurse=True, include_dirs=False):
                shutil.move(file, file.replace("/data0613/", "/"))

        # Episodic Reasoning needs the videos to be built. Check if the videos exist and create if needed
        _, video_home = cls.data_list["Episodic Reasoning"]
        video_home = join(cls.data_path, "video", video_home)
        video_exist_count = 0
        for video_dir in os.listdir(video_home):
            if os.path.exists(os.path.join(video_home, video_dir, f"video.mp4")):
                video_exist_count += 1
        if video_exist_count < len(os.listdir(video_home)):
            log.info(f"Creating Episodic Reasoning videos...")
            for video_dir in tqdm(os.listdir(video_home)):
                cls.create_video_from_frames(os.path.join(video_home, video_dir), fps=3)

        with open(join(cls.data_path, "video", "MVBench_videos_ntu.txt")) as f:
            for line in f.readlines():
                file = join(cls.data_path, "video", "nturgbd", line.strip())
                if not exists(file):
                    raise FileNotFoundError(f"File {file} needs to be manually downloaded, See: https://huggingface.co/datasets/OpenGVLab/MVBench")

    def __init__(self, split, sample=None):
        assert split in ["validation", "val"]
        if split == "validation":
            split = "val"
        super().__init__(split, sample)

    def qa_template(self, data):
        question = data['question']
        answer = data['answer']
        options = "\n".join(f"{chr(ord('A') + idx)}. {c}" for idx, c in enumerate(data['candidates']))
        answer_idx = data['candidates'].index(answer)
        answer = f"{chr(ord('A') + answer_idx)}."
        question = "\n".join(
            [
                question,
                options,
                "Please respond with only the letter of the correct answer.",
            ]
        )
        return question, answer

    def load(self):
        data = []
        for k, (json_src, video_home) in self.data_list.items():
            json_data = json.loads(read_file(os.path.join(self.data_path, "json", json_src)))
            video_home = join(self.data_path, "video", video_home)

            for qa_idx, qa_data in enumerate(json_data):
                if k == "Episodic Reasoning":
                    video_path = join(video_home, qa_data['video'], "video.mp4")
                else:
                    video_path = join(video_home, qa_data['video'])

                example_id = f"{k}_{qa_idx}"
                question, answer = self.qa_template(qa_data)
                data.append({
                    'question': question,
                    'answer': answer,
                    'video': video_path,
                    "metadata": dict(
                        example_id=example_id,
                        video_path=video_path,
                        task_type=k,
                        prefix=video_home,
                        clip_start_time=qa_data['start'] if 'start' in qa_data else None,
                        clip_end_time=qa_data['end'] if 'end' in qa_data else None,
                    )
                })
        return data

    def __len__(self):
        return len(self.data)

    def get(self, idx, rng):
        return dict(**self.data[idx], style="video_eval_multiple_choice")


class VideoEvalProMC(Dataset):
    home = join(VIDEO_DATA_HOME, "VideoEvalPro")
    parquet_path = join(home, "data", "test-00000-of-00001.parquet")
    video_path = join(home, "videos")

    @classmethod
    def download(cls, n_procs=1):
        if not exists(cls.parquet_path):
            log.info(f"Downloading video-eval-pro...")
            snapshot_download(
                repo_id="TIGER-Lab/VideoEval-Pro",
                repo_type="dataset",
                local_dir=cls.home,
                local_dir_use_symlinks=False,
                max_workers=n_procs
            )

            log.info(f"Merging tar file...")
            # Easiest to this with bash so we can avoid materializing a merged file
            subprocess.run('cat videos_part_*.tar.gz | tar -xzf -', shell=True, check=True, cwd=cls.video_path)
            subprocess.run('rm videos_part_*.tar.gz', shell=True, check=True, cwd=cls.video_path)

    def __init__(self, split):
        assert split == "test"
        self.data = pd.read_parquet(self.parquet_path)
        if exists(self.video_path):
            self.video_path = join(self.video_path, "videos_filtered")
        else:
            self.video_path = join(self.home, "videos_filtered")

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        row = self.data.iloc[item]
        metadata = {k: row[k] for k in [
            'answer_text', 'source', "qa_subtype", "qa_type", "options"]}
        metadata['example_id'] = f"{item}_{row['video']}"
        options = list(row['options'])
        question = row['question']
        formatted_question = "\n".join([question, "\n".join(options),
                                        "Please respond with only the letter of the correct answer."])
        return dict(
            question=formatted_question,
            answer=row["answer"],
            style="video_eval_multiple_choice",
            video=join(self.video_path, row["video"]),
            metadata=metadata
        )


class MLVU(DatasetBase):
    home = os.path.join(VIDEO_DATA_HOME, "MVLU")
    data_path = os.path.join(home, "MLVU")
    test_home = os.path.join(VIDEO_DATA_HOME, "MLVU_Test")
    val_mc_tasks = [
        "plotQA",
        "needle",
        "ego",
        "count",
        "order",
        "anomaly_reco",
        "topic_reasoning"
    ]
    val_gen_tasks = ["sub_scene", "summary"]

    @classmethod
    def download(cls, n_procs=1):
        try:
            if not exists(cls.home):
                snapshot_download(
                    repo_id="MLVU/MVLU",
                    repo_type="dataset",
                    local_dir=cls.home,
                    local_dir_use_symlinks=False,
                    token=os.environ.get("HF_ACCESS_TOKEN")
                )
            if not exists(cls.test_home):
                snapshot_download(
                    repo_id="MLVU/MLVU_Test",
                    repo_type="dataset",
                    local_dir=cls.test_home,
                    local_dir_use_symlinks=False,
                    token=os.environ.get("HF_ACCESS_TOKEN")
                )
                video_home = join(cls.test_home, "MLVU_Test")
                log.info(f"untaring videos...")
                subprocess.run('cat test_video.tar.gz.part-* | tar -xzf -', shell=True, check=True, cwd=video_home)
                subprocess.run('rm test_video.tar.gz.part-*', shell=True, check=True, cwd=video_home)
                log.info(f"Done")
        except GatedRepoError as e:
            e.add_note("MLVU requires accepting a licensing agreement, go to https://huggingface.co/datasets/MLVU/MLVU_Test and "
                       "https://huggingface.co/datasets/MLVU/MLVU, accept the agreement, and then either authenticate with HF or "
                       "set HF_ACCESS_TOKEN to your access token before running this script")
            raise e

    def __init__(self, split, task, use_resize=False):
        assert split in ["validation", "test"]
        assert task in ["multiple-choice", "generation"]
        self.task = task
        self.use_resize = use_resize
        super().__init__(split)

    def mc_qa_template(self, data):
        """lmms-eval uses the MVBench's template, but llava-video uses the different one, so just follow the PerceptionTest's template"""
        option_text = "\n".join(f"{chr(ord('A') + idx)}. {opt}" for idx, opt in enumerate(data['candidates']))
        answer_idx = data['candidates'].index(data['answer'])
        answer = f"{chr(ord('A') + answer_idx)}"
        question = "\n".join(
            [
                data["question"],
                option_text,
                "Answer with the option's letter from the given choices directly.",
            ]
        )
        return question, answer

    def load(self):
        task = self.task
        data = []
        if self.split == "test":
            assert self.task in "multiple-choice"
            gt_video_path = join(self.test_home, "MLVU_Test", "video")
            if not exists(gt_video_path):
                gt_video_path = join(self.test_home, "video")
                assert exists(gt_video_path)
            gt_path = os.path.join(self.test_home, "test-ground-truth", "test_mcq_gt.json")
            with open(gt_path, "r") as f:
                gt_data = json.load(f)

            data = []
            for qa_data in gt_data:
                video_path = os.path.join(gt_video_path, qa_data['video'])
                question, answer = self.mc_qa_template(qa_data)
                example = {
                    "question": question,
                    "answer": answer,
                    "video": video_path,
                    "style": "video_eval_multiple_choice",
                    "metadata": dict(
                        question_id=str(qa_data['question_id']),
                        video_id=qa_data['video'],
                        duration=qa_data['duration'],
                    )
                }
                data.append(example)

        elif task == "multiple-choice":
            question_id = 0
            for idx, task_type in enumerate(self.val_mc_tasks, 1):
                name = f"{idx}_{task_type}"
                json_data = json.loads(read_file(os.path.join(self.data_path, "json", f"{name}.json")))
                for qa_data in json_data:
                    video_path = os.path.join(self.data_path, "video", name, f"{qa_data['video']}")
                    question, answer = self.mc_qa_template(qa_data)
                    example = {
                        "question": question,
                        "answer": answer,
                        "video": video_path,
                        "style": "video_eval_multiple_choice",
                        "metadata": dict(
                            question_id=str(question_id),
                            video_id=qa_data['video'],
                            task_type=task_type,
                            duration=qa_data['duration'],
                        )
                    }
                    data.append(example)
                    question_id += 1
        else:
            question_id = 0
            for idx, task_type in enumerate(self.val_gen_tasks, 1):
                name = f"{len(self.val_mc_tasks) + idx}_{task_type}"
                json_data = json.loads(read_file(os.path.join(self.data_path, "json", f"{name}.json")))
                for qa_data in json_data:
                    video_path = os.path.join(self.data_path, "video", name, f"{qa_data['video']}")
                    example = {
                        "question": qa_data['question'],
                        "answer": qa_data['answer'],
                        "video": video_path,
                        "style": "demo",
                        "metadata": dict(
                            question_id=str(question_id),
                            question=qa_data['question'],
                            answer=qa_data['answer'],
                            video_id=qa_data['video'],
                            task_type=task_type,
                            duration=qa_data['duration'],
                        )
                    }
                    if "scoring_points" in qa_data:
                        example["metadata"]["scoring_points"] = qa_data["scoring_points"]
                    data.append(example)
                    question_id += 1
        data.sort(key=lambda x: x["metadata"]["duration"])
        return data

    def get(self, idx, rng):
        return self.data[idx]


class LongVideoBench(DatasetBase):
    home = join(VIDEO_DATA_HOME, "LongVideoBench")

    @staticmethod
    def time_to_seconds(time_str: str) -> float:
        h, m, s = time_str.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    @classmethod
    def download(cls, n_procs=1):
        try:
            if not exists(cls.home):
                snapshot_download(
                    repo_id="longvideobench/LongVideoBench",
                    repo_type="dataset",
                    local_dir=cls.home,
                    local_dir_use_symlinks=False,
                    token=os.environ.get("HF_ACCESS_TOKEN")
                )
                log.info(f"untaring...")
                subprocess.run('cat videos.tar.part.* | tar -xf -', shell=True, check=True, cwd=cls.home)
                subprocess.run('rm videos.tar.part.*', shell=True, check=True, cwd=cls.home)
                subprocess.run('tar -xf subtitles.tar', shell=True, check=True, cwd=cls.home)
                subprocess.run('rm subtitles.tar', shell=True, check=True, cwd=cls.home)
                log.info(f"Done")
        except GatedRepoError as e:
            e.add_note("LongVideoBench requires accepting a licensing agreement, go to https://huggingface.co/datasets/longvideobench/LongVideoBench and "
                       "accept the agreement, and then either authenticate with HF or "
                       "set HF_ACCESS_TOKEN to your access token before running this script")
            raise e

    duration_groups = ["15", "60", "600", "3600"]

    def __init__(self, split, allow_subtitle=True, difficulty="all", duration_group="all", with_subtitle=False):
        assert split in ["validation", "test"]
        assert difficulty in ["easy", "medium", "hard", "all"]
        assert duration_group in (self.duration_groups + ["all"])

        self.difficulty = difficulty
        self.duration_group = duration_group
        self.allow_subtitle = allow_subtitle
        self.with_subtitle = with_subtitle
        if with_subtitle:
            assert self.allow_subtitle is True, "with_subtitle at True, want to include subtitle questions"
            self.style = "video_eval_multiple_choice_w_subtitle"
        else:
            self.style = "video_eval_multiple_choice"
        super().__init__(split)

    def qa_template(self, qa_data):
        option_text = "\n".join(f"{chr(ord('A') + idx)}. {opt}" for idx, opt in enumerate(qa_data['candidates']))
        answer = f"{chr(ord('A') + qa_data['correct_choice'])}" if "correct_choice" in qa_data else None
        question = "\n".join(
            [
                qa_data["question"],
                option_text,
                "Answer with the option's letter from the given choices directly.",
            ]
        )
        return question, answer

    def load(self):
        if self.split == "validation":
            json_data = json.loads(read_file(os.path.join(self.home, "lvb_val.json")))
        else:
            json_data = json.loads(read_file(os.path.join(self.home, "lvb_test_wo_gt.json")))

        data = []
        for row_idx, qa_data in enumerate(json_data):
            question, answer = self.qa_template(qa_data)
            if not self.allow_subtitle and "subtitle" in question:
                continue
            if self.duration_group != "all" and qa_data["duration_group"] != self.duration_group:
                continue

            video_path = os.path.join(self.home, "videos", qa_data["video_path"])
            example = {
                "question": question,
                "answer": answer,
                "video": video_path,
                "metadata": dict(
                    question_id=qa_data["id"],
                    video_id=qa_data["video_id"],
                    level=qa_data["level"],
                    options=qa_data["candidates"],
                    question_category=qa_data["question_category"],
                    duration_group=qa_data["duration_group"],
                )
            }
            if self.split == "validation":
                example_id = f"{qa_data['video_id']}_{qa_data['id']}_{row_idx}"
                example['metadata']['example_id'] = example_id

            if self.with_subtitle:
                subtitle = json.loads(read_file(os.path.join(self.home, "subtitles", qa_data["subtitle_path"])))
                subtitle_dict = {}
                starting_timestamp_for_subtitles = qa_data['starting_timestamp_for_subtitles']
                ending_timestamp_for_subtitles = starting_timestamp_for_subtitles + qa_data['duration']
                for entry in subtitle:
                    if "timestamp" in entry:
                        sub_start, sub_end = entry["timestamp"]
                        if not isinstance(sub_end, float):
                            sub_end = qa_data['duration']
                        text = entry["text"]
                    else:
                        sub_start, sub_end = float(self.time_to_seconds(entry['start'])), float(self.time_to_seconds(entry['end']))
                        text = entry["line"]

                    if sub_end < starting_timestamp_for_subtitles or sub_start > ending_timestamp_for_subtitles:
                        continue
                    sub_start -= starting_timestamp_for_subtitles
                    sub_end -= starting_timestamp_for_subtitles
                    if sub_start < 0:
                        sub_start = 0.0
                    subtitle_dict[(sub_start, sub_end)] = text
                example['subtitle'] = subtitle_dict
            data.append(example)
        return data

    def get(self, idx, rng):
        return dict(**self.data[idx], style=self.style)


class LVBench(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "LVBench")

    @classmethod
    def download(cls, n_procs=8):
        if not exists(cls.data_path):
            raise ValueError("LVBench needs to be manually downloaded, follow the instructions in: https://huggingface.co/datasets/zai-org/LVBench")

    @staticmethod
    def parse_mcq_lines(text: str):
        """
        Assumes:
          - line 1 = question
          - lines 2..n = options (one per line)
        Accepts options like "(A) foo", "A) foo", "A. foo", or just "foo".
        Returns: (question, [(label, option_text), ...])
        """
        lines = [ln.strip() for ln in text.replace("\r\n", "\n").split("\n") if ln.strip()]
        if not lines:
            return "", []

        question = lines[0]
        raw_options = lines[1:]

        options = []
        for i, opt in enumerate(raw_options):
            # Try to pull a leading label; otherwise fall back to 1-based index
            # Patterns: (A) foo | A) foo | A. foo
            label = None
            if len(opt) >= 3:
                if opt[0] == "(" and ")" in opt[:4]:
                    label = opt[1:opt.index(")")]
                    opt = opt[opt.index(")") + 1:].strip()
                elif opt[1:3] in (") ", ").") and opt[0].isalnum():
                    label = opt[0]
                    opt = opt[3:].strip()
                elif len(opt) > 2 and opt[1] == "." and opt[0].isalnum():
                    label = opt[0]
                    opt = opt[2:].strip()

            if label is None:
                label = str(i + 1)  # fallback label

            options.append((label, opt))

        return question, options

    def __init__(self):
        super().__init__("test")

    def qa_template(self, data):
        question, options = self.parse_mcq_lines(data['question'])
        options = "\n".join(f"{idx}. {c}" for idx, c in options)
        answer = f"{data['answer']}."
        question = "\n".join(
            [
                question,
                options,
                "Please respond with only the letter of the correct answer.",
            ]
        )
        return question, answer

    def load(self):
        data = []

        with open(join(self.data_path, "video_info.meta.jsonl"), "r") as f:
            video_info = [json.loads(line) for line in f.readlines()]

        for info in video_info:
            for qa in info["qa"]:
                question, answer = self.qa_template(qa)
                data.append(dict(
                    video=join(self.data_path, info["key"] + ".mp4"),
                    question=question,
                    metadata=dict(
                        answer=answer,
                        type=info["type"],
                        uid=qa["uid"],
                        qtype=qa["question_type"],
                        time_reference=qa["time_reference"],
                    ),
                    style="video_eval_multiple_choice",
                ))

        return data

    def get(self, item, rng):
        return self.data[item]


class VideoMME(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "Video-MME")
    ORIG_REPO = "lmms-lab/Video-MME"
    HF_REPO = "allenai/Molmo2-VideoMME"
    duration = ["short", "medium", "long"]

    @classmethod
    def download(cls, n_procs=8):
        # Skip if fully set up
        if exists(join(cls.data_path, "videomme")) \
                and exists(join(cls.data_path, "data")) \
                and exists(join(cls.data_path, "subtitles.json")):
            return
        # 1. Download parquet + video zips from original repo
        if not exists(join(cls.data_path, "videomme")):
            log.info("Downloading Video-MME from lmms-lab...")
            snapshot_download(
                repo_id=cls.ORIG_REPO,
                repo_type="dataset",
                local_dir=cls.data_path,
                local_dir_use_symlinks=False,
                max_workers=n_procs,
            )
            for file in os.listdir(cls.data_path):
                if file.endswith(".zip"):
                    log.info(f"Extracting {file}")
                    with zipfile.ZipFile(join(cls.data_path, file), 'r') as zf:
                        zf.extractall(cls.data_path)
                    os.remove(join(cls.data_path, file))
        # 2. Download custom subtitles.json
        if not exists(join(cls.data_path, "subtitles.json")):
            hf_hub_download(
                repo_id=cls.HF_REPO, repo_type="dataset",
                filename="subtitles.json", local_dir=cls.data_path,
            )

    def __init__(self, split, duration="all", difficulty="all", with_subtitle=False):
        assert split in ["validation"]
        assert duration in (self.duration + ["all"])
        assert difficulty in ["easy", "medium", "hard", "all"]
        self.difficulty = difficulty
        self.target_duration = duration
        self.with_subtitle = with_subtitle
        super().__init__(split)

    def question_template(self, question, options):
        prompt = "Select the best answer to the following multiple-choice question based on the video."
        prompt += " Respond with only the letter (A, B, C, or D) of the correct option."
        question = "\n".join(
            [
                prompt,
                question,
                "\n".join(options),
                "The best answer is:"
            ]
        )
        return question

    def load(self):
        parquet_path = os.path.join(self.data_path, "videomme", "test-00000-of-00001.parquet")
        df = pd.read_parquet(resource_path(parquet_path))
        if self.target_duration != "all":
            df = df[df["duartion"] == self.target_duration]
        data = []
        video_dir = os.path.join(self.data_path, "data")
        subtitles = json.loads(read_file(resource_path(os.path.join(self.data_path, "subtitles.json"))))
        for idx, row in df.iterrows():
            question = self.question_template(row["question"], row["options"])
            video_path = os.path.join(video_dir, row["videoID"] + ".mp4")
            example_id = f"{row['question_id']}_{row['video_id']}_{idx}"
            example = {
                "question": question,
                "answer": row["answer"],
                "video": video_path,
                "style": "video_eval_multiple_choice",
                "metadata": dict(
                    example_id=example_id,
                    video_id=row["video_id"],
                    question_id=row["question_id"],
                    duration=row["duration"],
                    domain=row["domain"],
                    sub_category=row["sub_category"],
                    task_type=row["task_type"],
                )
            }
            if self.with_subtitle and row["videoID"] in subtitles:
                subtitle = {}
                for i, entry in subtitles[row["videoID"]].items():
                    subtitle[(entry["start"], entry["end"])] = entry["text"]
                example["subtitle"] = subtitle
                example["style"] = "video_eval_multiple_choice_w_subtitle"
            data.append(example)

        return data

    def get(self, idx, rng):
        return self.data[idx]


# Maps subset -> video root relative to VIDEO_DATA_HOME
_ACADEMIC_POINT_VIDEO_ROOTS = {
    "lvvis": "LV-VIS/{split}/videos-2fps",
    "ovis": "OVIS/videos-2fps",
    "burst": "TAO-Amodal/{split}/videos-2fps",
    "refdavis17": "Ref-DAVIS17/{split}/videos-2fps",
    "mevis": "mevis/MeViS_release/{split}/videos-2fps",
    "refyoutube": "Ref-YT-VOS/{split}/videos-2fps",
}


def _make_video_path(subset, split, video_name, dataset=None):
    """Reconstruct full video path from HF row fields."""
    root = join(VIDEO_DATA_HOME, _ACADEMIC_POINT_VIDEO_ROOTS[subset].format(split=split))
    if subset == "burst":
        return join(root, dataset or "", f"{video_name}.mp4")
    elif subset in ("refdavis17", "mevis", "refyoutube"):
        return join(root, video_name)
    else:
        return join(root, f"{video_name}.mp4")


class AcademicVideoPoint(DatasetBase):
    """Loads allenai/molmo2-academic-video-points from HuggingFace.

    Produces the same output format as the original AcademicVideoPoint
    """
    HF_SOURCE = "allenai/molmo2-academic-video-points"
    LOCAL_NAME = "molmo2-academic-video-points"

    @classmethod
    def download(cls, n_procs=None):
        _load_hf_dataset(cls.HF_SOURCE, "train", cls.LOCAL_NAME + "-train")
        _load_hf_dataset(cls.HF_SOURCE, "val", cls.LOCAL_NAME + "-val")

    def __init__(
        self,
        split: Literal["train", "val"] ="train",
        subset: str = "all",
        mode: str = "point_count",
        flat: bool = False,
        max_points: int = None,
        point_sort_by: str = "xy",
        max_seconds: int = None,
        fps: int = 2,
        use_clips_from_metadata: bool = False,
    ):
        assert split in ("train", "val")
        self.subset = subset
        self.mode = mode
        self.flat = flat
        self.max_points = max_points
        self.point_sort_by = point_sort_by
        self.max_seconds = max_seconds
        self.fps = fps
        self.use_clips_from_metadata = use_clips_from_metadata
        if self.use_clips_from_metadata:
            self.max_seconds = 63  # Clips from metadata are pre-computed for max 63 seconds
        super().__init__(split)

    def load(self):
        import math

        ds = _load_hf_dataset(self.HF_SOURCE, self.split, f"{self.LOCAL_NAME}-{self.split}")

        # Filter by subset
        if self.subset != "all":
            ds = ds.filter(lambda x: x == self.subset, input_columns="subset")

        video2msgs = {}
        formatted_data = []
        for row in ds:
            subset = row["subset"]
            video_path = _make_video_path(
                subset, self.split, row["video_name"],
                dataset=row.get("dataset"),
            )

            total_points = row["count"]
            if self.max_points is not None and total_points > self.max_points:
                continue

            timestamps = row["timestamps"]
            ann_start = min(timestamps)
            ann_end = max(timestamps)
            ann_duration = ann_end - ann_start
            if self.max_seconds is not None and ann_duration > self.max_seconds:
                continue

            label = row["category"].lower()
            example_id = video_path.replace(VIDEO_DATA_HOME + "/", "") + "_" + label

            msg = {
                "label": label,
                "answer": str(total_points),
                "count": total_points,
                "example_id": example_id,
            }

            if self.split == "val":
                # Fix eval questions instead of using different templates in data_formatter
                msg["question"] = f'How many "{label}" are there in the video?'

            # Clip times
            video_duration = row["video_duration"]
            if self.use_clips_from_metadata:
                msg["clip_start_time"] = row["clip_start_time"]
                msg["clip_end_time"] = row["clip_end_time"]
            elif self.max_seconds is not None:
                step = 1.0 / self.fps
                if video_duration <= self.max_seconds:
                    msg["clip_start_time"] = 0.0
                    msg["clip_end_time"] = video_duration
                else:
                    rand_start, rand_end = sample_random_clip(
                        video_duration=video_duration,
                        start_time=ann_start,
                        end_time=ann_end,
                        min_seconds=step,
                        max_seconds=self.max_seconds,
                        timestamp_step=step,
                        seed=42,
                    )
                    msg["clip_start_time"] = rand_start
                    msg["clip_end_time"] = rand_end

            if "clip_start_time" in msg:
                assert msg['clip_start_time'] >= 0, msg['clip_start_time']
                assert msg['clip_end_time'] <= video_duration, (msg['clip_end_time'], video_duration)
                assert msg['clip_start_time'] < msg['clip_end_time'], (msg['clip_start_time'], msg['clip_end_time'])
                assert msg['clip_end_time'] - msg['clip_start_time'] <= self.max_seconds, (msg['clip_end_time'], msg['clip_start_time'], self.max_seconds)
                assert msg['clip_start_time'] <= ann_start, (msg['clip_start_time'], ann_start)
                assert msg['clip_end_time'] >= ann_end, (msg['clip_end_time'], ann_end)

            # Sort by timestamp, adjust timestamps, sort points
            sorted_timestamps = []
            sorted_points = []
            for i, ts in sorted(enumerate(timestamps), key=lambda x: x[1]):
                if "clip_start_time" in msg:
                    ts = ts - msg["clip_start_time"]
                step = 1.0 / self.fps
                assert math.isclose(ts % step, 0, abs_tol=1e-6), f"Timestamp {ts} is not a multiple of step {1.0 / self.fps}"
                ts = math.floor(ts / step) * step 
                sorted_timestamps.append(ts)

                points = list(row["points"][i])
                if self.point_sort_by == "xy":
                    points = sorted(points, key=lambda p: (p["x"], p["y"]))
                elif self.point_sort_by == "yx":
                    points = sorted(points, key=lambda p: (p["y"], p["x"]))
                sorted_points.append(points)

            msg["points"] = sorted_points
            msg["timestamps"] = sorted_timestamps

            group_key = (video_path, msg.get("clip_start_time"), msg.get("clip_end_time"))
            video2msgs[group_key] = video2msgs.get(group_key, []) + [msg]
            if self.flat or self.max_seconds is not None:
                metadata = {
                    "points": sorted_points,
                    "timestamps": sorted_timestamps,
                    "count": msg["count"],
                    "subset": "object",
                }
                if self.max_seconds is not None:
                    metadata["clip_start_time"] = msg["clip_start_time"]
                    metadata["clip_end_time"] = msg["clip_end_time"]
                msg["video"] = video_path
                msg["metadata"] = metadata
                formatted_data.append(msg)

        if not self.flat and self.max_seconds is None:
            for (video_path, clip_start, clip_end), msgs in video2msgs.items():
                item = {
                    "video": video_path,
                    "message_list": msgs,
                }
                if clip_start is not None:
                    item["clip_start_time"] = clip_start
                if clip_end is not None:
                    item["clip_end_time"] = clip_end
                formatted_data.append(item)
        return formatted_data

    def __len__(self):
        return len(self.data)

    def get(self, idx, rng):
        if isinstance(self.mode, str):
            style = self.mode
        else:
            style = rng.choice(self.mode)
        return set_example_style(self.data[idx], f"video_{style}")


class EgoSchema(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "egoschema")

    @classmethod
    def download(cls, n_procs=8):
        if exists(join(cls.data_path, "videos")):
            return
        snapshot_download(
            repo_id="lmms-lab/egoschema",
            repo_type="dataset",
            local_dir=cls.data_path,
            local_dir_use_symlinks=False
        )
        log.info(f"Unzipping videos")
        for file in os.listdir(cls.data_path):
            if file.endswith(".zip"):
                log.info(f"Extracting {file}")
                with zipfile.ZipFile(join(cls.data_path, file), 'r') as zip_ref:
                    zip_ref.extractall(cls.data_path)
                os.remove(join(cls.data_path, file))

    def __init__(self, split):
        assert split in ["validation", "test"]
        super().__init__(split)

    def question_template(self, question, options):
        question = "\n".join(
            [
                question,
                "\n".join(options),
                "Answer with the option's letter from the given choices directly.",
            ]
        )
        return question

    def load(self):
        subset_tag = "Subset" if self.split == "validation" else "MC"
        parquet_path = os.path.join(self.data_path, subset_tag, "test-00000-of-00001.parquet")
        df = pd.read_parquet(resource_path(parquet_path))
        data = []
        for idx, row in df.iterrows():
            video_path = os.path.join(self.data_path, "videos", row["video_idx"] + ".mp4")
            question = self.question_template(row["question"], row["option"])

            if row["answer"] is not None:
                answer = "abcdefg".upper()[int(row["answer"])]
            else:
                answer = None
            example_id = f"{row['question_idx']}_{row['video_idx']}_{idx}"

            example = {
                "question": question,
                "answer": answer,
                "video": video_path,
                "metadata": dict(
                    example_id=example_id,
                    question_id=row["question_idx"],
                    video_id=row["video_idx"],
                    options=list(row["option"]),
                    answer_idx=row["answer"],
                )
            }
            data.append(example)
        return data

    def get(self, idx, rng):
        return dict(**self.data[idx], style="video_eval_multiple_choice")


class ActivityNet(DatasetBase):
    """ActivityNet Video dataset (Captioning / ActivityNetQA)"""
    home = join(VIDEO_DATA_HOME, "ActivityNet")
    video_path = join(home, "all-videos")

    # Number of videos available on disk per split as of our download.
    # Out of 19,994 annotated videos, 3,876 were unavailable (deleted from YouTube, etc.).
    # These counts may differ if the dataset is re-downloaded at a later date.
    AVAILABLE_VIDEOS = {"training": 8059, "validation": 3981, "testing": 4078}

    templates = [
        "What activity is being performed?",
        "What activity is occurring?",
        "What activity does the video depict?"
    ]

    @classmethod
    def download(cls, n_procs=1):
        if not exists(cls.video_path):
            raise FileNotFoundError(
                f"ActivityNet videos not found at: {cls.video_path}\n\n"
                "Please download the ActivityNet videos manually from YouTube\n"
                "using the video IDs in `activity_net.v1-3.min.json` and place\n"
                "them as a flat directory of .mp4 files under:\n\n"
                f"  {cls.video_path}/\n\n"
                "Expected layout:\n"
                f"  {cls.video_path}/<video_id>.mp4\n"
            )

        # Cache which videos are missing so load() can skip them
        # without per-file existence checks during distributed training.
        missing_videos_f = join(cls.home, "missing_videos.json")
        if not exists(missing_videos_f):
            log.info("Scanning for missing ActivityNet videos...")
            caption_data = json.loads(read_file(join(cls.home, "activity_net.v1-3.min.json")))
            missing = []
            for vid in caption_data["database"]:
                if not exists(join(cls.video_path, f"{vid}.mp4")):
                    missing.append(vid)
            with open(missing_videos_f, "w") as f:
                json.dump(missing, f)
            log.info(
                f"Found {len(missing)} missing ActivityNet videos "
                f"out of {len(caption_data['database'])}"
            )

    def __init__(
        self,
        split,
        flat: bool = False,
        max_per_video: Optional[int] = None,
        task: Literal["captioning", "qa", "all"] = "all",
        qa_format: bool = False,
    ):
        assert split in ["train", "validation"], f"Invalid split: {split}"
        self.flat = flat
        if split == "validation":
            split = "val"
        self.max_per_video = max_per_video
        self.task = task
        self.qa_format = qa_format
        super().__init__(split)

    def _load_missing_videos(self):
        """Load cached set of missing video IDs."""
        missing_videos_f = join(self.home, "missing_videos.json")
        if file_exists(missing_videos_f):
            return set(json.loads(read_file(missing_videos_f)))
        return set()

    def load(self):
        caption_json_path = join(self.home, "activity_net.v1-3.min.json")
        q_json_path = join(self.home, f"{self.split}_q.json")
        a_json_path = join(self.home, f"{self.split}_a.json")

        if self.task in ["qa", "all"]:
            for f in [q_json_path, a_json_path]:
                if not file_exists(f):
                    raise FileNotFoundError(
                        f"ActivityNet QA file not found: {f}\n"
                        "Please download the QA annotation files (train_q.json, train_a.json, "
                        "val_q.json, val_a.json) from:\n"
                        "  https://github.com/MILVLG/activitynet-qa/tree/master/dataset\n"
                        f"and place them in: {self.home}"
                    )

        missing_videos = self._load_missing_videos()

        data_list = []
        video2msgs = {}

        if self.task in ["captioning", "all"]:
            caption_data = json.loads(read_file(caption_json_path))
            for vid, anns in caption_data["database"].items():
                abs_video_path = join(self.video_path, vid, f"{vid}.mp4") # try nested path
                if not exists(abs_video_path):
                    abs_video_path = join(self.video_path, f"{vid}.mp4")
                if vid in missing_videos:
                    continue

                for ann in anns["annotations"]:
                    start, end = ann["segment"]
                    if pd.isna(start) or pd.isna(end):
                        continue
                    if end <= start:
                        continue

                    if self.qa_format:
                        import random
                        question = random.choice(self.templates)
                        msg = dict(
                            question=question,
                            answer=ann["label"],
                            style="activitynet_label",
                        )
                    else:
                        msg = dict(
                            text=ann["label"],
                            style="activitynet_short_caption",
                        )

                    if self.flat:
                        data_list.append({
                            "video": abs_video_path,
                            "meta": {
                                "clip_start_time": start,
                                "clip_end_time": end,
                            },
                            "message_list": [msg],
                        })
                    else:
                        key = (abs_video_path, start, end)
                        video2msgs.setdefault(key, [])
                        if msg not in video2msgs[key]:
                            video2msgs[key].append(msg)

        if self.task in ["qa", "all"]:
            q_df = pd.read_json(resource_path(q_json_path))
            a_df = pd.read_json(resource_path(a_json_path))
            qa_df = pd.merge(q_df, a_df, on="question_id", how="inner")

            for video_name, question, answer in zip(
                qa_df["video_name"], qa_df["question"], qa_df["answer"]
            ):
                abs_video_path = join(self.video_path, video_name, f"{video_name}.mp4")  # try nested path
                if not exists(abs_video_path):
                    abs_video_path = join(self.video_path, f"{video_name}.mp4")
                if video_name in missing_videos:
                    continue

                if not question.endswith("?"):
                    question = question.strip() + "?"

                msg = dict(
                    question=question,
                    answer=answer,
                    style="video_short_answer",
                )

                key = (abs_video_path, None, None)
                video2msgs.setdefault(key, [])
                video2msgs[key].append(msg)

                if self.flat:
                    data_list.append({
                        "video": abs_video_path,
                        "message_list": [msg],
                    })

        if not self.flat:
            for (video, start, end), msgs in video2msgs.items():
                if len(msgs) == 0:
                    continue
                meta = None
                if start is not None and end is not None:
                    meta = {
                        "clip_start_time": start,
                        "clip_end_time": end,
                    }
                groups = (
                    split_into_groups(msgs, self.max_per_video)
                    if self.max_per_video
                    else [msgs]
                )
                for group in groups:
                    formatted_ex = {
                        "video": video,
                        "message_list": group,
                    }
                    if meta is not None:
                        formatted_ex["metadata"] = meta
                    data_list.append(formatted_ex)

        return data_list

    def get(self, item, rng):
        return self.data[item]


class COIN(DatasetBase):
    """COIN dataset"""
    home = join(VIDEO_DATA_HOME, "coin")
    video_path = join(home, "videos")

    action_templates = [
        "What action is being performed?",
        "What action is occurring?",
        "What activity does the video depict?"
    ]

    segments_path = join(home, "video_segments")

    @classmethod
    def download(cls, n_procs=1):
        if not exists(cls.video_path):
            raise ValueError("COIN videos needs to be manually downloaded, follow the instructions in: https://github.com/coin-dataset/annotations")

        # Download corrupt clips metadata
        corrupt_file = join(cls.home, "coin_corrupt.parquet")
        if not exists(corrupt_file):
            log.info("Downloading COIN corrupt clips metadata...")
            maybe_download_file(
                "https://storage.googleapis.com/molmo-datasets/coin_corrupt.parquet",
                corrupt_file
            )

        # Scan video directory to cache file extensions so load() never
        # needs to do per-file existence checks.
        extensions_file = join(cls.home, "file_extensions.json")
        if not exists(extensions_file):
            log.info("Scanning COIN video extensions...")
            extensions = {}
            coin_json = join(cls.home, "COIN.json")
            data = json.load(open(resource_path(coin_json)))
            for video_id in data["database"]:
                found = None
                for ext in [".mkv", ".mp4", ".webm"]:
                    vp = join(cls.video_path, f"{video_id}{ext}")
                    if exists(vp):
                        found = ext
                        break
                extensions[video_id] = found
            with open(extensions_file, "w") as f:
                json.dump(extensions, f)
            n_missing = sum(1 for v in extensions.values() if v is None)
            log.info(
                f"Cached COIN extensions: {n_missing} missing out of {len(extensions)}"
            )
        else:
            with open(extensions_file) as f:
                extensions = json.load(f)

        # Extract video segments from full-length source videos using ffmpeg.
        # Uses scripts/extract_coin_clips.py
        import subprocess
        extract_script = join(os.path.dirname(__file__), "..", "..", "scripts", "extract_coin_clips.py")
        extract_script = os.path.abspath(extract_script)
        if not exists(extract_script):
            raise FileNotFoundError(
                f"Clip extraction script not found at: {extract_script}\n"
                "Expected at: scripts/extract_coin_clips.py"
            )
        cmd = [sys.executable, extract_script, "--coin-dir", cls.home, "--workers", str(n_procs)]
        log.info(f"Running COIN clip extraction: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            log.warning("COIN clip extraction finished with errors (see output above).")

    def __init__(
        self,
        split,
        flat: bool = False,
        task: Literal["caption_clip", "all"] = "caption_clip",
        max_per_video: Optional[int] = None,
        qa_format: bool = False,
    ):
        assert split in ["train", "test"], f"Invalid split: {split}"
        self.flat = flat
        self.task = task
        self.max_per_video = max_per_video
        self.qa_format = qa_format
        super().__init__(split)

    def _load_extensions(self):
        extensions_file = join(self.home, "file_extensions.json")
        if file_exists(extensions_file):
            return json.loads(read_file(extensions_file))
        return {}

    def load(self):
        json_path = join(self.home, "COIN.json")
        data = json.load(open(resource_path(json_path)))

        corrupt_path = join(self.home, "coin_corrupt.parquet")
        corrupt = pd.read_parquet(resource_path(corrupt_path))
        corrupt = corrupt.groupby("video_id")
        corrupt_video_ids = corrupt.groups.keys()

        extensions = self._load_extensions()
        precomputed = len(extensions) > 0

        data_list = []
        video2msgs = defaultdict(list)
        skipped = 0

        for video_id, v in data["database"].items():
            if self.split not in v["subset"]:
                continue

            if precomputed:
                ext = extensions.get(video_id)
                if ext is None:
                    skipped += 1
                    continue
                abs_video_path = join(self.video_path, f"{video_id}{ext}")
            else:
                # Fallback: scan filesystem (slow on remote FS).
                for ext in [".mkv", ".mp4", ".webm"]:
                    abs_video_path = join(self.video_path, f"{video_id}{ext}")
                    if file_exists(abs_video_path):
                        break
                else:
                    skipped += 1
                    continue

            for ann in v["annotation"]:
                segment = ann.get("segment", None)
                if segment is None:
                    continue
                if segment[1] <= segment[0]:
                    continue
                start, end = segment

                if self.task in ["caption_clip", "all"]:
                    if self.qa_format:
                        import random
                        question = random.choice(self.action_templates)
                        msg = dict(
                            question=question,
                            answer=ann["label"],
                            style="coin_label",
                        )
                    else:
                        msg = dict(
                            text=ann["label"],
                            style="coin_label",
                        )

                    # Skip corrupt segments
                    if video_id in corrupt_video_ids:
                        meta_check = {
                            "clip_start_time": segment[0],
                            "clip_end_time": segment[1],
                        }
                        corrupt_group = corrupt.get_group(video_id)
                        if meta_check in corrupt_group["metadata"].values:
                            continue

                    if self.flat:
                        data_list.append({
                            "video": abs_video_path,
                            "meta": {
                                "clip_start_time": start,
                                "clip_end_time": end,
                            },
                            "message_list": [msg],
                        })
                    else:
                        video2msgs[(abs_video_path, start, end)].append(msg)

        log.warning(
            f"Skipped {skipped}/{skipped + len(data['database'])} missing COIN videos."
        )

        if not self.flat:
            for (video, start, end), msgs in video2msgs.items():
                if len(msgs) == 0:
                    continue
                groups = (
                    split_into_groups(msgs, self.max_per_video)
                    if self.max_per_video
                    else [msgs]
                )
                for group in groups:
                    data_list.append({
                        "video": video,
                        "metadata": {
                            "clip_start_time": start,
                            "clip_end_time": end,
                        },
                        "message_list": group,
                    })

        return data_list

    def get(self, item, rng):
        return self.data[item]


class EpicKitchens(DatasetBase):
    """Epic Kitchens 100 dataset for short video clip captioning.

    Epic Kitchens is a large-scale dataset of egocentric videos in kitchen environments
    with temporal action annotations. Each clip contains narrations describing kitchen
    activities like "open door", "take cup", etc.

    When use_extracted_clips=True (default), it looks for pre-extracted clips in clips_path.
    When use_extracted_clips=False, it uses the original full videos with clip timing metadata.
    """
    home = join(VIDEO_DATA_HOME, "epic-kitchens")
    clips_path = join(VIDEO_DATA_HOME, "epic-kitchens-clips")
    annotations_path = join(home, "epic-kitchens-100-annotations")

    corrupt_clips = [
        "P01_102_229.910_234.300.mp4", "P01_102_90.240_93.490.mp4",
        "P01_104_65.170_92.450.mp4", "P02_118_722.700_723.550.mp4",
    ]

    templates = [
        "What action is being performed?",
        "What action is occurring?",
        "What activity does the video depict?"
    ]

    @classmethod
    def download(cls, n_procs=1):
        if not exists(cls.home):
            raise FileNotFoundError(
                f"Epic Kitchens data not found at: {cls.home}\n\n"
                "Please download the Epic Kitchens 100 dataset and annotations.\n"
                "See: https://epic-kitchens.github.io/2026"
            )

        # Auto-generate EPIC_100_train_with_missing.csv if it doesn't exist.
        # The official dataset only ships EPIC_100_train.csv; we add a boolean
        # 'missing' column by checking which clips actually exist on disk.
        train_csv = join(cls.annotations_path, "EPIC_100_train.csv")
        train_with_missing_csv = join(cls.annotations_path, "EPIC_100_train_with_missing.csv")
        if not exists(train_with_missing_csv) and exists(train_csv):
            log.info("Scanning for missing Epic Kitchens clips to build train_with_missing.csv ...")
            df = pd.read_csv(train_csv)
            missing_flags = []
            for row in df.itertuples(index=False):
                start_time = cls._parse_timestamp(row.start_timestamp)
                stop_time = cls._parse_timestamp(row.stop_timestamp)
                # video_id already includes participant_id (e.g., "P02_104")
                video_fname = f"{row.video_id}_{start_time:.3f}_{stop_time:.3f}.mp4"
                clip_path = join(cls.clips_path, row.participant_id, "videos", video_fname)
                missing_flags.append(not exists(clip_path))
            df["missing"] = missing_flags
            n_missing = sum(missing_flags)
            df.to_csv(train_with_missing_csv, index=True)
            log.info(
                f"Wrote {train_with_missing_csv}: "
                f"{n_missing} missing out of {len(df)} clips"
            )

        if not exists(cls.clips_path):
            # Call extraction script to generate clips
            extract_script = join(os.path.dirname(__file__), "..", "..", "scripts", "extract_epic_kitchens_clips.py")
            extract_script = os.path.abspath(extract_script)
            if not exists(extract_script):
                raise FileNotFoundError(f"Epic Kitchens extraction script not found: {extract_script}")
            annotation_csv = join(cls.annotations_path, "EPIC_100_train_with_missing.csv")
            cmd = [
                sys.executable, extract_script,
                "--annotation-path", annotation_csv,
                "--videos-root", cls.home,
                "--clips-root", cls.clips_path,
                "--workers", str(n_procs),
            ]
            log.info(f"Running Epic Kitchens clip extraction: {' '.join(cmd)}")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                raise RuntimeError(f"Epic Kitchens clip extraction failed with exit code {result.returncode}")


    def __init__(
        self,
        split,
        flat: bool = False,
        max_per_video: Optional[int] = None,
        use_extracted_clips: bool = True,
        qa_format: bool = False,
    ):
        assert split in ["train", "validation"], f"Invalid split: {split}"
        self.flat = flat
        self.max_per_video = max_per_video
        self.use_extracted_clips = use_extracted_clips
        self.qa_format = qa_format
        super().__init__(split)

    @staticmethod
    def _parse_timestamp(timestamp_str):
        """Parse timestamp string like '00:00:01.089' to seconds."""
        parts = timestamp_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds

    def _get_video_path(self, participant_id, video_id, start_time, stop_time):
        """Get the appropriate video path, preferring extracted clips when available."""
        if self.use_extracted_clips:
            # video_id already includes participant_id (e.g., "P02_104")
            video_fname = f"{video_id}_{start_time:.3f}_{stop_time:.3f}.mp4"
            if video_fname in self.corrupt_clips:
                return None, None
            clip_path = join(self.clips_path, participant_id, "videos", video_fname)
            return clip_path, None

        original_path = join(self.home, participant_id, "videos", f"{video_id}.mp4")
        metadata = {
            "clip_start_time": start_time,
            "clip_end_time": stop_time,
        }
        return original_path, metadata

    def load(self):
        # train split has _with_missing.csv with a 'missing' column;
        # validation split only has the plain CSV without that column.
        if self.split == "train":
            annotation_file = join(self.annotations_path, "EPIC_100_train_with_missing.csv")
        else:
            annotation_file = join(self.annotations_path, "EPIC_100_validation.csv")

        if not file_exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

        df = pd.read_csv(resource_path(annotation_file))

        data_list = []
        video2msgs = defaultdict(list)
        skipped = 0

        for row in df.itertuples(index=False):
            # Skip missing clips (only present in the train CSV)
            if hasattr(row, "missing") and row.missing:
                skipped += 1
                continue

            start_time = self._parse_timestamp(row.start_timestamp)
            stop_time = self._parse_timestamp(row.stop_timestamp)

            video_path, clip_metadata = self._get_video_path(
                row.participant_id, row.video_id, start_time, stop_time
            )
            if video_path is None:
                skipped += 1
                continue

            narration = row.narration

            if self.qa_format:
                import random
                msg = dict(
                    question=random.choice(self.templates),
                    answer=narration,
                    style="epic_kitchens_label",
                )
            else:
                msg = dict(
                    text=narration,
                    style="epic_kitchens_short_caption",
                )

            if self.flat:
                formatted_ex = {
                    "video": video_path,
                    "message_list": [msg],
                }
                if clip_metadata is not None:
                    formatted_ex["metadata"] = clip_metadata
                data_list.append(formatted_ex)
            else:
                key = (
                    video_path,
                    start_time if clip_metadata else None,
                    stop_time if clip_metadata else None,
                )
                video2msgs[key].append(msg)

        if skipped > 0:
            log.warning(
                f"Skipped {skipped} epic-kitchens clips "
                f"(missing or corrupt) in {self.split} split."
            )

        if not self.flat:
            for (video, start, stop), msgs in video2msgs.items():
                if len(msgs) == 0:
                    continue
                clip_metadata = None
                if start is not None and stop is not None:
                    clip_metadata = {
                        "clip_start_time": start,
                        "clip_end_time": stop,
                    }
                groups = (
                    split_into_groups(msgs, self.max_per_video)
                    if self.max_per_video
                    else [msgs]
                )
                for group in groups:
                    formatted_ex = {
                        "video": video,
                        "message_list": group,
                    }
                    if clip_metadata is not None:
                        formatted_ex["metadata"] = clip_metadata
                    data_list.append(formatted_ex)

        return data_list

    def get(self, item, rng):
        return self.data[item]


class MomentsInTime(DatasetBase):
    """Moments in Time dataset for short video clip action recognition.

    Moments in Time is a large-scale dataset of 3-second video clips labeled with actions.
    Each clip shows a single action or activity (e.g., "running", "opening", "smiling").
    """
    home = join(VIDEO_DATA_HOME, "moments_in_time")
    video_dir = join(home, "Moments_in_Time_Raw")

    templates = [
        "What action is being performed?",
        "What action is occurring?",
        "What activity does the video depict?"
    ]

    @classmethod
    def download(cls, n_procs=1):
        if not exists(cls.video_dir):
            raise FileNotFoundError(
                f"Moments in Time data not found at: {cls.video_dir}\n\n"
                "Please download the Moments in Time dataset manually.\n"
                "See: http://moments.csail.mit.edu/"
            )

        # NOTE: We create symlinks with sequential names (0000000.mp4, 0000001.mp4, etc.) because
        # the original video filenames from Moments in Time contain special characters and long names
        # that can cause problems with GCS (Google Cloud Storage) data loaders when training on GCP.
        # The sequential naming ensures compatibility with cloud storage systems. Feel free to not rename
        # if you are only training locally and want to preserve original filenames

        # Generate {split}_with_missing_renamed.csv from original {split}.csv if it doesn't exist
        for split in ["train", "validation"]:
            original_csv = join(cls.video_dir, f"{split}.csv")
            renamed_csv = join(cls.video_dir, f"{split}_with_missing_renamed.csv")

            if not exists(original_csv):
                continue

            if exists(renamed_csv):
                continue

            log.info(f"Generating {split}_with_missing_renamed.csv from {split}.csv...")
            df = pd.read_csv(original_csv)
            original_video_dir = join(cls.video_dir, f"{split}-videos")

            # Create transformed_video_path with sequential numbering
            df['transformed_video_path'] = df['label'] + '/' + pd.Series(range(len(df))).apply(lambda x: f'{x:07d}.mp4')

            # Check if the original videos exist in train-videos directory
            df['missing'] = df['video_path'].apply(
                lambda path: not exists(join(original_video_dir, path))
            )

            df.to_csv(renamed_csv, index=False)

            n_missing = df['missing'].sum()
            log.info(
                f"Wrote {renamed_csv}: "
                f"{n_missing} missing out of {len(df)} videos"
            )

            # Create renamed video directory structure if it doesn't exist
            renamed_video_dir = join(cls.video_dir, f"{split}-videos-renamed")
            if not exists(renamed_video_dir):
                log.info(f"Creating renamed video directory structure at {renamed_video_dir}...")
                original_video_dir = join(cls.video_dir, f"{split}-videos")

                # Create symlinks for all videos with sequential names
                created_count = 0
                for idx, row in enumerate(df.itertuples(index=False)):
                    label = row.label
                    original_path = join(original_video_dir, row.video_path)
                    renamed_path = join(renamed_video_dir, row.transformed_video_path)

                    # Create label subdirectory if it doesn't exist
                    label_dir = join(renamed_video_dir, label)
                    os.makedirs(label_dir, exist_ok=True)

                    # Create symlink if original video exists and symlink doesn't exist yet
                    if exists(original_path) and not exists(renamed_path):
                        os.symlink(original_path, renamed_path)
                        created_count += 1

                log.info(f"Created {created_count} video symlinks in {renamed_video_dir}")

    def __init__(
        self,
        split,
        flat: bool = False,
        max_per_video: Optional[int] = None,
        qa_format: bool = False
    ):
        assert split in ["train", "validation"], f"Invalid split: {split}"
        self.video_path = join(self.video_dir, f"{split}-videos-renamed")
        self.flat = flat
        self.max_per_video = max_per_video
        self.qa_format = qa_format
        super().__init__(split)

    def load(self):
        csv_path = join(self.video_dir, f"{self.split}_with_missing_renamed.csv")
        if not file_exists(csv_path):
            raise FileNotFoundError(
                f"CSV file not found: {csv_path}\n"
                "Please ensure the Moments in Time dataset is properly set up with the CSV file."
            )

        data = pd.read_csv(resource_path(csv_path))
        data_list = []
        video2msgs = {}
        skipped = 0

        # Use `itertuples` since this dataset is huge and itertuples is much faster than `iterrows`
        for row in data.itertuples(index=False):
            label = row.label
            abs_video_path = join(self.video_path, row.transformed_video_path)
            if row.missing:
                skipped += 1
                continue

            if '+' in label:
                label = label.replace('+', ' ')
                if label.startswith("child ") or label.startswith("adult "):
                    continue

            if self.qa_format:
                import random
                msg = dict(
                    question=random.choice(self.templates),
                    answer=label,
                    style="moments_in_time_label"
                )
            else:
                msg = dict(
                    text=label,
                    style="moments_in_time_label"
                )

            if self.flat:
                formatted_ex = {
                    "video": abs_video_path,
                    "message_list": [msg]
                }
                data_list.append(formatted_ex)
            else:
                video2msgs[abs_video_path] = video2msgs.get(abs_video_path, [])
                video2msgs[abs_video_path].append(msg)

        if skipped > 0:
            log.warning(f"Skipped {skipped}/{len(data)} missing Moments in Time videos.")

        if not self.flat:
            for video, msgs in video2msgs.items():
                groups = (
                    split_into_groups(msgs, self.max_per_video)
                    if self.max_per_video
                    else [msgs]
                )
                for group in groups:
                    formatted_ex = {
                        "video": video,
                        "message_list": group,
                    }
                    data_list.append(formatted_ex)

        return data_list

    def get(self, item, rng):
        return self.data[item]


class Kinetics710(DatasetBase):
    """Kinetics710 dataset combining K600 and K700 videos with K710 labels"""
    root_path = join(VIDEO_DATA_HOME, "kinetics")

    templates = [
        "What action is being performed?",
        "What is the person doing?",
        "What action is the person taking?",
        "What activity does the video depict?"
    ]

    @classmethod
    def download(cls, n_procs=1):
        """Dataset must be manually downloaded from Kinetics website.

        NOTE: Kinetics710 combines K600 and K700 videos, mapping them to K710 label space.
        The download() method verifies that necessary files exist and auto-generates
        kinetics_existing_videos.json by scanning the video directories.

        Required structure:
        - kinetics/kinetics600/{train,val}/{label}/{video_id}.mp4
        - kinetics/kinetics700/{train,val}/{label}/{video_id}.mp4
        - kinetics/kinetics710/k710_label_map.txt
        - kinetics/kinetics710/map_k600.json
        - kinetics/kinetics710/map_k700.json
        - kinetics/kinetics710/k600_{train,val}_list_videos.txt
        - kinetics/kinetics710/k700_{train,val}_list_videos.txt
        - kinetics/kinetics_corrupt.parquet (auto-downloaded from GCS)
        - kinetics/kinetics_existing_videos.json (auto-generated)
        """
        # Download corrupt videos metadata
        corrupt_file = join(cls.root_path, "kinetics_corrupt.parquet")
        if not exists(corrupt_file):
            log.info("Downloading Kinetics corrupt videos metadata...")
            maybe_download_file(
                "https://storage.googleapis.com/molmo-datasets/kinetics_corrupt.parquet",
                corrupt_file
            )

        required_files = [
            join(cls.root_path, "kinetics710", "k710_label_map.txt"),
            join(cls.root_path, "kinetics710", "map_k600.json"),
            join(cls.root_path, "kinetics710", "map_k700.json"),
            join(cls.root_path, "kinetics710", "k600_train_list_videos.txt"),
            join(cls.root_path, "kinetics710", "k600_val_list_videos.txt"),
            join(cls.root_path, "kinetics710", "k700_train_list_videos.txt"),
            join(cls.root_path, "kinetics710", "k700_val_list_videos.txt"),
        ]

        for file_path in required_files:
            if not exists(resource_path(file_path)):
                raise FileNotFoundError(
                    f"Required file not found: {file_path}. "
                    f"Please follow the instructions on https://github.com/cvdfoundation/kinetics-dataset "
                    f"and download Kinetics710 dataset manually."
                )

        # Auto-generate kinetics_existing_videos.json by scanning video directories
        existing_videos_file = join(cls.root_path, "kinetics_existing_videos.json")
        if not exists(existing_videos_file):
            log.info("Scanning Kinetics video directories to generate kinetics_existing_videos.json...")
            existing_videos = []

            # Scan both kinetics600 and kinetics700 directories
            for dataset in ["kinetics600", "kinetics700"]:
                for split in ["train", "val"]:
                    split_dir = join(cls.root_path, dataset, split)
                    if not exists(split_dir):
                        log.warning(f"Directory not found: {split_dir}, skipping...")
                        continue

                    # Walk through label directories
                    for label in os.listdir(split_dir):
                        label_dir = join(split_dir, label)
                        if not os.path.isdir(label_dir):
                            continue

                        # Find all .mp4 files
                        for video_file in os.listdir(label_dir):
                            if video_file.endswith('.mp4'):
                                # Store relative path from root_path
                                relative_path = join(dataset, split, label, video_file)
                                existing_videos.append(relative_path)

            # Write to JSON file
            with open(existing_videos_file, 'w') as f:
                json.dump(existing_videos, f, indent=2)

            log.info(f"Generated {existing_videos_file} with {len(existing_videos)} videos")

    def __init__(
        self,
        split,
        flat: bool = False,
        max_per_video: Optional[int] = None,
        qa_format: bool = False
    ):
        assert split in ["train", "val"], f"Invalid split: {split}"
        self.split = split
        self.flat = flat
        self.max_per_video = max_per_video
        self.qa_format = qa_format
        super().__init__(split)

    def _load_mappings(self):
        """Load all label mappings in one place"""
        # Load k710 labels
        k710_labels_path = resource_path(join(self.root_path, "kinetics710", "k710_label_map.txt"))
        k710_labels = Path(k710_labels_path).read_text().splitlines()

        # Load mapping files
        k700_map_path = resource_path(join(self.root_path, "kinetics710", "map_k700.json"))
        k600_map_path = resource_path(join(self.root_path, "kinetics710", "map_k600.json"))
        k700_map = json.load(open(k700_map_path))
        k600_map = json.load(open(k600_map_path))

        return {
            'k710_labels': {i: line for i, line in enumerate(k710_labels)},
            'k600_to_k710': {i: int(v) for i, v in enumerate(k600_map)},
            'k700_to_k710': {i: int(v) for i, v in enumerate(k700_map)},
        }

    def _load_video_list(self, dataset_name):
        """Load video list for a specific dataset (k600 or k700)"""
        file_path = join(self.root_path, "kinetics710", f"{dataset_name}_{self.split}_list_videos.txt")
        lines = Path(resource_path(file_path)).read_text().splitlines()

        return {
            line.strip().split(" ")[0]: int(line.strip().split(" ")[1])
            for line in lines
        }

    def load(self):
        # Load all mappings in a more organized way
        mappings = self._load_mappings()

        # Load video lists and create mappings to k710 labels
        k600_video_to_k710_label = {}
        k700_video_to_k710_label = {}

        # Process k600 videos
        k600_list = self._load_video_list("k600")
        for video_path, label_id in k600_list.items():
            video_id = video_path.split("/")[-1]
            k600_video_to_k710_label[video_id] = mappings['k710_labels'][label_id]

        # Process k700 videos
        k700_list = self._load_video_list("k700")
        for video_path, label_id in k700_list.items():
            video_id = video_path.split("/")[-1]
            k700_video_to_k710_label[video_id] = mappings['k710_labels'][label_id]

        corrupt = pd.read_parquet(resource_path(join(self.root_path, "kinetics_corrupt.parquet")))
        corrupt_videos = set(corrupt['video'].apply(lambda x: join(self.root_path, x)).values)

        with open(resource_path(join(self.root_path, "kinetics_existing_videos.json")), "r") as f:
            existing_videos = json.load(f)
        existing_videos = set([join(self.root_path, el) for el in existing_videos])

        # Process videos and create dataset
        data_list = []
        video2msgs = {}
        skipped = 0

        for video_id, label in list(k700_video_to_k710_label.items()):
            # Get the K710 label text
            abs_video_path = join(self.root_path, "kinetics700", self.split, label, video_id)
            if abs_video_path not in existing_videos:
                skipped += 1
                continue
            if abs_video_path in corrupt_videos:
                skipped += 1
                continue

            if self.qa_format:
                msg = dict(
                    question=random.choice(self.templates),
                    answer=label,
                    style="kinetics_label"
                )
            else:
                msg = dict(
                    text=label,
                    style="kinetics_label"
                )

            if self.flat:
                formatted_ex = {
                    "video": abs_video_path,
                    "message_list": [msg]
                }
                data_list.append(formatted_ex)
            else:
                video2msgs[abs_video_path] = video2msgs.get(abs_video_path, [])
                video2msgs[abs_video_path].append(msg)

        for video_id, label in list(k600_video_to_k710_label.items()):
            abs_video_path = join(self.root_path, "kinetics600", self.split, label.replace(" ", "_"), video_id)
            if abs_video_path in corrupt_videos:
                skipped += 1
                continue
            if abs_video_path not in existing_videos:
                skipped += 1
                continue

            if self.qa_format:
                msg = dict(
                    question=random.choice(self.templates),
                    answer=label,
                    style="kinetics_label"
                )
            else:
                msg = dict(
                    text=label,
                    style="kinetics_label"
                )

            if self.flat:
                formatted_ex = {
                    "video": abs_video_path,
                    "message_list": [msg]
                }
                data_list.append(formatted_ex)
            else:
                video2msgs[abs_video_path] = video2msgs.get(abs_video_path, [])
                video2msgs[abs_video_path].append(msg)

        log.warning(f"Skipped {skipped}/{len(k700_video_to_k710_label) + len(k600_video_to_k710_label)} missing Kinetics710 videos.")

        if not self.flat:
            for video, msgs in video2msgs.items():
                if len(msgs) == 0:
                    continue
                if self.max_per_video:
                    for msg_group in split_into_groups(msgs, self.max_per_video):
                        formatted_ex = {
                            "video": video,
                            "message_list": msg_group,
                        }
                        data_list.append(formatted_ex)
                else:
                    formatted_ex = {
                        "video": video,
                        "message_list": msgs,
                    }
                    data_list.append(formatted_ex)

        return data_list

    def get(self, item, rng):
        return self.data[item]


class Youcook2(DatasetBase):
    """YouCook2 dataset for video clip captioning and action localization."""
    home = join(VIDEO_DATA_HOME, "youcook2")
    video_path = join(home, "videos")

    # Hardcoded corrupt segments (only 2 known corrupt segments)
    corrupt_segments = [
        {"video_id": "UB1_MNpdvgs", "clip_start_time": 595, "clip_end_time": 613},
        {"video_id": "UB1_MNpdvgs", "clip_start_time": 624, "clip_end_time": 635},
    ]

    templates = [
        "What action is being performed?",
        "What action is occurring?",
        "What activity does the video depict?"
    ]

    @classmethod
    def download(cls, n_procs=1):
        """Dataset must be manually downloaded.

        Required structure:
        - youcook2/youcookii_annotations_trainval.json
        - youcook2/videos/{video_id}.{mp4,mkv,webm}

        Videos should be placed as flat files under youcook2/videos/
        (e.g., youcook2/videos/video001.mp4, youcook2/videos/video002.mkv, etc.)

        Note: Corrupt segments are hardcoded in the class (only 2 known corrupt segments).
        """
        if not exists(cls.home):
            raise FileNotFoundError(
                f"Youcook2 data not found at: {cls.home}\n\n"
                "Please download the Youcook2 dataset manually and place videos under:\n"
                f"  {cls.video_path}/\n\n"
                "See: http://youcook2.eecs.umich.edu/download"
            )

        annotations_file = join(cls.home, "youcookii_annotations_trainval.json")
        if not exists(resource_path(annotations_file)):
            raise FileNotFoundError(
                f"Annotations file not found: {annotations_file}\n"
                "Please download youcookii_annotations_trainval.json"
            )

        # Scan video directory to cache file extensions so load() never
        # needs to do per-file existence checks.
        extensions_file = join(cls.home, "video_extensions.json")
        if not exists(extensions_file):
            log.info("Scanning Youcook2 video extensions...")
            extensions = {}
            youcook_json = join(cls.home, "youcookii_annotations_trainval.json")
            data = pd.read_json(resource_path(youcook_json))
            for video_id in data["database"]:
                found = None
                for ext in [".mkv", ".mp4", ".webm"]:
                    vp = join(cls.video_path, f"{video_id}{ext}")
                    if exists(vp):
                        found = ext
                        break
                extensions[video_id] = found
            with open(extensions_file, "w") as f:
                json.dump(extensions, f)
            n_missing = sum(1 for v in extensions.values() if v is None)
            log.info(
                f"Cached Youcook2 extensions: {n_missing} missing out of {len(extensions)}"
            )

    def __init__(
        self,
        split,
        flat: bool = False,
        task: Literal["caption_clip", "caption_start_end", "all"] = "caption_clip",
        max_per_video: Optional[int] = None,
        qa_format: bool = False,
    ):
        """
        Args:
            split: Dataset split ("train", "validation", or "test")
            flat: If True, each example contains only a single message
            task: Task type - "caption_clip", "caption_start_end", or "all"
            max_per_video: Max messages per video (only used when flat=False)
            qa_format: If True, use QA format with question/answer
        """
        assert split in ["train", "validation", "test"], f"Invalid split: {split}"
        self.flat = flat
        self.task = task
        self.max_per_video = max_per_video
        self.qa_format = qa_format
        super().__init__(split)

    def _load_extensions(self):
        """Load cached extension mapping."""
        extensions_file = join(self.home, "video_extensions.json")
        if file_exists(extensions_file):
            return json.loads(read_file(extensions_file))
        return {}

    def load(self):
        json_path = join(self.home, "youcookii_annotations_trainval.json")
        data = pd.read_json(resource_path(json_path))

        # Build set of corrupt segments for fast lookup
        corrupt_set = {
            (seg["video_id"], seg["clip_start_time"], seg["clip_end_time"])
            for seg in self.corrupt_segments
        }

        extensions = self._load_extensions()
        precomputed = len(extensions) > 0

        data_list = []
        video2msgs = defaultdict(list)
        skipped = 0

        for video_id, v in data['database'].items():
            if self.split not in v['subset']:
                continue

            if precomputed:
                ext = extensions.get(video_id)
                if ext is None:
                    skipped += 1
                    continue
                abs_video_path = join(self.video_path, f"{video_id}{ext}")
            else:
                # Fallback: scan filesystem (slow on remote FS).
                for ext in [".mkv", ".mp4", ".webm"]:
                    abs_video_path = join(self.video_path, f"{video_id}{ext}")
                    if file_exists(abs_video_path):
                        break
                else:
                    skipped += 1
                    continue

            for ann in v['annotations']:
                segment = ann.get('segment', None)
                if segment is None:
                    continue
                if segment[1] <= segment[0]:
                    continue

                start, end = segment
                sentence = ann['sentence']
                if not sentence.endswith('.'):
                    sentence = sentence + "."

                if self.task in ["caption_clip", "all"]:
                    if self.qa_format:
                        question = random.choice(self.templates)
                        msg = dict(
                            question=question,
                            answer=sentence,
                            style="youcook2_label"
                        )
                    else:
                        msg = dict(
                            text=sentence,
                            style="video_short_caption"
                        )

                    meta = {
                        "clip_start_time": start,
                        "clip_end_time": end,
                    }

                    # Skip corrupt segments
                    if (video_id, start, end) not in corrupt_set:
                        video2msgs[(abs_video_path, start, end)].append(msg)
                        if self.flat:
                            formatted_ex = {
                                "video": abs_video_path,
                                "metadata": meta,
                                "message_list": [msg]
                            }
                            data_list.append(formatted_ex)

                if self.task in ["caption_start_end", "all"]:
                    msg = dict(
                        start_time=start,
                        end_time=end,
                        text=sentence,
                        style="video_clip_short_caption_start_end"
                    )
                    video2msgs[(abs_video_path, None, None)].append(msg)
                    if self.flat:
                        formatted_ex = {
                            "video": abs_video_path,
                            "message_list": [msg]
                        }
                        data_list.append(formatted_ex)

        log.warning(f"Skipped {skipped}/{skipped + len(data['database'])} missing Youcook2 videos.")

        if not self.flat:
            for (video, start, end), msgs in video2msgs.items():
                if len(msgs) == 0:
                    continue

                meta = None
                if start is not None and end is not None:
                    meta = {
                        "clip_start_time": start,
                        "clip_end_time": end,
                    }

                groups = (
                    split_into_groups(msgs, self.max_per_video)
                    if self.max_per_video
                    else [msgs]
                )

                for group in groups:
                    formatted_ex = {
                        "video": video,
                        "message_list": group,
                    }
                    if meta is not None:
                        formatted_ex["metadata"] = meta
                    data_list.append(formatted_ex)

        return data_list

    def get(self, item, rng):
        return self.data[item]


class PerceptionTest(DatasetBase):
    """PerceptionTest dataset for video understanding evaluation."""

    home = join(VIDEO_DATA_HOME, "perception_test")

    @classmethod
    def download(cls, n_procs=8):
        if not exists(cls.home):
            os.makedirs(cls.home, exist_ok=True)

        # Check if annotation files exist
        val_ann = join(cls.home, "mc_question_val", "validation-00000-of-00001.parquet")
        test_ann = join(cls.home, "Test", "mc_question", "test-00000-of-00001.parquet")
        train_ann = join(cls.home, "all_train.json")

        if exists(val_ann) or exists(test_ann) or exists(train_ann):
            # At least some files exist, nothing to download
            return

        log.info("PerceptionTest requires manual download of annotations and videos.")
        log.info("Please download the following files and place them in the appropriate directories:")
        log.info(f"  Validation annotations: mc_question_val/validation-00000-of-00001.parquet -> {cls.home}/mc_question_val/")
        log.info(f"  Validation videos: -> {cls.home}/videos/")
        log.info(f"  Test annotations: Test/mc_question/test-00000-of-00001.parquet -> {cls.home}/Test/mc_question/")
        log.info(f"  Test videos: -> {cls.home}/Test/videos/")
        log.info(f"  Train annotations: all_train.json -> {cls.home}/")
        log.info(f"  Train videos: -> {cls.home}/train_videos/")
        log.info("Dataset available at: https://github.com/google-deepmind/perception_test")

    def __init__(self, split, flat=False, max_per_video=None):
        self.split = split
        self.flat = flat
        self.max_per_video = max_per_video

        if split not in ["train", "validation", "test"]:
            raise ValueError(f"Invalid split: {split}")

        super().__init__(split)

    def qa_template(self, question, options, answer_id):
        """Format question text with options."""
        prefixes = "ABCDEFG"
        option_text = "\n".join(
            f"{prefix}. {opt}" for prefix, opt in zip(prefixes, options)
        )
        question = "\n".join([
            question,
            option_text,
            "Answer with the option's letter from the given choices directly.",
        ])
        if answer_id is not None:
            answer = prefixes[answer_id]
        else:
            answer = None
        return question, answer

    def load(self):
        data_list = []

        if self.split == "validation":
            parquet_path = join(self.home, "mc_question_val", "validation-00000-of-00001.parquet")
            df = pd.read_parquet(parquet_path)

            for idx, row in df.iterrows():
                video_path = join(self.home, "videos", row["video_name"] + ".mp4")
                question, answer = self.qa_template(row["question"], row["options"], int(row["answer_id"]))
                example = {
                    "question": question,
                    "answer": answer,
                    "video": video_path,
                    "metadata": {
                        "question_id": row["question_id"],
                        "video_id": row["video_name"],
                        "answer_idx": int(row["answer_id"]),
                        "area": row["area"],
                        "reasoning": row["reasoning"],
                    }
                }
                data_list.append(example)

        elif self.split == "test":
            parquet_path = join(self.home, "Test", "mc_question", "test-00000-of-00001.parquet")
            df = pd.read_parquet(parquet_path)

            for idx, row in df.iterrows():
                video_path = join(self.home, "Test", "videos", row["video_name"] + ".mp4")
                question, _ = self.qa_template(row["question"], row["options"], None)
                example = {
                    "question": question,
                    "video": video_path,
                    "metadata": {
                        "question_id": row["question_id"],
                        "video_id": row["video_name"],
                    }
                }
                data_list.append(example)

        elif self.split == "train":
            json_path = join(self.home, "all_train.json")
            with open(json_path) as f:
                train_anns = json.load(f)

            video2msgs = {}
            for video_id, ann in train_anns.items():
                video_path = join(self.home, "train_videos", video_id + ".mp4")
                for qa_data in ann["mc_question"]:
                    if video_path not in video2msgs:
                        video2msgs[video_path] = []
                    msg = {
                        "question": qa_data['question'],
                        "options": qa_data["options"],
                        "answer_idx": qa_data["answer_id"],
                        "style": "video_multiple_choice",
                    }
                    if self.flat:
                        formatted_ex = {
                            "video": video_path,
                            "message_list": [msg]
                        }
                        data_list.append(formatted_ex)
                    video2msgs[video_path].append(msg)

            if not self.flat:
                for video_path, msgs in video2msgs.items():
                    if len(msgs) == 0:
                        continue
                    if self.max_per_video:
                        for msg_group in split_into_groups(msgs, self.max_per_video):
                            formatted_ex = {
                                "video": video_path,
                                "message_list": msg_group
                            }
                            data_list.append(formatted_ex)
                    else:
                        formatted_ex = {
                            "video": video_path,
                            "message_list": msgs,
                        }
                        data_list.append(formatted_ex)

        return data_list

    def get(self, item, rng):
        return {**self.data[item], "style": "video_eval_multiple_choice"}


class Ego4d(DatasetBase):
    """Ego4d NLQ/MQ subsets
    **Please use Ego4dCachedClips for faster training as it uses pre-extracted clips.**

    NLQ: the query is expressed in text (e.g., "What did I put in the drawer?"),
    and the output response is the temporal window where the answer is visible
    or deducible. Annotators wrote these queries based on a set of 13 template
    questions.

    MQ: in which the query is the name of a high-level activity or "moment",
    and the response consists of all temporal windows where the activity occurs
    (e.g., "When did I read to my children?"). They established a taxonomy of
    110 activities in a data-driven, semi-automatic manner by mining the narration
    summaries. Moments capture high-level activities in the camera wearer's
    day, e.g., setting the table is a moment. For MQ, we provide the taxonomy of
    labels and ask annotators to label clips with each and every temporal segment
    containing a moment instance.

    ## Workflow (how clips are generated and used):
    1. **download()**: Mirrors load() logic to determine which clips are needed,
       generates ego4d_clips_to_extract.jsonl, and calls the extraction script
       to extract .mp4 clip files to ego4d-clips/ directory using ffmpeg.
    2. **__init__()**: Scans ego4d-clips/ directory and builds self.existing_clips
       set containing all available clip filenames.
    3. **load()**: Processes annotations, calls get_video_path() to check if clips
       exist in self.existing_clips, and creates dataset examples using the
       pre-extracted clips (much faster than on-the-fly extraction).
    """
    home = join(VIDEO_DATA_HOME, "Ego4d")
    video_path = join(home, "ego4d_data", "v2", "full_scale")
    clips_path = join(VIDEO_DATA_HOME, "ego4d-clips")

    @classmethod
    def download(cls, n_procs=1):
        """Ego4d dataset must be manually downloaded.

        This method:
        1. Verifies required files exist (full videos and annotations)
        2. Mirrors load() logic to determine which clips are needed for all tasks
        3. Generates ego4d_clips_to_extract.jsonl with clip specifications
        4. Calls extraction script to extract .mp4 clips using ffmpeg
        5. The extracted clips are then used by load() for faster training

        Required structure:
        - Ego4d/ego4d_data/ego4d.json (metadata)
        - Ego4d/ego4d_data/v2/full_scale/{video_uid}.mp4 (original videos)
        - Ego4d/ego4d_data/v2/annotations/nlq_{split}.json (NLQ annotations)
        - Ego4d/ego4d_data/v2/annotations/moments_{split}.json (MQ annotations)
        - ego4d-clips/{video_uid}_{start}_{end}_custom_clip.mp4 (pre-extracted clips, optional)

        Download from: https://ego4d-data.org/

        ## Clip Extraction (Optional but Recommended)

        Pre-extracting clips significantly speeds up training. To extract clips:

        ### Step 1: Understand which clips to extract
        The load() method in this class determines which video segments are needed based on:
        - **MQ (Moments Query)**: Activity annotations with temporal windows
          - For 'mq_label_clip' task: Extract exact [start, end] segments from annotations
          - For 'mq_label_start_end' task: Extract segments based on video_segment_length
          - For 'mq_temporal_grounding' task: Merges overlapping moments (75% threshold)
        - **NLQ (Natural Language Query)**: Query-based temporal annotations
          - For 'nlq_temporal_grounding' task: Extract segments based on video_segment_length

        Key extraction logic:
        - Clip naming: {video_uid}_{start:.3f}_{end:.3f}_custom_clip.mp4
        - Start/end times from label['video_start_time'] and label['video_end_time']
        - Skip clips where end <= start
        - For segmented tasks: use extract_segment() logic to determine segment boundaries

        ### Step 2: Extract the clips
        Use scripts/extract_ego4d_clips.py:
        ```bash
        python scripts/extract_ego4d_clips.py \\
            --ego4d_videos /path/to/Ego4d/ego4d_data/v2/full_scale \\
            --output_dir /path/to/ego4d-clips \\
            --annotations /path/to/Ego4d/ego4d_data/v2/annotations \\
            --splits train val test
        ```

        The script will:
        1. Parse NLQ and MQ annotation files for each split
        2. Collect all unique (video_uid, start, end) tuples
        3. Extract clips using ffmpeg with format: {video_uid}_{start:.3f}_{end:.3f}_custom_clip.mp4
        4. Save clips to the output directory

        ### Step 3: Verify extraction
        After extraction, the clips directory should contain files like:
        - 001e3e4e-2743-47fc-8564-d5efd11f9e90_5.133_15.200_custom_clip.mp4
        - 00299f75-e5b0-4748-9fc2-cad639e49d50_123.456_145.789_custom_clip.mp4

        Note: If clips are not pre-extracted, the dataset will attempt to extract segments
        on-the-fly from full videos during training, which is significantly slower.
        """
        required_files = [
            join(cls.home, "ego4d_data", "ego4d.json"),
        ]

        for file_path in required_files:
            if not exists(resource_path(file_path)):
                raise FileNotFoundError(
                    f"Required file not found: {file_path}\n\n"
                    "Please download the Ego4d dataset manually.\n"
                    "See: https://ego4d-data.org/\n\n"
                    "Required structure:\n"
                    f"- {cls.home}/ego4d_data/ego4d.json (metadata)\n"
                    f"- {cls.video_path}/{{video_uid}}.mp4 (original videos)\n"
                    f"- {cls.home}/ego4d_data/v2/annotations/nlq_{{split}}.json (NLQ annotations)\n"
                    f"- {cls.home}/ego4d_data/v2/annotations/moments_{{split}}.json (MQ annotations)\n"
                    f"- {cls.clips_path}/{{video_uid}}_{{start}}_{{end}}_custom_clip.mp4 (pre-extracted clips)\n"
                )

        # Extract clips if they don't exist yet
        if not exists(cls.clips_path) or len([f for f in os.listdir(cls.clips_path) if f.endswith('.mp4')]) == 0:
            log.info("Clips directory not found or empty. Extracting clips from full videos...")

            # Check if full videos and annotations exist
            if not exists(cls.video_path):
                raise FileNotFoundError(
                    f"Full videos directory not found: {cls.video_path}\n"
                    "Cannot extract clips without full videos. Please download from https://ego4d-data.org/"
                )

            annotations_path = join(cls.home, "ego4d_data", "v2", "annotations")
            if not exists(annotations_path):
                raise FileNotFoundError(
                    f"Annotations directory not found: {annotations_path}\n"
                    "Cannot extract clips without annotations. Please download from https://ego4d-data.org/"
                )

            # Generate clips JSONL
            clips_jsonl_path = join(cls.home, "ego4d_clips_to_extract.jsonl")
            if not exists(clips_jsonl_path):
                log.info("Generating clips JSONL from Ego4d annotations (mirroring load() logic)...")

                # Load ego4d metadata
                ego4d_meta_json_path = join(cls.home, "ego4d_data", "ego4d.json")
                ego4d_meta = json.load(open(resource_path(ego4d_meta_json_path)))
                video_uid_to_duration = {el['video_uid']: el['duration_sec'] for el in ego4d_meta['videos']}

                # Set seed for deterministic segment extraction (same as load())
                np.random.seed(42)
                random.seed(42)

                # Default parameters
                video_segment_length = 180

                # Helper to extract segment
                def extract_segment_static(moment_start, moment_end, video_dur, seg_length):
                    segment_start = random.uniform(max(0, moment_end - seg_length), moment_start)
                    segment_end = min(segment_start + seg_length, video_dur)
                    return segment_start, segment_end

                # Collect all unique clips
                clips_to_extract = set()

                # Process only train split
                split = "train"

                # Process MQ annotations
                mq_path = join(annotations_path, f"moments_{split}.json")
                mq_df = pd.read_json(resource_path(mq_path))
                mq_df['video_uid'] = mq_df['videos'].apply(lambda x: x['video_uid'])

                for _, row in mq_df.iterrows():
                    video_uid = row['video_uid']
                    video_dur = video_uid_to_duration.get(video_uid)

                    for clip in row['videos']['clips']:
                        for ann in clip['annotations']:
                            for label in ann['labels']:
                                if 'label' not in label:
                                    continue
                                start = label['video_start_time']
                                end = label['video_end_time']
                                if end <= start:
                                    continue

                                # mq_label_clip: extract exact clip
                                clips_to_extract.add((video_uid, start, end))

                                # mq_label_start_end with video_segment_length: extract segment
                                if video_segment_length is not None and video_dur:
                                    if end - start <= video_segment_length:
                                        segment_start, segment_end = extract_segment_static(
                                            start, end, video_dur, video_segment_length
                                        )
                                        clips_to_extract.add((video_uid, segment_start, segment_end))

                # Process NLQ annotations
                nlq_path = join(annotations_path, f"nlq_{split}.json")
                nlq_df = pd.read_json(resource_path(nlq_path))
                nlq_df['video_uid'] = nlq_df['videos'].apply(lambda x: x['video_uid'])

                for _, row in nlq_df.iterrows():
                    video_uid = row['video_uid']
                    video_dur = video_uid_to_duration.get(video_uid)

                    for clip in row['videos']['clips']:
                        for ann in clip['annotations']:
                            for query in ann['language_queries']:
                                start = query['clip_start_sec']
                                end = query['clip_end_sec']
                                if end <= start:
                                    continue

                                # nlq_temporal_grounding with video_segment_length: extract segment
                                if video_segment_length is not None and video_dur:
                                    if end - start <= video_segment_length:
                                        segment_start, segment_end = extract_segment_static(
                                            start, end, video_dur, video_segment_length
                                        )
                                        clips_to_extract.add((video_uid, segment_start, segment_end))

                # Write JSONL file
                log.info(f"Writing {len(clips_to_extract)} clips to {clips_jsonl_path}...")
                with open(clips_jsonl_path, 'w') as f:
                    for video_uid, start, end in sorted(clips_to_extract):
                        video_path = join(cls.video_path, f"{video_uid}.mp4")
                        clip_data = {
                            'video': video_path,
                            'clip_start_time': start,
                            'clip_end_time': end,
                            'video_uid': video_uid
                        }
                        f.write(json.dumps(clip_data) + '\n')

                log.info(f"Generated clips JSONL with {len(clips_to_extract)} unique clips")

            # Find extraction script
            extract_script = join(os.path.dirname(__file__), "..", "..", "scripts", "extract_ego4d_clips.py")
            extract_script = os.path.abspath(extract_script)
            if not exists(extract_script):
                raise FileNotFoundError(
                    f"Clip extraction script not found at: {extract_script}\n"
                    "Expected at: scripts/extract_ego4d_clips.py"
                )

            # Create output directory
            os.makedirs(cls.clips_path, exist_ok=True)

            # Run extraction script with sharding
            # For simplicity, use single shard or split into n_procs shards
            log.info(f"Running Ego4d clip extraction with {n_procs} shard(s)...")
            import ipdb; ipdb.set_trace()
            for shard_id in range(n_procs):
                cmd = [
                    sys.executable, extract_script,
                    "--shard_id", str(shard_id),
                    "--num_shards", str(n_procs),
                    "--clips_jsonl", clips_jsonl_path,
                    "--output_path", cls.clips_path,
                ]
                log.info(f"Running shard {shard_id + 1}/{n_procs}: {' '.join(cmd)}")
                result = subprocess.run(cmd)
                if result.returncode != 0:
                    log.warning(f"Shard {shard_id} finished with errors (see output above).")
        else:
            n_clips = len([f for f in os.listdir(cls.clips_path) if f.endswith('.mp4')])
            log.info(f"Clips directory exists with {n_clips} clips. Skipping extraction.")

    def __init__(
        self,
        split,
        task: Literal["mq_label_clip", "mq_label_start_end",
                      "mq_temporal_grounding", "nlq_temporal_grounding",
                      "all"] = "all",
        max_per_video: Optional[int] = None,
        video_segment_length: Optional[int] = 180,
        use_extracted_clips: bool = True
    ):
        """
        Args:
            split (str): Dataset split to use. Must be one of ["train", "val", "test"].
            task (Literal, optional): Task type to include in the dataset. Options:
                - "mq_label_clip": given a short clip, cropped from start to end,
                  label what's shown in the video.
                - "mq_label_start_end": Given a few minute long video and a start and
                  end timestamp in the prompt, output a label for that segment.
                - "nlq_temporal_grounding": Given a natural language query, localize the
                  part in the video which shows the answer to the question.
                - "all": Include all task types
                Defaults to "all".
            max_per_video (Optional[int], optional): Maximum number of messages to group
                per video example. If None, all messages for a video are grouped together.
                Defaults to None.
            video_segment_length: as ego4d videos are long, for some of the tasks such as
                mq_label_start_end and temporal_grounding, if this arg is provided, we split
                the video into segments with some max length and only load that segment
                during training.
            use_extracted_clips: whether to use pre-extracted clips or extract on-the-fly
                from full videos. When True, looks for clips in clips_path first.
        """
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        assert task in ["mq_label_clip", "mq_label_start_end",
                        "mq_temporal_grounding", "nlq_temporal_grounding",
                        "all"], f"Invalid task: {task}"
        if self.video_path.startswith("gs://"):
            raise ValueError("Ego4d dataset not supported on GCP. Please use Ego4dCachedClips if you're training on GCP.")
        self.task = task
        self.max_per_video = max_per_video
        self.video_segment_length = video_segment_length
        self.use_extracted_clips = use_extracted_clips

        # Build set of existing clips by scanning the clips directory
        if self.use_extracted_clips:
            clips_dir = resource_path(self.clips_path)
            if exists(clips_dir):
                self.existing_clips = set(
                    f for f in os.listdir(clips_dir)
                    if f.endswith('_custom_clip.mp4')
                )
            else:
                self.existing_clips = set()

        super().__init__(split)

    def get_video_path(self, video_uid, start_time, end_time):
        """Get the appropriate video path, preferring extracted clips when available."""
        if self.use_extracted_clips:
            # Try to find pre-extracted clip
            clip_filename = f"{video_uid}_{start_time:.3f}_{end_time:.3f}_custom_clip.mp4"
            clip_path = join(self.clips_path, clip_filename)

            if clip_filename in self.existing_clips:
                return clip_path, None  # No metadata needed for extracted clips
            else:
                return None, None
        # Fall back to original video with metadata
        original_path = join(self.video_path, f"{video_uid}.mp4")
        metadata = {
            "clip_start_time": start_time,
            "clip_end_time": end_time,
        }
        return original_path, metadata

    def extract_segment(self, moment_start, moment_end, video_dur):
        """
        Extract a segment from the video that contains the moment.
        Args:
            moment_start (float): Start time of the moment in seconds.
            moment_end (float): End time of the moment in seconds.
            video_dur (float): Duration of the full ego4d video in seconds.
        """
        segment_start = random.uniform(max(0, moment_end - self.video_segment_length), moment_start)
        segment_end = min(segment_start + self.video_segment_length, video_dur)
        moment_start_within_segment = moment_start - segment_start
        moment_end_within_segment = moment_end - segment_start

        return segment_start, segment_end, moment_start_within_segment, moment_end_within_segment

    def load(self):
        ego4d_meta_json_path = join(self.home, "ego4d_data", "ego4d.json")
        ego4d_meta = json.load(open(resource_path(ego4d_meta_json_path)))
        video_uid_to_duration = {el['video_uid']: el['duration_sec'] for el in ego4d_meta['videos']}

        nlq_json_path = join(self.home, "ego4d_data", "v2", "annotations", f"nlq_{self.split}.json")
        nlq_df = pd.read_json(resource_path(nlq_json_path))
        nlq_df['video_uid'] = nlq_df['videos'].apply(lambda x: x['video_uid'])

        mq_json_path = join(self.home, "ego4d_data", "v2", "annotations", f"moments_{self.split}.json")
        mq_df = pd.read_json(resource_path(mq_json_path))
        mq_df['video_uid'] = mq_df['videos'].apply(lambda x: x['video_uid'])

        data_list = []
        video2msgs = defaultdict(list)
        skipped = 0

        np.random.seed(42)
        random.seed(42)

        # Helper function to check overlap and merge moments
        def merge_overlapping_moments(moments, new_start, new_end, threshold=0.75):
            for i, (existing_start, existing_end) in enumerate(moments):
                intersection_start = max(new_start, existing_start)
                intersection_end = min(new_end, existing_end)
                intersection = max(0, intersection_end - intersection_start)

                new_len = new_end - new_start
                existing_len = existing_end - existing_start

                if new_len > 0 and existing_len > 0:
                    new_overlap_ratio = intersection / new_len
                    existing_overlap_ratio = intersection / existing_len

                    if new_overlap_ratio > threshold or existing_overlap_ratio > threshold:
                        moments.pop(i)
                        if intersection_end > intersection_start:
                            moments.append((intersection_start, intersection_end))
                        return True
            return False

        # Process MQ tasks
        if self.task in ["mq_temporal_grounding", "mq_label_clip", "mq_label_start_end", "all"]:
            for _, row in mq_df.iterrows():
                video_uid = row['video_uid']
                video_dur = video_uid_to_duration.get(video_uid)

                # Collect all labels for temporal grounding
                if self.task in ["mq_temporal_grounding", "all"]:
                    label_to_moments = {}

                for clip in row['videos']['clips']:
                    for ann in clip['annotations']:
                        for label in ann['labels']:
                            if 'label' not in label:
                                continue

                            start, end = label['video_start_time'], label['video_end_time']
                            if end <= start:
                                skipped += 1
                                continue

                            label_name = label['label']

                            # Handle temporal grounding
                            if self.task in ["mq_temporal_grounding", "all"]:
                                if label_name not in label_to_moments:
                                    label_to_moments[label_name] = []

                                if not merge_overlapping_moments(label_to_moments[label_name], start, end):
                                    label_to_moments[label_name].append((start, end))

                            # Handle clip labeling
                            if self.task in ["mq_label_clip", "all"]:
                                # Get the appropriate video path (extracted clip or original video)
                                video_path, clip_metadata = self.get_video_path(video_uid, start, end)
                                if video_path is None:
                                    continue

                                msg = dict(
                                    answer=label_name,
                                    question="Label the clip.",
                                    style="ego4d_mq_label_clip"
                                )

                                # Use extracted clip path if available, otherwise use original path with metadata
                                if clip_metadata is None:
                                    # Using extracted clip - no start/end needed in key
                                    video_key = (video_path, None, None)
                                else:
                                    # Using original video with clip metadata
                                    video_key = (video_path, start, end)

                                if msg not in video2msgs[video_key]:
                                    video2msgs[video_key].append(msg)

                            # Handle start/end labeling
                            if self.task in ["mq_label_start_end", "all"]:
                                if self.video_segment_length is not None:
                                    if end - start > self.video_segment_length:
                                        skipped += 1
                                        continue

                                    segment_start, segment_end, moment_start_within_segment, moment_end_within_segment = \
                                        self.extract_segment(start, end, video_dur)

                                    # Get the appropriate video path (extracted segment clip or original video)
                                    video_path, clip_metadata = self.get_video_path(video_uid, segment_start, segment_end)
                                    if video_path is None:
                                        continue

                                    msg = dict(
                                        answer=label_name,
                                        style="ego4d_mq_label_start_end",
                                        question=f"Label the segment from {moment_start_within_segment:.2f} to {moment_end_within_segment:.2f}.",
                                    )

                                    # Use extracted clip path if available, otherwise use original path with metadata
                                    if clip_metadata is None:
                                        # Using extracted clip - no start/end needed in key
                                        video_key = (video_path, None, None)
                                    else:
                                        # Using original video with clip metadata
                                        video_key = (video_path, segment_start, segment_end)

                                    video2msgs[video_key].append(msg)
                                else:
                                    # Get the appropriate video path (extracted clip or original video)
                                    video_path, clip_metadata = self.get_video_path(video_uid, start, end)

                                    if video_path is None:
                                        continue

                                    msg = dict(
                                        answer=label_name,
                                        style="ego4d_mq_label_start_end",
                                        question=f"Label the segment from {start:.2f} to {end:.2f}.",
                                    )

                                    # Use extracted clip path if available, otherwise use original path
                                    if clip_metadata is None:
                                        # Using extracted clip - no start/end needed in key
                                        video_key = (video_path, None, None)
                                    else:
                                        # Using original video with clip metadata
                                        video_key = (video_path, None, None)

                                    video2msgs[video_key].append(msg)

                # Process temporal grounding messages
                if self.task in ["mq_temporal_grounding", "all"]:
                    for label_name, moments in label_to_moments.items():
                        for start, end in moments:
                            if self.video_segment_length and end - start > self.video_segment_length:
                                continue

                            # Collect all moments for this label (including the current one)
                            localized_moments = []

                            if self.video_segment_length is not None:
                                segment_start, segment_end, moment_start_within_segment, moment_end_within_segment = \
                                    self.extract_segment(start, end, video_dur)

                                # Get the appropriate video path (extracted segment clip or original video)
                                video_path, clip_metadata = self.get_video_path(video_uid, segment_start, segment_end)

                                if video_path is None:
                                    continue

                                # Use extracted clip path if available, otherwise use original path with metadata
                                if clip_metadata is None:
                                    # Using extracted clip - no start/end needed in key
                                    video_key = (video_path, None, None)
                                else:
                                    # Using original video with clip metadata
                                    video_key = (video_path, segment_start, segment_end)

                                # Add current moment relative to segment
                                localized_moments.append((moment_start_within_segment, moment_end_within_segment))

                                # Add other overlapping moments within this segment
                                for other_start, other_end in moments:
                                    if other_start == start and other_end == end:
                                        continue
                                    if other_start < segment_end and other_end > segment_start:
                                        clipped_start = max(other_start, segment_start) - segment_start
                                        clipped_end = min(other_end, segment_end) - segment_start
                                        localized_moments.append((clipped_start, clipped_end))
                            else:
                                # Get the appropriate video path (extracted clip or original video)
                                video_path, clip_metadata = self.get_video_path(video_uid, start, end)

                                if video_path is None:
                                    continue

                                # Use extracted clip path if available, otherwise use original path
                                if clip_metadata is None:
                                    # Using extracted clip - no start/end needed in key
                                    video_key = (video_path, None, None)
                                else:
                                    # Using original video with clip metadata
                                    video_key = (video_path, None, None)

                                # Add all moments in absolute time
                                for moment_start, moment_end in moments:
                                    localized_moments.append((moment_start, moment_end))

                            answer = "\n".join([f"{s:.2f} - {e:.2f}" for s, e in localized_moments])

                            msg = dict(
                                question=f"Localize the event {label_name} in the video.",
                                style="ego4d_mq_temporal_grounding",
                                answer=answer
                            )
                            video2msgs[video_key].append(msg)

        # Process NLQ tasks
        if self.task in ["nlq_temporal_grounding", "all"]:
            for _, row in nlq_df.iterrows():
                video_uid = row['video_uid']
                video_dur = video_uid_to_duration.get(video_uid)

                for clip in row['videos']['clips']:
                    for ann in clip['annotations']:
                        for query in ann['language_queries']:
                            if (question := query.get('query')) is None:
                                continue
                            start, end = query['clip_start_sec'], query['clip_end_sec']
                            if end <= start:
                                skipped += 1
                                continue

                            if self.video_segment_length is not None:
                                if end - start > self.video_segment_length:
                                    skipped += 1
                                    continue

                                segment_start, segment_end, moment_start_within_segment, moment_end_within_segment = \
                                    self.extract_segment(start, end, video_dur)

                                # Get the appropriate video path (extracted segment clip or original video)
                                video_path, clip_metadata = self.get_video_path(video_uid, segment_start, segment_end)
                                if video_path is None:
                                    continue

                                answer = f"Start: {moment_start_within_segment:.2f}, end: {moment_end_within_segment:.2f}"

                                # Use extracted clip path if available, otherwise use original path with metadata
                                if clip_metadata is None:
                                    # Using extracted clip - no start/end needed in key
                                    video_key = (video_path, None, None)
                                else:
                                    # Using original video with clip metadata
                                    video_key = (video_path, segment_start, segment_end)
                            else:
                                # Get the appropriate video path (extracted clip or original video)
                                video_path, clip_metadata = self.get_video_path(video_uid, start, end)
                                if video_path is None:
                                    continue

                                answer = f"Start: {start:.2f}, end: {end:.2f}"

                                # Use extracted clip path if available, otherwise use original path
                                if clip_metadata is None:
                                    # Using extracted clip - no start/end needed in key
                                    video_key = (video_path, None, None)
                                else:
                                    # Using original video with clip metadata
                                    video_key = (video_path, None, None)

                            msg = dict(
                                question=question,
                                answer=answer,
                                style="ego4d_nlq_temporal_grounding"
                            )
                            video2msgs[video_key].append(msg)

        if skipped > 0:
            log.warning(f"Skipped {skipped} clips due to invalid start and end times.")

        for video_start_end, msgs in video2msgs.items():
            video, start, end = video_start_end
            meta = None
            if start is not None and end is not None:
                meta = {
                    "clip_start_time": start,
                    "clip_end_time": end,
                }
            if len(msgs) == 0:
                continue
            if self.max_per_video:
                for msg_group in split_into_groups(msgs, self.max_per_video):
                    formatted_ex = {"video": video, "message_list": msg_group}
                    if meta is not None:
                        formatted_ex["metadata"] = meta
                    data_list.append(formatted_ex)
            else:
                formatted_ex = {"video": video, "message_list": msgs}
                if meta is not None:
                    formatted_ex["metadata"] = meta
                data_list.append(formatted_ex)

        return data_list

    def get(self, item, rng):
        return self.data[item]


class Ego4dCachedClips(DatasetBase):
    """Ego4d with cached clips extracted offline.

    This dataset uses pre-generated metadata for faster loading compared to the
    regular Ego4d class which processes annotations on-the-fly.

    ## Cache files:
    - **ego4d_cached_train_metadata.jsonl**: Full cached metadata with message_list
    - **ego4d_cached_train_metadata.parquet**: Same data in parquet format with added columns:
      - `uid`: Video UID extracted from clip filename
      - `start`: Start timestamp extracted from clip filename
      - `end`: End timestamp extracted from clip filename
      - These columns enable users to extract clips themselves from the original videos

    Clip filename format: {uid}_{start}_{end}_custom_clip.mp4
    Example: dd08bc58-b614-4ba7-b883-a213560621dd_347.625_349.000_custom_clip.mp4
    """
    home = join(VIDEO_DATA_HOME, "Ego4d")
    clips_path = join(VIDEO_DATA_HOME, "ego4d-clips")

    @classmethod
    def download(cls, n_procs=1):
        """Check for required Ego4d cached clips files.

        This dataset uses pre-extracted clips and cached metadata for faster loading.

        Required files:
        - Ego4d/ego4d_cached_train_metadata.parquet (auto-downloaded from GCS)
        - ego4d-clips/{clip_filename}.mp4 (pre-extracted clip files)

        Note: The parquet file contains uid, start, and end columns extracted from
        clip filenames to enable users to extract clips themselves.

        ## Generating the required files:

        ### Step 1: Extract clips
        See the Ego4d.download() method documentation for detailed clip extraction instructions.
        Use scripts/extract_ego4d_clips.py to extract clips from full Ego4d videos.

        ### Step 2: Generate metadata JSONL
        The ego4d_cached_train_metadata.jsonl file should contain one entry per clip:
        ```json
        {
          "clip": "video_uid_start_end_custom_clip.mp4",
          "corrupt": false,
          ...other metadata fields (message_list, metadata, etc.)...
        }
        ```

        Fields:
        - clip (str): Filename of the clip (matches files in ego4d-clips/)
        - corrupt (bool): Whether the clip is corrupted and should be skipped
        - message_list (list): List of message dicts with 'question', 'answer', 'style' fields
        - metadata (dict, optional): Additional metadata for the example

        ### Step 3: Identify corrupt clips
        During extraction or validation, identify clips that:
        - Failed to extract properly
        - Have encoding errors
        - Cannot be decoded
        - Have duration mismatches

        Mark these clips with "corrupt": true in the JSONL file. The load() method
        will automatically filter them out.

        ### Step 4: Consolidate files
        If you have a separate corrupt clips file (e.g., corrupt.parquet), use:
        ```bash
        python scripts/consolidate_ego4d_metadata.py \\
            --metadata Ego4d/ego4d_cached_train_metadata.jsonl \\
            --corrupt Ego4d/corrupt.parquet \\
            --output Ego4d/ego4d_cached_train_metadata_consolidated.jsonl
        ```

        See: https://ego4d-data.org/ for downloading the original dataset.
        """
        # Download cached metadata from GCS
        metadata_file = join(cls.home, "ego4d_cached_train_metadata.parquet")
        if not exists(metadata_file):
            log.info("Downloading Ego4d cached metadata from GCS...")
            maybe_download_file(
                "https://storage.googleapis.com/molmo-datasets/ego4d_cached_train_metadata.parquet",
                metadata_file
            )

        required_files = [
            metadata_file,
        ]

        missing_files = []
        for file_path in required_files:
            if not exists(resource_path(file_path)):
                missing_files.append(file_path)

        if missing_files:
            raise FileNotFoundError(
                f"Required files not found:\n" +
                "\n".join(f"  - {f}" for f in missing_files) +
                "\n\n"
                "This dataset requires pre-extracted clips.\n\n"
                "To generate clips:\n"
                "1. Obtain Ego4d full videos and annotations from https://ego4d-data.org/\n"
                "2. See Ego4d.load() in olmo/data/academic_video_datasets.py for how clips are determined\n"
                "3. Use scripts/extract_ego4d_clips.py to extract the clips\n"
                "4. Generate ego4d_cached_train_metadata.jsonl with 'corrupt' field for each clip\n"
            )

        # Verify clips directory exists and has clips
        if not exists(cls.clips_path):
            raise FileNotFoundError(
                f"Clips directory not found: {cls.clips_path}\n"
                "Please extract Ego4d clips first."
            )

        clip_files = [f for f in os.listdir(cls.clips_path) if f.endswith('.mp4')]
        if len(clip_files) == 0:
            raise FileNotFoundError(
                f"No clips found in {cls.clips_path}\n"
                "Please extract Ego4d clips first."
            )

        log.info(f"Ego4d dataset ready: {len(clip_files)} clips available")

    def __init__(
        self,
        split,
        task: Literal["all"] = "all",
        max_per_video: Optional[int] = None,
    ):
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        assert task in ["all"], f"Invalid task: {task}"
        self.task = task
        self.max_per_video = max_per_video

        super().__init__(split)

    def load(self):
        ego4d_meta_json_path = join(self.home, "ego4d_cached_train_metadata.jsonl")
        ego4d_meta = pd.read_json(resource_path(ego4d_meta_json_path), lines=True)

        # Filter out corrupt clips if 'corrupt' field exists
        if 'corrupt' in ego4d_meta.columns:
            ego4d_meta = ego4d_meta[~ego4d_meta['corrupt']]
            ego4d_meta = ego4d_meta.drop(columns=['corrupt'])

        ego4d_meta['video'] = ego4d_meta['clip'].apply(lambda x: join(self.clips_path, x))
        ego4d_meta = ego4d_meta.drop(columns=['clip'])
        data_list = ego4d_meta.to_dict(orient="records")

        return data_list

    def get(self, item, rng):
        return self.data[item]


class NeXTQA(DatasetBase):
    data_path = join(VIDEO_DATA_HOME, "NeXTQA")

    @classmethod
    def download(cls, n_procs=1):
        required = [
            join(cls.data_path, "train.csv"),
            join(cls.data_path, "val.csv"),
            join(cls.data_path, "map_vid_vidorID.json"),
        ]
        missing = [f for f in required if not exists(f)]
        if missing:
            raise FileNotFoundError(
                "NeXTQA data not found. Missing files:\n"
                + "\n".join(f"  {f}" for f in missing) + "\n\n"
                "Please download train.csv, val.csv, test.csv, and map_vid_vidorID.json from:\n"
                "  https://github.com/doc-doc/NExT-QA/tree/main/dataset/nextqa\n"
                "and place them at:\n"
                f"  {cls.data_path}/\n\n"
                "For videos, follow the instructions at https://github.com/doc-doc/NExT-QA\n"
            )

    def __init__(self, split, task="multiple-choice", flat: bool = False,
                 max_per_video: Optional[int] = None, difficulty="all"):
        if task == "multiple-choice":
            assert split in ["train", "val", "test"]
        else:
            raise NotImplementedError(f"Task {task} not implemented")
        assert difficulty in ["easy", "medium", "hard", "all"]
        self.difficulty = difficulty
        self.task = task
        self.flat = flat
        self.max_per_video = max_per_video
        super().__init__(split)

    @staticmethod
    def mc_qoa_template(data):
        options = [data[f'a{idx}'].strip() for idx in range(5)]
        option_text = "\n".join(
            f"{chr(ord('A') + idx)}. {options[idx]}" for idx in range(5)
        )
        answer = f"{chr(ord('A') + int(data['answer']))}"
        question = "\n".join(
            [
                data["question"].strip(),
                option_text,
                "Answer with the option's letter from the given choices directly.",
            ]
        )
        return question, options, answer

    def load(self):
        task = self.task
        data_list = []
        if task == "multiple-choice" and self.split == "test":
            df_path = join(self.data_path, "MC", "test-00000-of-00001.parquet")
            df = pd.read_parquet(resource_path(df_path))

            for idx, row in df.iterrows():
                video_path = join(self.data_path, "NExTVideo", f"{row['video']}.mp4")
                question, options, answer = self.mc_qoa_template(row)

                example_id = f"{row['video']}_{idx}_type_{row['type']}"
                if self.difficulty != "all":
                    # difficulty filtering for test not available without v1 DatasetSampleDifficulty
                    pass

                example = {
                    "question": question,
                    "answer": answer,
                    "video": video_path,
                    "style": "video_eval_multiple_choice",
                    "metadata": dict(
                        example_id=example_id,
                        question_id=str(idx),
                        question_type=row["type"],
                        video_id=row["video"],
                        options=options,
                    )
                }
                data_list.append(example)

        elif task == "multiple-choice" and self.split in {"train", "val"}:
            df_path = join(self.data_path, f"{self.split}.csv")
            id_map_path = join(self.data_path, "map_vid_vidorID.json")

            df = pd.read_csv(resource_path(df_path))
            id_map = json.load(open(resource_path(id_map_path)))
            df["video_path"] = df["video"].apply(
                lambda x: join(self.data_path, "NExTVideo-all-videos", id_map[str(x)] + ".mp4")
            )

            video2msgs = {}
            for row in df.itertuples(False):
                video_path = row.video_path
                video2msgs[video_path] = video2msgs.get(video_path, [])
                msg = dict(
                    question=row.question,
                    options=[row.a0, row.a1, row.a2, row.a3, row.a4],
                    answer_idx=row.answer,
                    style="video_multiple_choice",
                )
                video2msgs[video_path].append(msg)

                if self.flat:
                    formatted_ex = {
                        "video": video_path,
                        "message_list": [msg]
                    }
                    data_list.append(formatted_ex)

            if not self.flat:
                for video, msgs in video2msgs.items():
                    if len(msgs) == 0:
                        continue
                    if self.max_per_video:
                        for msg in split_into_groups(msgs, self.max_per_video):
                            formatted_ex = {
                                "video": video,
                                "message_list": msg
                            }
                            data_list.append(formatted_ex)
                    else:
                        formatted_ex = {
                            "video": video,
                            "message_list": msgs,
                        }
                        data_list.append(formatted_ex)
        else:
            raise NotImplementedError(f"Task {task} not implemented")

        return data_list

    def get(self, idx, rng):
        return self.data[idx]


class CharadesSTA(DatasetBase):
    """CharadesSTA Video dataset"""
    video_path = join(VIDEO_DATA_HOME, "Charades")

    # Videos with incorrect segment annotations
    INCORRECT_SEGMENTS = {"LEOL6", "AKKWU"}

    templates = [
        "What action is being performed?",
        "What is the person doing?",
        "What action is the person taking?",
        "What activity does the video depict?"
    ]

    @classmethod
    def download(cls, n_procs=1):
        annotations = join(cls.video_path, "charades_sta_train.txt")
        videos_dir = join(cls.video_path, "Charades_v1")
        if not exists(annotations):
            raise FileNotFoundError(
                f"CharadesSTA annotations not found at: {annotations}\n\n"
                "Please download the dataset manually and place it at:\n"
                f"  {cls.video_path}/\n"
            )
        if not exists(videos_dir):
            raise FileNotFoundError(
                f"CharadesSTA videos not found at: {videos_dir}\n\n"
                "Please download Charades_v1 videos and place them at:\n"
                f"  {videos_dir}/\n"
            )

        # Auto-generate existing_videos.json by scanning Charades_v1/
        existing_videos_file = join(cls.video_path, "existing_videos.json")
        if not exists(existing_videos_file):
            log.info("Scanning Charades_v1/ to generate existing_videos.json...")
            existing = []
            for f in os.listdir(videos_dir):
                if f.endswith(".mp4"):
                    existing.append(f.replace(".mp4", ""))
            with open(existing_videos_file, "w") as f:
                json.dump(existing, f)
            log.info(f"Generated {existing_videos_file} with {len(existing)} videos")

    def __init__(
            self,
            split,
            flat: bool = False,
            task: Literal["caption_clip", "all"] = "caption_clip",
            qa_format: bool = False
    ):
        assert split in ["train"], f"Invalid split: {split}"
        self.split = split
        self.flat = flat
        self.task = task
        self.qa_format = qa_format
        super().__init__(split)

    def load(self):
        data = Path(resource_path(join(VIDEO_DATA_HOME, "Charades", "charades_sta_train.txt"))).read_text().splitlines()
        data_list = []
        video2msgs = defaultdict(list)
        skipped = 0
        with open(resource_path(join(VIDEO_DATA_HOME, "Charades", "existing_videos.json")), "r") as f:
            existing_videos = json.load(f)

        video2segments = defaultdict(list)
        path_to_id = {}
        for line in data:
            rest, caption = line.strip().split("##")
            video_id, start, end = rest.split(" ")
            abs_video_path = join(self.video_path, "Charades_v1", f"{video_id}.mp4")
            if video_id not in existing_videos:
                skipped += 1
                continue
            path_to_id[abs_video_path] = video_id

            if self.qa_format:
                msg = dict(
                    question=random.choice(self.templates),
                    answer=caption,
                    style="charades_sta"
                )
            else:
                msg = dict(
                    text=caption,
                    style="charades_sta"
                )

            start, end = float(start), float(end)

            if end <= start:
                skipped += 1
                continue

            video2segments[abs_video_path].append((start, end, caption))

            if self.task in ["caption_clip", "all"]:
                if self.flat:
                    formatted_ex = {
                        "video": abs_video_path,
                        "metadata": {
                            "clip_start_time": start,
                            "clip_end_time": end,
                        },
                        "message_list": [msg]
                    }
                    data_list.append(formatted_ex)
                else:
                    video2msgs[(abs_video_path, start, end)].append(msg)

        if not self.flat:
            for video_start_end, msgs in video2msgs.items():
                video, start, end = video_start_end
                if len(msgs) == 0:
                    continue
                formatted_ex = {
                    "video": video,
                    "metadata": {
                        "clip_start_time": start,
                        "clip_end_time": end,
                    },
                    "message_list": msgs,
                }
                data_list.append(formatted_ex)

        log.warning(f"Skipped {skipped} missing or corrupt CharadesSTA annotations.")

        return data_list

    def get(self, item, rng):
        return self.data[item]



class CameraBenchTrain(DatasetBase):
    HOME = join(VIDEO_DATA_HOME, "CameraBench")
    PARQUET_PATH = join(HOME, "camerabench_qa.parquet")
    VIDEO_DIR = join(HOME, "train")
    HF_REPO = "allenai/Molmo2-CameraBenchTrain"

    @classmethod
    def download(cls, n_procs=None):
        if exists(cls.PARQUET_PATH):
            return
        hf_hub_download(
            repo_id=cls.HF_REPO,
            repo_type="dataset",
            filename="camerabench_qa.parquet",
            local_dir=cls.HOME,
        )

    def __init__(self, split):
        self.download()
        if not exists(self.VIDEO_DIR):
            raise FileNotFoundError(
                f"CameraBench videos not found at {self.VIDEO_DIR}. "
                f"Please download them following the instructions at "
                f"https://github.com/sy77777en/CameraBench?tab=readme-ov-file#-how-to-access--evaluate-on-camerabench "
                f"and place them at {self.VIDEO_DIR}"
            )
        super().__init__(split)

    def load(self):
        data = pd.read_parquet(resource_path(self.PARQUET_PATH))
        data_list = []
        for _, row in data.iterrows():
            video_path = os.path.join(self.VIDEO_DIR, row["video_path"])
            msgs = []
            for ex in row["mc_qa_list"]:
                question = ex["Question"]
                answer = ex["Answer"]
                neg_options = list(ex["NegativeAnswers"])
                answer_idx = random.randint(0, len(neg_options))
                neg_options.insert(answer_idx, answer)
                msgs.append(dict(
                    question=question,
                    answer_idx=answer_idx,
                    options=neg_options,
                    style="video_multiple_choice",
                ))
            if not msgs:
                continue
            data_list.append({
                "video": video_path,
                "message_list": msgs,
            })
        return data_list

    def get(self, idx, rng):
        return self.data[idx]


class CLEVRER(DatasetBase):
    """CLEVRER Video QA dataset — CoLlision Events for Video REpresentation and Reasoning"""
    HOME = join(VIDEO_DATA_HOME, "CLEVRER")
    QUESTIONS_URL = "http://data.csail.mit.edu/clevrer/questions/{split}.json"
    VIDEOS_URL = "http://data.csail.mit.edu/clevrer/videos/{split}/video_{split}.zip"

    @classmethod
    def download(cls, n_procs=None):
        os.makedirs(cls.HOME, exist_ok=True)
        for split in ["train", "validation", "test"]:
            json_path = join(cls.HOME, f"{split}.json")
            maybe_download_file(cls.QUESTIONS_URL.format(split=split), json_path)
        # Videos: skip if batch folders already exist (e.g. video_00000-01000)
        if any(
            d.startswith("video_") and "-" in d
            for d in os.listdir(cls.HOME)
            if os.path.isdir(join(cls.HOME, d))
        ):
            return
        for split in ["train", "validation", "test"]:
            maybe_download_and_unzip(cls.HOME, cls.VIDEOS_URL.format(split=split))

    def __init__(self, split):
        assert split in ["train", "validation", "test"], f"Invalid split: {split}"
        self.download()
        super().__init__(split)

    @staticmethod
    def format_options_and_answer(choices):
        choice_texts = []
        correct_choices = []
        answer_idx = None
        for i, choice in enumerate(choices):
            choice_texts.append(choice['choice'])
            if choice['answer'] == 'correct':
                answer_idx = i
                correct_choices.append(f"{chr(ord('A') + choice['choice_id'])}")
        multiple_correct = len(correct_choices) > 1
        return choice_texts, answer_idx, multiple_correct

    def load(self):
        json_path = join(self.HOME, f"{self.split}.json")
        with open(resource_path(json_path)) as f:
            data = json.load(f)

        data_list = []
        for ex in data:
            video = ex['video_filename']
            video_id = int(video.split(".")[0].replace("video_", ""))
            k = video_id // 1000
            video_folder = f"video_{k * 1000:05d}-{(k + 1) * 1000:05d}"
            video_path = join(self.HOME, video_folder, video)

            msgs = []
            for q in ex['questions']:
                if "answer" in q:
                    msgs.append(dict(
                        question=q['question'],
                        answer=q['answer'],
                        style="video_short_answer",
                    ))
                elif "choices" in q:
                    options, answer_idx, multiple_correct = self.format_options_and_answer(q['choices'])
                    if multiple_correct:
                        continue
                    if answer_idx is not None:
                        msgs.append(dict(
                            question=q['question'],
                            options=options,
                            answer_idx=answer_idx,
                            style="video_multiple_choice",
                        ))

            if not msgs:
                continue
            data_list.append({
                "video": video_path,
                "message_list": msgs,
            })
        return data_list

    def get(self, idx, rng):
        return self.data[idx]


class FunQA(DatasetBase):
    """FunQA Video QA dataset"""
    HOME = join(VIDEO_DATA_HOME, "FunQA")
    HF_REPO = "fesvhtr/FunQA"

    @classmethod
    def download(cls, n_procs=None):
        os.makedirs(cls.HOME, exist_ok=True)
        # Download all JSON annotation files
        for fname in ["FunQA_train.json", "FunQA_val.json", "FunQA_test.json", "Funqa_mcqa_v1.json"]:
            if not exists(join(cls.HOME, fname)):
                hf_hub_download(
                    repo_id=cls.HF_REPO,
                    repo_type="dataset",
                    filename=fname,
                    local_dir=cls.HOME,
                    token=os.environ.get("HF_TOKEN"),
                )
        # Download and extract video zips per split
        for split in ["train", "val", "test"]:
            video_dir = join(cls.HOME, split)
            zip_name = f"{split}.zip"
            if not exists(video_dir) or not os.listdir(video_dir):
                zip_path = join(cls.HOME, zip_name)
                if not exists(zip_path):
                    hf_hub_download(
                        repo_id=cls.HF_REPO,
                        repo_type="dataset",
                        filename=zip_name,
                        local_dir=cls.HOME,
                        token=os.environ.get("HF_TOKEN"),
                    )
                log.info(f"Extracting {zip_path} to {cls.HOME}...")
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(cls.HOME)

    def __init__(self, split):
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.download()
        super().__init__(split)

    @staticmethod
    def format_options_and_answer(answer_idx, options):
        modified_options = []
        for i, opt in enumerate(options):
            opt_content = opt.replace(f"Options {i+1}: ", "").strip()
            modified_options.append(opt_content)
        return modified_options

    def load(self):
        video_path = join(self.HOME, self.split)
        oe_json_path = join(self.HOME, f"FunQA_{self.split}.json")
        mc_json_path = join(self.HOME, "Funqa_mcqa_v1.json")

        with open(resource_path(oe_json_path)) as f:
            oe_data = json.load(f)
        with open(resource_path(mc_json_path)) as f:
            mc_data = json.load(f)

        all_files = set(list_directory(self.HOME, recurse=True, include_dirs=False))

        video2msgs = {}
        for ex in oe_data:
            if ex['task'] in ['H1', 'C1', 'M1', 'C4', 'C5']:
                continue  # Skip timestamp-output tasks
            if ex['task'].startswith("H"):
                video_dir = f"{self.split}_humor"
            elif ex['task'].startswith("C"):
                video_dir = f"{self.split}_creative"
            elif ex['task'].startswith("M"):
                video_dir = f"{self.split}_magic"
            else:
                raise ValueError(f"Unknown task {ex['task']}")

            video = join(video_path, video_dir, ex['visual_input'])
            if video not in all_files:
                continue

            msg = dict(
                question=ex['instruction'],
                answer=ex['output'],
                style="video_short_answer",
            )
            video2msgs.setdefault(video, []).append(msg)

        for ex in mc_data:
            if ex['visual_input'].startswith("H"):
                video_dir = f"{self.split}_humor"
            elif ex['visual_input'].startswith("C"):
                video_dir = f"{self.split}_creative"
            elif ex['visual_input'].startswith("M"):
                video_dir = f"{self.split}_magic"
            else:
                continue

            video = join(video_path, video_dir, ex['visual_input'])
            if video not in all_files:
                continue

            sentences = ex['instruction'].split("\n")
            question = sentences[-2].replace("The Question is:", "").strip()
            options_str = sentences[-1]
            options_list = ast.literal_eval(options_str.replace(" The Options are: ", ""))

            try:
                gt_idx = int(ex['gt'])
                if gt_idx > len(options_list):
                    continue
            except Exception:
                continue

            options = self.format_options_and_answer(gt_idx, options_list)
            msg = dict(
                question=question,
                options=options,
                answer_idx=gt_idx - 1,
                style="video_multiple_choice",
            )
            video2msgs.setdefault(video, []).append(msg)

        data_list = []
        for video, msgs in video2msgs.items():
            if msgs:
                data_list.append({
                    "video": video,
                    "message_list": msgs,
                })
        return data_list

    def get(self, idx, rng):
        return self.data[idx]


class How2QA(DatasetBase):
    """How2QA Video QA dataset — multiple choice QA on How2 instructional video clips."""
    HOME = join(VIDEO_DATA_HOME, "how2QA")
    CLIP_DIR = join(HOME, "video-clips")
    EXTENSIONS_PATH = join(HOME, "extensions.json")
    CSV_URL = "https://raw.githubusercontent.com/ych133/How2R-and-How2QA/master/how2QA/how2QA_{split}_release.csv"
    CORRUPT_VIDEOS = {"Z1NEMNKgR9U"}
    CORRUPT_CLIPS = {"onGTfd-EKrs_89.15_89.44.mp4", "bkl7eK8B6ig_147.73_149.83.mp4"}

    @classmethod
    def download(cls, n_procs=None):
        for split in ["train", "val"]:
            csv_path = join(cls.HOME, f"how2QA_{split}_release.csv")
            maybe_download_file(cls.CSV_URL.format(split=split), csv_path)

    def _build_extensions_json(self):
        """Scan CLIP_DIR to build extensions.json mapping video IDs to file extensions.

        Clip filenames follow {vid}_{start}_{end}.{ext}. We parse from the right:
        the last two underscore-separated parts of the stem are numeric (start, end),
        and everything before them is the video ID.
        """
        log.info(f"Building {self.EXTENSIONS_PATH} from clips in {self.CLIP_DIR} ...")
        extensions = {}
        for fname in os.listdir(self.CLIP_DIR):
            stem, dot, ext = fname.rpartition(".")
            if not dot or not ext:
                continue
            parts = stem.rsplit("_", 2)
            if len(parts) < 3:
                continue
            vid = parts[0]
            # Validate that parts[1] and parts[2] look numeric
            try:
                float(parts[1])
                float(parts[2])
            except ValueError:
                continue
            # First extension seen for a vid wins; they should all agree
            if vid not in extensions:
                extensions[vid] = ext
        if not extensions:
            raise FileNotFoundError(
                f"Could not build extensions.json: no valid clip files found in {self.CLIP_DIR}.\n"
                f"Expected clip filename format: {{vid}}_{{start}}_{{end}}.{{ext}} "
                f"(e.g. abc123_10.5_20.3.mp4)"
            )
        with open(self.EXTENSIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(extensions, f)
        log.info(f"Wrote {self.EXTENSIONS_PATH} with {len(extensions)} video IDs")

    def __init__(self, split, flatten=False):
        if split == "validation":
            split = "val"
        assert split in ["train", "val"]
        self.flatten = flatten
        self.download()
        if not exists(self.CLIP_DIR):
            raise FileNotFoundError(
                f"How2QA video clips not found at {self.CLIP_DIR}.\n"
                f"You must download the full How2 videos from "
                f"https://github.com/ych133/How2R-and-How2QA and clip them yourself.\n"
                f"Place the clips in: {self.CLIP_DIR}\n"
                f"Expected clip filename format: {{vid}}_{{start}}_{{end}}.{{ext}} "
                f"(e.g. abc123_10.5_20.3.mp4)"
            )
        if not exists(self.EXTENSIONS_PATH):
            self._build_extensions_json()
        super().__init__(split)

    def load(self):
        data = pd.read_csv(resource_path(join(self.HOME, f'how2QA_{self.split}_release.csv')), header=None)
        with open(resource_path(self.EXTENSIONS_PATH), encoding="utf-8") as f:
            extensions = json.load(f)

        data_list = []
        errors = defaultdict(int)
        clip_to_question = defaultdict(list)
        for row in data.itertuples(index=False):
            start, end = eval(row[1].replace(':', ','))
            if start >= end:
                errors["invalid_clip"] += 1
                continue

            vid = row[0]
            if vid not in extensions:
                errors["missing_in_extensions"] += 1
                continue
            if vid in self.CORRUPT_VIDEOS:
                errors["corrupt"] += 1
                continue

            ext = extensions[vid]
            start = round(start, 2)
            end = round(end, 2)
            clip_id = f"{vid}_{start}_{end}.{ext}"
            if clip_id in self.CORRUPT_CLIPS:
                errors["corrupt_clip"] += 1
                continue
            clip_path = join(self.CLIP_DIR, clip_id)
            answer = row[6]
            neg_options = list(row[2:5])
            question = row[5]
            answer_idx = random.randint(0, len(neg_options))
            neg_options.insert(answer_idx, answer)
            msg = dict(
                question=question,
                answer_idx=answer_idx,
                options=neg_options,
                style="video_multiple_choice",
            )
            clip_to_question[clip_path].append(msg)

        if errors:
            log.info(f"How2QA {self.split}: skipped rows: {dict(errors)}")

        for clip_path, msgs in clip_to_question.items():
            if self.flatten:
                for msg in msgs:
                    data_list.append(dict(video=clip_path, message_list=[msg]))
            else:
                data_list.append(dict(video=clip_path, message_list=msgs))
        return data_list

    def get(self, idx, rng):
        return self.data[idx]


class IntentQA(DatasetBase):
    """IntentQA Video QA dataset — intent-driven multi-choice QA on NExT-QA videos."""
    HOME = join(VIDEO_DATA_HOME, "IntentQA")
    VIDEO_DIR = join(HOME, "videos")
    CSV_URL = "https://raw.githubusercontent.com/JoseponLee/IntentQA/main/datasets/IntentQA/{split}.csv"

    @classmethod
    def download(cls, n_procs=None):
        os.makedirs(cls.HOME, exist_ok=True)
        for split in ["train", "val", "test"]:
            csv_path = join(cls.HOME, f"{split}.csv")
            maybe_download_file(cls.CSV_URL.format(split=split), csv_path)

    def __init__(self, split, answer_type="all", flat=False):
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        assert answer_type in ["open_ended", "multi_choice", "all"], f"Invalid answer type: {answer_type}"
        self.answer_type = answer_type
        self.flat = flat
        self.download()
        if not exists(self.VIDEO_DIR):
            raise FileNotFoundError(
                f"IntentQA videos not found at {self.VIDEO_DIR}.\n"
                f"This dataset uses NExT-QA videos. Download from:\n"
                f"  https://drive.google.com/drive/folders/17xu7AGS1VZ8m9J4MeZEzH2515I9MiAPg\n"
                f"Place MP4 files at: {self.VIDEO_DIR}"
            )
        super().__init__(split)

    def load(self):
        csv_path = resource_path(join(self.HOME, f"{self.split}.csv"))
        df = pd.read_csv(csv_path)

        data_list = []
        video2msgs = {}
        for row in df.itertuples(index=False):
            abs_video_path = join(self.VIDEO_DIR, f"{row.video_id}.mp4")

            question = row.question
            options = [getattr(row, f'a{i}') for i in range(5)]
            answer_idx = row.answer
            if answer_idx >= len(options):
                continue
            answer = options[answer_idx]

            if self.answer_type in ("open_ended", "all"):
                msg = dict(
                    question=question,
                    answer=answer,
                    style="video_short_answer",
                )
                video2msgs.setdefault(abs_video_path, []).append(msg)
                if self.flat:
                    data_list.append(dict(
                        video=abs_video_path,
                        metadata=dict(video_id=row.video_id, type=row.type),
                        message_list=[msg],
                    ))

            if self.answer_type in ("multi_choice", "all"):
                msg = dict(
                    question=question,
                    options=options,
                    answer_idx=answer_idx,
                    style="video_multiple_choice",
                )
                video2msgs.setdefault(abs_video_path, []).append(msg)
                if self.flat:
                    data_list.append(dict(
                        video=abs_video_path,
                        metadata=dict(video_id=row.video_id, type=row.type),
                        message_list=[msg],
                    ))

        if not self.flat:
            for video, msgs in video2msgs.items():
                if msgs:
                    data_list.append(dict(video=video, message_list=msgs))

        return data_list

    def get(self, idx, rng):
        return self.data[idx]


class SocialIQ2(DatasetBase):
    """Social-IQ 2.0 Video QA dataset — social intelligence MC QA on video clips."""
    HOME = join(VIDEO_DATA_HOME, "social-iq2")
    QA_DIR = join(HOME, "qa")
    VIDEO_DIR = join(HOME, "video")
    HF_REPO = "PediaMedAI/Social-IQ-Video"

    @classmethod
    def download(cls, n_procs=None):
        if exists(join(cls.QA_DIR, "qa_train.json")):
            return
        os.makedirs(cls.QA_DIR, exist_ok=True)
        for hf_split, local_name in [("train", "qa_train"), ("validation", "qa_val"), ("test", "qa_test")]:
            ds = datasets.load_dataset(cls.HF_REPO, split=hf_split)
            ds.to_json(join(cls.QA_DIR, f"{local_name}.json"))

    def __init__(self, split, flat=False):
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.flat = flat
        self.download()
        if not exists(self.VIDEO_DIR):
            raise FileNotFoundError(
                f"Social-IQ 2.0 videos not found at {self.VIDEO_DIR}.\n"
                f"Download from: https://github.com/abwilf/Social-IQ-2.0-Challenge\n"
                f"Place MP4 files at: {self.VIDEO_DIR}"
            )
        super().__init__(split)

    def load(self):
        qa_path = join(self.QA_DIR, f"qa_{self.split}.json")
        data = pd.read_json(resource_path(qa_path), orient="records", lines=True)
        all_videos = set(list_directory(self.VIDEO_DIR))

        data_list = []
        video2msgs = {}
        skip_count = 0
        for _, row in data.iterrows():
            video_path = join(self.VIDEO_DIR, f"{row['vid_name']}.mp4")
            if video_path not in all_videos:
                skip_count += 1
                continue

            options = [row['a0'], row['a1'], row['a2'], row['a3']]
            answer_idx = row['answer_idx']
            if answer_idx >= len(options):
                continue
            msg = dict(
                question=row['q'],
                answer_idx=answer_idx,
                options=options,
                style="video_multiple_choice",
            )

            if self.flat:
                data_list.append(dict(video=video_path, message_list=[msg]))
            else:
                video2msgs.setdefault(video_path, []).append(msg)

        if not self.flat:
            for video, msgs in video2msgs.items():
                if msgs:
                    data_list.append(dict(video=video, message_list=msgs))

        if skip_count:
            log.warning(f"SocialIQ2 {self.split}: skipped {skip_count} rows with missing videos")
        return data_list

    def get(self, idx, rng):
        return self.data[idx]


class SUTDTrafficQA(DatasetBase):
    """SUTD-TrafficQA — multi-choice QA on traffic event videos."""
    HOME = join(VIDEO_DATA_HOME, "SUTD-TrafficQA")
    VIDEO_DIR = join(HOME, "compressed_videos")
    DOWNLOAD_PAGE = "https://sutdcv.github.io/SUTD-TrafficQA/#/download"
    ZENODO_URL = "https://zenodo.org/records/7431011"

    @classmethod
    def download(cls, n_procs=None):
        # Annotations and videos are access-restricted on Zenodo; manual download required.
        pass

    def __init__(self, split, flat=False):
        assert split in ["train", "test"], f"Invalid split: {split}"
        self.flat = flat
        jsonl_path = join(self.HOME, f"R2_{split}.jsonl")
        if not exists(jsonl_path):
            raise FileNotFoundError(
                f"SUTD-TrafficQA annotations not found at {jsonl_path}.\n"
                f"This dataset requires manual download (access-restricted on Zenodo).\n"
                f"\n"
                f"Steps:\n"
                f"  1. Go to {self.ZENODO_URL}\n"
                f"  2. Request access and download R2_train.jsonl and R2_test.jsonl\n"
                f"  3. Place them in: {self.HOME}\n"
                f"  4. Download compressed_videos/ and place at: {self.VIDEO_DIR}\n"
                f"\n"
                f"Alternatively, request via: {self.DOWNLOAD_PAGE}\n"
                f"\n"
                f"Expected directory structure:\n"
                f"  {self.HOME}/\n"
                f"    R2_train.jsonl\n"
                f"    R2_test.jsonl\n"
                f"    compressed_videos/\n"
                f"      *.mp4"
            )
        super().__init__(split)

    def load(self):
        tmp = pd.read_json(
            resource_path(join(self.HOME, f'R2_{self.split}.jsonl')),
            orient="values", lines=True,
        )
        # First row is column headers, rest is data
        cols = tmp.iloc[0].tolist()
        data = tmp.iloc[1:].copy()
        data.columns = cols

        data_list = []
        video2msgs = {}
        for _, row in data.iterrows():
            video_path = join(self.VIDEO_DIR, row['vid_filename'])
            options = [row['option0'], row['option1'], row['option2'], row['option3']]
            question = row['q_body']
            answer = options[row['answer']]
            options = [o for o in options if o]  # filter empty options
            answer_idx = options.index(answer)

            msg = dict(
                question=question,
                answer_idx=answer_idx,
                options=options,
                style="video_multiple_choice",
            )

            if self.flat:
                data_list.append(dict(video=video_path, message_list=[msg]))
            else:
                video2msgs.setdefault(video_path, []).append(msg)

        if not self.flat:
            for video, msgs in video2msgs.items():
                if msgs:
                    data_list.append(dict(video=video, message_list=msgs))

        return data_list

    def get(self, idx, rng):
        return self.data[idx]


class STAR(DatasetBase):
    """STAR (Situated Reasoning in Real-World Videos) QA dataset.

    Uses Charades videos with clip-level QA annotations.
    Annotations auto-downloaded from STAR benchmark S3 bucket.
    Videos (Charades) require manual download from https://prior.allenai.org/projects/charades
    """
    HOME = join(VIDEO_DATA_HOME, "STAR")
    VIDEO_DIR = join(VIDEO_DATA_HOME, "Charades", "Charades_v1")
    S3_BASE = "https://star-benchmark.s3.us-east.cloud-object-storage.appdomain.cloud/Question_Answer_SituationGraph"
    FILE_SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}

    @classmethod
    def download(cls, num_procs=None):
        os.makedirs(cls.HOME, exist_ok=True)
        for file_split in ["train", "val", "test"]:
            json_path = join(cls.HOME, f"STAR_{file_split}.json")
            compressed_path = join(cls.HOME, f"STAR_{file_split}_qa.json")
            if exists(compressed_path):
                continue
            maybe_download_file(
                f"{cls.S3_BASE}/STAR_{file_split}.json",
                json_path,
            )
            with open(resource_path(json_path), 'r') as f:
                data = json.load(f)
            data = [
                {k: x[k] for k in
                 ["video_id", "start", "end", "question", "answer", "choices"] if k in x}
                for x in data
            ]
            with open(compressed_path, 'w') as f:
                json.dump(data, f)

    def __init__(
        self,
        split,
        answer_type="all",
        flat=False,
        max_per_video=None,
    ):
        assert split in ["train", "validation", "test"], f"Invalid split: {split}"
        assert answer_type in ["open_ended", "multi_choice", "all"], f"Invalid answer type: {answer_type}"
        self.answer_type = answer_type
        self.flat = flat
        self.max_per_video = max_per_video
        if not exists(self.VIDEO_DIR):
            raise FileNotFoundError(
                f"Charades videos not found at {self.VIDEO_DIR}.\n"
                f"STAR uses Charades_v1 videos. Download from:\n"
                f"  https://prior.allenai.org/projects/charades\n"
                f"Place MP4 files at: {self.VIDEO_DIR}"
            )
        super().__init__(split)

    @staticmethod
    def format_options_and_answer(answer, choices):
        choice_texts = []
        answer_idx = None
        for i, choice in enumerate(choices):
            choice_texts.append(choice['choice'])
            if choice['choice'] == answer:
                answer_idx = i
        return choice_texts, answer_idx

    def load(self):
        file_split = self.FILE_SPLIT_MAP[self.split]
        json_path = join(self.HOME, f"STAR_{file_split}.json")
        compressed_path = join(self.HOME, f"STAR_{file_split}_qa.json")
        if exists(compressed_path):
            with open(resource_path(compressed_path), 'r') as f:
                data = json.load(f)
        else:
            with open(resource_path(json_path), 'r') as f:
                data = json.load(f)

        data_list = []
        video2msgs = {}
        video2meta = {}
        num_no_gt_ans = 0
        total_num = 0
        for ex in data:
            video_id = ex['video_id']
            start = float(ex['start'])
            end = float(ex['end'])
            abs_video_path = join(self.VIDEO_DIR, f"{video_id}.mp4")

            clip_id = f"{video_id}_{start}_{end}"
            video2msgs.setdefault(clip_id, [])
            if clip_id not in video2meta:
                video2meta[clip_id] = [start, end, abs_video_path]

            msgs = []
            if self.answer_type in ("open_ended", "all"):
                msg = dict(
                    question=ex['question'],
                    answer=ex['answer'],
                    style="video_short_answer"
                )
                msgs.append(msg)

            if self.answer_type in ("multi_choice", "all"):
                choices = ex['choices']
                options, answer_idx = self.format_options_and_answer(ex['answer'], choices)
                if answer_idx is not None:
                    msg = dict(
                        question=ex['question'],
                        options=options,
                        answer_idx=answer_idx,
                        style="video_multiple_choice"
                    )
                    msgs.append(msg)
                else:
                    num_no_gt_ans += 1
                total_num += 1

            video2msgs[clip_id] += msgs

            if self.flat:
                for msg in msgs:
                    data_list.append({
                        "video": abs_video_path,
                        "metadata": {
                            "clip_start_time": start,
                            "clip_end_time": end,
                        },
                        "message_list": [msg]
                    })

        if num_no_gt_ans > 0:
            log.warning(f"STAR: Skipped {num_no_gt_ans} / {total_num} MC examples without GT answer.")

        if not self.flat:
            for clip_id, msgs in video2msgs.items():
                if len(msgs) == 0:
                    continue
                start, end, video = video2meta[clip_id]
                if self.max_per_video:
                    for msg_group in split_into_groups(msgs, self.max_per_video):
                        data_list.append({
                            "video": video,
                            "message_list": msg_group,
                            "metadata": {
                                "clip_start_time": float(start),
                                "clip_end_time": float(end),
                            }
                        })
                else:
                    data_list.append({
                        "video": video,
                        "message_list": msgs,
                        "metadata": {
                            "clip_start_time": float(start),
                            "clip_end_time": float(end),
                        }
                    })
        return data_list

    def get(self, item, rng):
        return self.data[item]


class SportsQA(DatasetBase):
    """SportsQA Video QA dataset.

    Annotations auto-downloaded from figshare.
    Videos require manual download from https://huggingface.co/datasets/HopLeeTop/Sports-QA
    """
    HOME = join(VIDEO_DATA_HOME, "SportsQA")
    FIGSHARE_URL = "https://ndownloader.figshare.com/files/57448480"

    @classmethod
    def download(cls, num_procs=None):
        maybe_download_and_unzip(cls.HOME, cls.FIGSHARE_URL, expected_dir="meta-data")

    def __init__(self, split, max_per_video=None, flat=False):
        assert split in ["train", "val", "test"]
        self.max_per_video = max_per_video
        self.flat = flat
        if not exists(self.HOME):
            raise FileNotFoundError(
                f"SportsQA data not found at {self.HOME}.\n"
                f"Download videos from:\n"
                f"  https://huggingface.co/datasets/HopLeeTop/Sports-QA\n"
                f"Place video files (.avi/.mp4/.webm) at: {self.HOME}"
            )
        super().__init__(split)

    def load(self):
        json_path = join(self.HOME, "meta-data", f"{self.split}.json")
        df = pd.read_json(resource_path(json_path))

        # Build/load cached file listing for extension resolution
        all_files_f = join(self.HOME, "all_videos.json")
        if file_exists(all_files_f):
            log.info(f"Using cached filelist {all_files_f}")
            with open(resource_path(all_files_f)) as f:
                all_files = json.load(f)
        else:
            video_files = list_directory(self.HOME, recurse=True, include_files=True, include_dirs=False)
            all_files = [os.path.relpath(el, self.HOME) for el in video_files]
            if get_global_rank() == 0:
                log.info(f"Saving filelist to {all_files_f}")
                with open(all_files_f, 'w') as f:
                    json.dump(all_files, f)
        all_files = set(all_files)

        data_list = []
        video2msgs = {}
        skipped = 0
        for _, row in df.iterrows():
            video_id = row['video']
            abs_video_path = None
            for ext in [".avi", ".mp4", ".webm"]:
                if f"{video_id}{ext}" in all_files:
                    abs_video_path = join(self.HOME, f"{video_id}{ext}")
                    break
            if abs_video_path is None:
                skipped += 1
                continue

            msg = dict(
                question=row['question'],
                answer=row['answer'],
                style="video_short_answer"
            )
            video2msgs.setdefault(abs_video_path, []).append(msg)

            if self.flat:
                data_list.append({"video": abs_video_path, "message_list": [msg]})

        if skipped > 0:
            log.warning(f"SportsQA: Skipped {skipped} examples with missing videos.")

        if not self.flat:
            for video, msgs in video2msgs.items():
                if self.max_per_video:
                    for msg_group in split_into_groups(msgs, self.max_per_video):
                        data_list.append({"video": video, "message_list": msg_group})
                else:
                    data_list.append({"video": video, "message_list": msgs})
        return data_list

    def get(self, item, rng):
        return self.data[item]


class RoadTextVQA(DatasetBase):
    """RoadTextVQA Video QA dataset.

    Annotations auto-downloaded from http://cvit.iiit.ac.in/images/datasets/RoadTextVQA/
    Videos require manual download from https://github.com/georg3tom/RoadtextVQA
    """
    HOME = join(VIDEO_DATA_HOME, "RoadTextVQA")
    BASE_URL = "http://cvit.iiit.ac.in/images/datasets/RoadTextVQA"

    @classmethod
    def download(cls, num_procs=None):
        for split in ["train", "val", "test"]:
            maybe_download_file(f"{cls.BASE_URL}/{split}.json", join(cls.HOME, f"{split}.json"))

    def __init__(self, split, flat=False, answer_type="open_ended"):
        assert split in ["train", "val", "test"]
        self.flat = flat
        self.answer_type = answer_type
        video_dir = join(self.HOME, "videos")
        if not exists(video_dir):
            raise FileNotFoundError(
                f"RoadTextVQA videos not found at {video_dir}.\n"
                f"Download from:\n"
                f"  https://github.com/georg3tom/RoadtextVQA\n"
                f"Place video files at: {video_dir}"
            )
        super().__init__(split)

    def load(self):
        json_path = join(self.HOME, f"{self.split}.json")
        with open(resource_path(json_path)) as f:
            data = json.load(f)

        data_list = []
        video2msgs = {}
        video_dir = join(self.HOME, "videos")
        for ex in data['data']:
            video_path = join(video_dir, ex['video'])

            if len(ex['answer']) == 1:
                answer = ex['answer'][0]
            else:
                answer = random.choice(ex['answer'])

            msg = dict(
                question=ex['question'],
                answer=answer,
                style="video_short_answer"
            )
            video2msgs.setdefault(video_path, []).append(msg)

            if self.flat:
                data_list.append({
                    "video": video_path,
                    "metadata": {
                        "questionId": ex['questionId'],
                        "answer": ex['answer']
                    },
                    "message_list": [msg]
                })

        if not self.flat:
            for video, msgs in video2msgs.items():
                if len(msgs) == 0:
                    continue
                data_list.append({"video": video, "message_list": msgs})
        return data_list

    def get(self, item, rng):
        return self.data[item]


class VideoLocalizedNarratives(DatasetBase):
    """Video Localized Narratives QA dataset.

    Annotations auto-downloaded from https://google.github.io/video-localized-narratives/
    Videos (OOPS transformed) require manual download.
    """
    HOME = join(VIDEO_DATA_HOME, "video-localized-narratives")
    VIDEO_PATH = join(VIDEO_DATA_HOME, "oops/oops_video")
    QA_ZIP_URL = "https://storage.googleapis.com/video-localized-narratives/videoqa.zip"

    @classmethod
    def download(cls, num_procs=None):
        maybe_download_and_unzip(cls.HOME, cls.QA_ZIP_URL, expected_dir="videoqa")

    def __init__(self, split, flat=False):
        assert split in ["train", "val"]
        self.flat = flat
        self.download()
        if not exists(self.VIDEO_PATH):
            raise FileNotFoundError(
                f"OOPS videos not found at {self.VIDEO_PATH}.\n"
                f"Download OOPS videos from: https://oops.cs.columbia.edu/data/\n"
                f"Place them at: {self.VIDEO_PATH}"
            )
        super().__init__(split)

    def load(self):
        json_path = join(self.HOME, "videoqa", "text_output",
                         f"oops_{self.split}", "qa_text_output.json")
        data = pd.read_json(resource_path(json_path))

        data_list = []
        for ex in data['annotations']:
            video_path = join(self.VIDEO_PATH, ex['video_name'] + '.mp4')
            qa_pairs = ex['qa_pairs']
            msgs = []
            for q in qa_pairs:
                msg = dict(
                    question=q['raw_question'],
                    answer=q['raw_answer'],
                    style="video_short_answer"
                )
                msgs.append(msg)
                if self.flat:
                    data_list.append({
                        "video": video_path,
                        "metadata": {
                            "video_name": ex['video_name'],
                            "question_id": q['question_id'],
                        },
                        "message_list": [msg]
                    })
            if not self.flat:
                if len(msgs) == 0:
                    continue
                data_list.append({
                    "video": video_path,
                    "message_list": msgs,
                    "metadata": {"video_name": ex['video_name']}
                })
        return data_list

    def get(self, item, rng):
        return self.data[item]


class VideoLocalizedNarrativesCaptionHf(Dataset):
    """Video Localized Narratives caption dataset (multi-source: OOPS, Kinetics, UVO, OVIS).

    Annotations auto-downloaded from https://google.github.io/video-localized-narratives/
    Videos require manual download from respective sources.
    """
    HOME = join(VIDEO_DATA_HOME, "video-localized-narratives")
    HF_DIR = join(VIDEO_DATA_HOME, "video-localized-narratives", "hf_dataset")
    HF_REPO = "allenai/Molmo2-VideoLocalizedNarrativesCaptionHf"

    @classmethod
    def download(cls, num_procs=None):
        if file_exists(cls.HF_DIR):
            return
        snapshot_download(
            repo_id=cls.HF_REPO, repo_type="dataset",
            local_dir=cls.HOME,
            allow_patterns=["hf_dataset/*"],
        )

    def __init__(self, split):
        assert split in ["train"]
        self.download()
        video_dirs = [
            join(VIDEO_DATA_HOME, "oops/oops_video/train"),
            join(VIDEO_DATA_HOME, "kinetics/kinetics700/train"),
            join(VIDEO_DATA_HOME, "UVO/uvo_videos_sparse"),
            join(VIDEO_DATA_HOME, "UVO/uvo_videos_dense"),
            join(VIDEO_DATA_HOME, "OVIS/train_mp4_5fps"),
        ]
        missing = [d for d in video_dirs if not exists(d)]
        if missing:
            log.warning(
                f"VideoLocalizedNarrativesCaptionHf: some video directories are missing:\n"
                + "\n".join(f"  - {d}" for d in missing)
                + "\nDownload videos from their respective sources:\n"
                + "  OOPS: https://oops.cs.columbia.edu/data/\n"
                + "  Kinetics-700: https://github.com/cvdfoundation/kinetics-dataset\n"
                + "  UVO: https://sites.google.com/view/unidentified-video-object\n"
                + "  OVIS: https://songbai.site/ovis/"
            )
        self.data = datasets.Dataset.load_from_disk(self.HF_DIR)

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        messages = [dict(text=c, object=a.lower(), style="video_object_caption")
                    for c, a in zip(ex['captions'], ex['actor_names'])]
        return dict(
            video=join(VIDEO_DATA_HOME, ex["video"]),
            message_list=messages,
            metadata=dict(id=ex["id"])
        )


class CinepileHf(Dataset):
    """CinePile Video QA dataset — multi-choice QA on movie clips.

    Annotations auto-downloaded from HuggingFace.
    Videos (YouTube movie clips) require manual download.
    """
    HOME = join(VIDEO_DATA_HOME, "cinepile")
    VIDEO_DIR = join(HOME, "videos")
    HF_REPO = "allenai/Molmo2-cinepile"
    HF_DIR = join(HOME, "hf_dataset")

    @classmethod
    def download(cls, n_procs=None):
        if exists(cls.HF_DIR):
            return
        ds = datasets.load_dataset(cls.HF_REPO)
        ds.save_to_disk(cls.HF_DIR)

    def __init__(self, split, with_subtitle: bool = False):
        assert split in ["train", "test"], f"Invalid split: {split}"
        self.with_subtitle = with_subtitle
        self.download()
        if not exists(self.VIDEO_DIR):
            raise FileNotFoundError(
                f"CinePile videos not found at {self.VIDEO_DIR}.\n"
                f"Download YouTube clips using video IDs from the dataset.\n"
                f"Place MP4 files at: {self.VIDEO_DIR}/{{video_id}}/{{video_id}}.mp4"
            )
        self.data = datasets.load_from_disk(self.HF_DIR)[split]

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        abs_video_path = join(self.VIDEO_DIR, ex['video_id'], f"{ex['video_id']}.mp4")

        out = dict(
            video=abs_video_path,
            metadata=dict(video_id=ex['video_id'], movie_name=ex["movie_name"]),
        )
        style = "video_multiple_choice"
        if self.with_subtitle:
            subtitle = ex.get('subtitles', '')
            if subtitle:
                out["subtitle"] = subtitle.replace('<subtitle> ', '').strip()
                style = "video_multiple_choice_w_subtitle"

        messages = []
        for row in ex["examples"]:
            question = row['question']
            options = row['choices']
            answer_idx = row['answer_key_position']
            if answer_idx >= len(options):
                raise IndexError(f"Answer index {answer_idx} out of bounds for options {options}")
            messages.append(dict(
                question=question,
                answer_idx=answer_idx,
                options=options,
                style=style,
            ))
        out["message_list"] = messages
        return out


class NewsVideoQA(DatasetBase):
    """NewsVideoQA — open-ended QA on news broadcast video clips.

    Annotations auto-downloaded from HuggingFace.
    Videos require manual download from: https://cvit.iiit.ac.in/research/projects/cvit-projects/videoqa
    """
    HOME = join(VIDEO_DATA_HOME, "NewsVideoQA")
    HF_REPO = "allenai/Molmo2-NewsVideoQA"
    CORRUPT_FILES = {'389x18', '76x16', '254x17', '254x21', '389x15', '254x20', '389x17'}

    @classmethod
    def download(cls, n_procs=None):
        data_dir = join(cls.HOME, "final_data_feb_16")
        if exists(join(data_dir, "json_files", "raw_files", "new_train.json")):
            return
        os.makedirs(data_dir, exist_ok=True)
        for fname in ["json_files/raw_files/new_train.json",
                       "json_files\raw_files\new_val.json"]:
            hf_hub_download(
                repo_id=cls.HF_REPO, repo_type="dataset",
                filename=fname, local_dir=data_dir,
            )

    def __init__(self, split, flat=False, filter_empty_answers=True):
        assert split in ["train", "val"]
        self.flat = flat
        self.filter_empty_answers = filter_empty_answers
        self.download()
        video_dir = join(self.HOME, "final_data_feb_16", "videos")
        if not exists(video_dir):
            raise FileNotFoundError(
                f"NewsVideoQA videos not found at {video_dir}.\n"
                f"Download from: https://cvit.iiit.ac.in/research/projects/cvit-projects/videoqa\n"
                f"Extract videos into: {video_dir}"
            )
        super().__init__(split)

    def load(self):
        json_path = join(self.HOME, "final_data_feb_16", "json_files", "raw_files", f"new_{self.split}.json")
        with open(resource_path(json_path)) as f:
            data = json.load(f)

        video_dir = join(self.HOME, "final_data_feb_16", "videos")
        data_list = []
        video2msgs = {}
        for item in data:
            if item['uni_clipped_id'] in self.CORRUPT_FILES:
                continue
            answer = item['answer']
            if self.filter_empty_answers and len(answer.strip()) == 0:
                continue

            video_path = join(video_dir, self.split, f"{item['uni_clipped_id']}.mp4")
            msg = dict(
                question=item['question'],
                answer=answer,
                style="video_short_answer",
            )

            if self.flat:
                data_list.append(dict(
                    video=video_path,
                    message_list=[msg],
                    metadata=dict(decode_method="av_noseek"),
                ))
            else:
                video2msgs.setdefault(video_path, []).append(msg)

        if not self.flat:
            for video, msgs in video2msgs.items():
                data_list.append(dict(
                    video=video,
                    message_list=msgs,
                    metadata=dict(decode_method="av_noseek"),
                ))
        return data_list

    def get(self, idx, rng):
        return self.data[idx]


class Countix(DatasetBase):
    """Countix — repetition counting QA on Kinetics-700 videos.

    Annotations auto-downloaded from HuggingFace.
    Videos (Kinetics-700) require manual download.
    """
    HOME = join(VIDEO_DATA_HOME, "Countix")
    VIDEO_DIR = join(VIDEO_DATA_HOME, "kinetics", "kinetics700")
    HF_REPO = "allenai/Molmo2-Countix"

    @classmethod
    def _find_annotation_dir(cls):
        """Return dir with annotations: v1 path (VIDEO_DIR) or v2 path (HOME)."""
        for d in [cls.VIDEO_DIR, cls.HOME]:
            if exists(join(d, "countix_train_mapped.csv")):
                return d
        return None

    @classmethod
    def download(cls, n_procs=None):
        if cls._find_annotation_dir() is not None:
            return
        os.makedirs(cls.HOME, exist_ok=True)
        for fname in ["countix_train_mapped.csv", "class_to_question.json"]:
            hf_hub_download(
                repo_id=cls.HF_REPO,
                repo_type="dataset",
                filename=fname,
                local_dir=cls.HOME,
            )

    def __init__(self, split, answer_format="oe"):
        assert split in ["train"]
        assert answer_format in ["mc", "oe"]
        self.answer_format = answer_format
        self.download()
        if not exists(self.VIDEO_DIR):
            raise FileNotFoundError(
                f"Kinetics-700 videos not found at {self.VIDEO_DIR}.\n"
                f"Countix uses Kinetics-700 train videos. Download from:\n"
                f"  https://github.com/cvdfoundation/kinetics-dataset\n"
                f"Place videos at: {self.VIDEO_DIR}/train/{{label}}/{{video_name}}.mp4"
            )
        super().__init__(split)

    @staticmethod
    def _generate_consecutive_options(correct, min_val=1):
        possible_offsets = [o for o in range(-3, 1) if correct + o >= min_val]
        start_offset = random.choice(possible_offsets)
        start = correct + start_offset
        return [start + i for i in range(4)]

    def load(self):
        anno_dir = self._find_annotation_dir()
        csv_path = join(anno_dir, "countix_train_mapped.csv")
        data = pd.read_csv(resource_path(csv_path))
        with open(resource_path(join(anno_dir, "class_to_question.json"))) as f:
            question_template = json.load(f)

        data_list = []
        for _, row in data.iterrows():
            video_path = join(self.VIDEO_DIR, row['video_path'])
            question = random.choice(question_template[row['class']])
            count = row['count']

            if self.answer_format == "oe":
                msg = dict(
                    question=question,
                    answer=str(count),
                    style="video_short_answer",
                )
            else:
                options = self._generate_consecutive_options(count, min_val=1)
                answer_idx = options.index(count)
                msg = dict(
                    question=question,
                    options=options,
                    answer_idx=answer_idx,
                    style="video_multiple_choice",
                )
            data_list.append(dict(video=video_path, message_list=[msg]))
        return data_list

    def get(self, item, rng):
        return self.data[item]


class Paxion(DatasetBase):
    """Paxion — action antonym MC QA on SSV2 + Ego4d videos.

    Annotations auto-downloaded from HuggingFace.
    Videos (Something-Something V2 + Ego4d) require manual download.
    """
    HOME = join(VIDEO_DATA_HOME, "paxion")
    SSV2_VIDEO_DIR = join(VIDEO_DATA_HOME, "sth-sth-v2", "videos")
    EGO4D_VIDEO_DIR = join(VIDEO_DATA_HOME, "Ego4d", "ego4d_data", "v2", "full_scale")
    EGO4D_CLIPS_DIR = join(VIDEO_DATA_HOME, "paxion-ego4d-clips")
    HF_REPO = "allenai/Molmo2-Paxion"
    CORRUPT_FILES = {"703d550a-0a84-4bcf-9b45-e25c864ade70"}
    CORRUPT_CLIPS = {"1348c9f9-fc8b-40c7-b1ab-5b6281e5d390_978.617_979.098.mp4"}
    QUESTION_TEMPLATES = [
        "What activity does the video depict?",
        "What is the action performed by the person in the video?",
        "Which one of these descriptions correctly matches the actions in the video?",
    ]

    @classmethod
    def download(cls, n_procs=None):
        if exists(join(cls.HOME, "ssv2")):
            return
        snapshot_download(
            repo_id=cls.HF_REPO,
            repo_type="dataset",
            local_dir=cls.HOME,
            max_workers=n_procs or 1,
        )

    def __init__(self, split, flat=False, max_per_video=None):
        assert split in ["train", "val", "test"]
        self.flat = flat
        self.max_per_video = max_per_video
        self.download()
        if not exists(self.SSV2_VIDEO_DIR):
            raise FileNotFoundError(
                f"Something-Something V2 videos not found at {self.SSV2_VIDEO_DIR}.\n"
                f"Download from:\n"
                f"  https://developer.qualcomm.com/software/ai-datasets/something-something\n"
                f"Place videos at: {self.SSV2_VIDEO_DIR}"
            )
        if not exists(self.EGO4D_VIDEO_DIR) and not exists(self.EGO4D_CLIPS_DIR):
            raise FileNotFoundError(
                f"Ego4d videos not found.\n"
                f"Either place full-scale videos at: {self.EGO4D_VIDEO_DIR}/{{video_uid}}.mp4\n"
                f"  (download from https://ego4d-data.org/)\n"
                f"Or place pre-extracted clips at: {self.EGO4D_CLIPS_DIR}/"
            )
        super().__init__(split)

    @staticmethod
    def _clean_label(label):
        """Strip leading '#X' and capitalize."""
        if isinstance(label, str) and label.startswith("#") and len(label) > 1 and not label[1].isspace():
            label = label[2:].lstrip()
        if isinstance(label, str) and label and not label[0].isupper():
            label = label[0].upper() + label[1:]
        return label

    def load(self):
        ssv2_path = join(self.HOME, "ssv2", "antonyms", f"{self.split}_with_rel_path.json")
        ego4d_path = join(self.HOME, "ego4d",
                          "egoclip_subset_action_antonyms_train_val_test_split", f"{self.split}.jsonl")

        ssv2_df = pd.read_json(resource_path(ssv2_path))
        ego4d_df = pd.read_json(resource_path(ego4d_path), lines=True)

        data_list = []
        video2msgs = defaultdict(list)
        rng = random.Random(42)

        # SSV2 data
        for row in ssv2_df.itertuples(False):
            abs_video_path = join(self.SSV2_VIDEO_DIR, row.rel_vid_path)
            question = rng.choice(self.QUESTION_TEMPLATES)
            options = [row.label, row.label_action_antonym_clip_text, 'Not sure']
            options = rng.sample(options, len(options))
            answer_idx = options.index(row.label)
            msg = dict(
                question=question, options=options,
                answer_idx=answer_idx, style="video_multiple_choice",
            )
            video2msgs[(abs_video_path, None, None)].append(msg)
            if self.flat:
                data_list.append(dict(video=abs_video_path, message_list=[msg]))

        # Ego4d data (try extracted clips first, fall back to full-scale videos)
        for row in ego4d_df.itertuples(False):
            if row.clip_start >= row.clip_end:
                continue
            if row.video_uid in self.CORRUPT_FILES:
                continue

            # Try extracted clip first
            clip_filename = f"{row.video_uid}_{float(row.clip_start):.3f}_{float(row.clip_end):.3f}.mp4"
            clip_path = join(self.EGO4D_CLIPS_DIR, clip_filename)
            if clip_filename in self.CORRUPT_CLIPS:
                continue
            if exists(clip_path):
                abs_video_path = clip_path
                clip_metadata = None
                video_key = (abs_video_path, None, None)
            else:
                abs_video_path = join(self.EGO4D_VIDEO_DIR, f"{row.video_uid}.mp4")
                clip_metadata = dict(clip_start_time=row.clip_start, clip_end_time=row.clip_end)
                video_key = (abs_video_path, row.clip_start, row.clip_end)

            question = rng.choice(self.QUESTION_TEMPLATES)
            label = self._clean_label(row.clip_text)
            antonym = self._clean_label(row.action_antonym_clip_text)
            options = [label, antonym, 'Not sure']
            options = rng.sample(options, len(options))
            answer_idx = options.index(label)
            msg = dict(
                question=question, options=options,
                answer_idx=answer_idx, style="video_multiple_choice",
            )
            video2msgs[video_key].append(msg)
            if self.flat:
                data_list.append(dict(
                    video=abs_video_path, message_list=[msg], metadata=clip_metadata,
                ))

        if not self.flat:
            for (video, start, end), msgs in video2msgs.items():
                if not msgs:
                    continue
                meta = None
                if start is not None and end is not None:
                    meta = dict(clip_start_time=start, clip_end_time=end)
                if self.max_per_video:
                    for msg_group in split_into_groups(msgs, self.max_per_video):
                        ex = dict(video=video, message_list=msg_group)
                        if meta is not None:
                            ex["metadata"] = meta
                        data_list.append(ex)
                else:
                    ex = dict(video=video, message_list=msgs)
                    if meta is not None:
                        ex["metadata"] = meta
                    data_list.append(ex)
        return data_list

    def get(self, item, rng):
        return self.data[item]


class TGIF(DatasetBase):
    """TGIF — video QA on GIF-sourced clips (action, transition, count, frameqa).

    Annotations auto-downloaded from HuggingFace.
    Videos (GIFs converted to MP4) require manual download and conversion.
    """
    HOME = join(VIDEO_DATA_HOME, "TGIF")
    VIDEO_DIR = join(VIDEO_DATA_HOME, "TGIF", "videos")
    HF_REPO = "allenai/Molmo2-TGIF"
    SUBSETS = ["action", "count", "transition", "frameqa"]

    @classmethod
    def download(cls, n_procs=None):
        if exists(join(cls.HOME, "Train_action_question.csv")):
            return
        snapshot_download(
            repo_id=cls.HF_REPO,
            repo_type="dataset",
            local_dir=cls.HOME,
            max_workers=n_procs or 1,
            ignore_patterns=["videos_*.zip"],
        )

    def __init__(self, split, answer_type="all", subset="all", flat=False):
        assert split in ["train", "test"]
        assert answer_type in ["open_ended", "multi_choice", "all"]
        assert subset in ["all"] + self.SUBSETS
        self.answer_type = answer_type
        self.subset = subset
        self.flat = flat
        self.download()
        if not exists(self.VIDEO_DIR):
            raise FileNotFoundError(
                f"TGIF videos not found at {self.VIDEO_DIR}.\n"
                f"\n"
                f"1. Download GIFs from one of:\n"
                f"   - https://github.com/raingo/TGIF-Release\n"
                f"   - Google Drive: https://drive.google.com/a/vision.snu.ac.kr/file/d/11wdvsTYIPcSTRMVry1tufILiNE4aAMp5/view\n"
                f"   - Dropbox: https://www.dropbox.com/sh/jluwiizm55ugvoz/AABE6ttq5DrrB_5mRrGHaxuAa\n"
                f"   See: https://github.com/YunseokJANG/tgif-qa/blob/master/dataset/README.md\n"
                f"\n"
                f"2. Convert GIFs to MP4 (the dataset expects .mp4 files):\n"
                f"   mkdir -p {self.VIDEO_DIR}\n"
                f"   for f in /path/to/gifs/*.gif; do\n"
                f'       name=$(basename "$f" .gif)\n'
                f'       ffmpeg -i "$f" -movflags faststart -pix_fmt yuv420p \\\n'
                f'           -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \\\n'
                f'           "{self.VIDEO_DIR}/$name.mp4"\n'
                f"   done"
            )
        super().__init__(split)

    def load(self):
        subsets = self.SUBSETS if self.subset == "all" else [self.subset]
        subset2data = {}
        for s in subsets:
            csv_path = join(self.HOME, f"{self.split.capitalize()}_{s}_question.csv")
            subset2data[s] = pd.read_csv(resource_path(csv_path), sep="\t")

        data_list = []
        video2msgs = {}

        for subset, df in subset2data.items():
            for row in df.itertuples(False):
                msgs = []
                question = row.question
                abs_video_path = join(self.VIDEO_DIR, f"{row.gif_name}.mp4")

                if subset in ["action", "transition"]:
                    options = [getattr(row, f'a{i}') for i in range(1, 6)]
                    answer_idx = row.answer
                    if answer_idx >= len(options):
                        raise IndexError(f"Answer index {answer_idx} out of bounds for options {options}")
                    answer = options[answer_idx]

                    if self.answer_type in ["multi_choice", "all"]:
                        mc_msg = dict(
                            question=question, options=options,
                            answer_idx=answer_idx, style="video_multiple_choice",
                        )
                        msgs.append(mc_msg)
                        if self.flat:
                            data_list.append(dict(video=abs_video_path, message_list=[mc_msg]))

                    if self.answer_type in ["open_ended", "all"]:
                        oe_msg = dict(
                            question=question, answer=str(answer),
                            style="video_short_answer",
                        )
                        msgs.append(oe_msg)
                        if self.flat:
                            data_list.append(dict(video=abs_video_path, message_list=[oe_msg]))

                elif subset in ["count", "frameqa"]:
                    if self.answer_type in ["open_ended", "all"]:
                        msg = dict(
                            question=question, answer=str(row.answer),
                            style="video_short_answer",
                        )
                        msgs.append(msg)
                        if self.flat:
                            data_list.append(dict(video=abs_video_path, message_list=[msg]))

                video2msgs.setdefault(abs_video_path, []).extend(msgs)

        if not self.flat:
            for video, msgs in video2msgs.items():
                if not msgs:
                    continue
                data_list.append(dict(video=video, message_list=msgs))
        return data_list

    def get(self, item, rng):
        return self.data[item]


class TVQA(DatasetBase):
    """TVQA dataset — multi-choice QA on TV show clips.

    Annotations auto-downloaded from HuggingFace.
    Video frames require manual download from:
    https://nlp.cs.unc.edu/data/jielei/tvqa/tvqa_public_html/download_tvqa.html
    """
    HOME = join(VIDEO_DATA_HOME, "TVQA") if VIDEO_DATA_HOME else None
    FRAMES_DIR = join(HOME, "video-frames", "frames_hq") if HOME else None
    HF_REPO = "allenai/Molmo2-TVQA"
    DOWNLOAD_URL = "https://nlp.cs.unc.edu/data/jielei/tvqa/tvqa_public_html/download_tvqa.html"

    SHOW_TO_FRAME_DIR = {
        "Grey's Anatomy": "grey_frames",
        "How I Met You Mother": "met_frames",
        "The Big Bang Theory": "bbt_frames",
        "House M.D.": "house_frames",
        "Castle": "castle_frames",
        "Friends": "friends_frames",
    }
    CORRUPT_FILES = {
        "grey_s03e15_seg02_clip_05/video_00001_00051.mp4",
        "castle_s08e08_seg02_clip_14/video_00001_00019.mp4",
    }

    @classmethod
    def download(cls, n_procs=None):
        if cls.HOME is None:
            return
        for fname in ["tvqa_train.jsonl", "tvqa_val.jsonl",
                       "tvqa_test_public.jsonl", "tvqa_preprocessed_subtitles.jsonl",
                       "missing_clips.txt"]:
            if not exists(join(cls.HOME, fname)):
                hf_hub_download(
                    repo_id=cls.HF_REPO, repo_type="dataset",
                    filename=fname, local_dir=cls.HOME,
                )

    def __init__(self, split, flat=False, max_per_video=None, with_subtitle=False):
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        if split == "test":
            split = "test_public"
        self.flat = flat
        self.max_per_video = max_per_video
        self.with_subtitle = with_subtitle
        self.download()
        if not exists(self.FRAMES_DIR):
            raise FileNotFoundError(
                f"TVQA video frames not found at {self.FRAMES_DIR}.\n"
                f"Please download frames from: {self.DOWNLOAD_URL}\n"
                f"Extract to: {self.FRAMES_DIR}/"
            )
        super().__init__(split)

    def get_clip_subtitles(self, subtitle_df, vid_name, start, end):
        """Extract subtitles that overlap with the given time range."""
        if not self.with_subtitle:
            return {}
        sub = subtitle_df.get(vid_name, [])
        if not isinstance(sub, list):
            return {}
        clip_sub = {}
        for el in sub:
            try:
                if not all(key in el for key in ['start', 'end', 'text']):
                    continue
                sub_start, sub_end = float(el['start']), float(el['end'])
                if not (sub_end <= start or sub_start >= end):
                    clip_sub[(sub_start - start, sub_end - start)] = el['text']
            except (ValueError, TypeError, KeyError):
                continue
        return clip_sub

    def load(self):
        json_path = join(self.HOME, f"tvqa_{self.split}.jsonl")
        df = pd.read_json(json_path, lines=True)

        subtitle_df = None
        if self.with_subtitle:
            subtitle_json_path = join(self.HOME, "tvqa_preprocessed_subtitles.jsonl")
            subtitle_df = pd.read_json(subtitle_json_path, lines=True)
            subtitle_df = subtitle_df.set_index("vid_name")
            subtitle_df = subtitle_df['sub'].to_dict()

        # Missing clips cache
        missing_clip_f = join(self.HOME, "missing_clips.txt")
        generate_clips = not exists(missing_clip_f)
        if generate_clips:
            missing_clips = set()
            log.info("TVQA clips will be re-generated")
        else:
            log.info("TVQA clips are pre-built")
            with open(missing_clip_f, 'r') as f:
                missing_clips = set(x.strip() for x in f.read().split("\n") if x.strip())

        # First pass: collect valid rows
        valid_rows = []
        for row in df.itertuples(False):
            start, end = [float(t) for t in row.ts.split("-")]
            if pd.isna(start) or pd.isna(end):
                continue
            valid_rows.append(row)

        data_list = []
        video2msgs = {}
        video2meta = {}
        corrupt = 0

        for row in valid_rows:
            frames_dir = join(
                self.FRAMES_DIR,
                self.SHOW_TO_FRAME_DIR[row.show_name],
                row.vid_name,
            )
            start, end = [float(t) for t in row.ts.split("-")]
            clip_key = f"{Path(frames_dir).parent.name}/{Path(frames_dir).name}:{row.ts}"

            if generate_clips:
                try:
                    fps = 3
                    start_frame = int(start * fps) + 1
                    end_frame = int(end * fps)
                    abs_video_path = _create_video_from_frame_range(
                        frames_dir, start_frame, end_frame, fps
                    )
                except Exception as e:
                    abs_video_path = None
                    missing_clips.add(clip_key)
            else:
                if clip_key in missing_clips:
                    abs_video_path = None
                else:
                    fps = 3
                    start_frame = int(start * fps) + 1
                    end_frame = int(end * fps)
                    abs_video_path = os.path.join(
                        frames_dir, f"video_{start_frame:05d}_{end_frame:05d}.mp4"
                    )

            if abs_video_path is None:
                corrupt += 1
                continue

            video_key = f"{Path(abs_video_path).parent.name}/{Path(abs_video_path).name}"
            if video_key in self.CORRUPT_FILES:
                corrupt += 1
                continue

            video2msgs.setdefault(abs_video_path, [])
            video2meta.setdefault(abs_video_path, {})

            question = row.q
            options = [getattr(row, f'a{i}') for i in range(5)]
            answer_idx = row.answer_idx
            if answer_idx >= len(options):
                raise IndexError(f"Answer index {answer_idx} out of bounds for options {options}")

            style = "video_multiple_choice"
            clip_sub = {}
            if self.with_subtitle and subtitle_df is not None:
                clip_sub = self.get_clip_subtitles(subtitle_df, row.vid_name, start, end)
                if clip_sub:
                    style = "video_multiple_choice_w_subtitle"

            msg = dict(
                question=question,
                answer_idx=answer_idx,
                options=options,
                style=style,
            )

            if self.flat:
                formatted_ex = {
                    "video": abs_video_path,
                    "metadata": {"show_name": row.show_name, "ts": row.ts},
                    "message_list": [msg],
                }
                if clip_sub:
                    formatted_ex["subtitle"] = clip_sub
                data_list.append(formatted_ex)
            else:
                video2msgs[abs_video_path].append(msg)
                video2meta[abs_video_path].update({
                    "show_name": row.show_name,
                    "ts": row.ts,
                })
                if clip_sub:
                    video2meta[abs_video_path]["subtitle"] = clip_sub

        if get_global_rank() == 0:
            log.warning(f"Skipped {corrupt} corrupt TVQA videos.")

        if not self.flat:
            for video, msgs in video2msgs.items():
                if not msgs:
                    continue
                subtitle = video2meta[video].get("subtitle", None)
                if self.max_per_video:
                    for group in split_into_groups(msgs, self.max_per_video):
                        formatted_ex = {
                            "video": video,
                            "metadata": video2meta[video],
                            "message_list": group,
                        }
                        if subtitle:
                            formatted_ex["subtitle"] = subtitle
                        data_list.append(formatted_ex)
                else:
                    formatted_ex = {
                        "video": video,
                        "metadata": video2meta[video],
                        "message_list": msgs,
                    }
                    if subtitle:
                        formatted_ex["subtitle"] = subtitle
                    data_list.append(formatted_ex)

        if generate_clips and get_global_rank() == 0:
            log.info("Caching missing clips data")
            with open(missing_clip_f, 'w') as f:
                f.write("\n".join(missing_clips))

        return data_list

    def get(self, item, rng):
        return self.data[item]