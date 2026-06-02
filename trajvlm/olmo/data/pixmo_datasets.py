import logging
import os
from os.path import join, exists, relpath
import logging
import re
from os.path import join, exists

import datasets
import numpy as np
from datasets import DatasetDict

from olmo.data.dataset import DATA_HOME, Dataset
from olmo.data.dataset_builders.download_urls import download_pixmo_urls, filter_and_group_data
from olmo.data.utils import save_local_dataset
from olmo.io import file_exists
from olmo.preprocessing.detect_counting_question import is_pixmo_point_and_count_question
from olmo.util import flatten_lists, resource_path

if DATA_HOME is not None:
    PIXMO_DATASETS = join(DATA_HOME, "pixmo_datasets")
else:
    PIXMO_DATASETS = None
"""Where to save local version of the data after URLs filtering"""

if "PIXMO_IMAGE_DIR" in os.environ:
    PIXMO_IMAGES = os.environ["PIXMO_IMAGE_DIR"]
elif DATA_HOME is not None:
    PIXMO_IMAGES = join(DATA_HOME, "pixmo_images")
else:
    PIXMO_IMAGES = None
"""Where to save downloaded images"""


VERIFY = True
"""Verify SSL certificates when downloading"""

NO_POINT_PREFIX = [
    "No pointing: ",
    "No pointing: ",
    "no pointing:\n",
    "No pointing:\n",
    "Not pointing:\n",
    "No Points: ",
    "No Points: ",
    "NO POINTING\n",
    "No pontiing\n",
    "No Points:\n ",
    "No pointing\n",
    "Do not point. ",
    "Refrain from pointing. ",
    "Avoid generating points . ",
    "For this question, do not use points. ",
    "Refrain from using points:\n",
    "Don't include points in your response. ",
    "Don't point. ",
    "Don't use points. ",
    "Please don't use points.\n\n",
    "Please don't use points.\n\n",
    "Respond without using points. ",
    "Respond without pointing:\n",
    "Do not generate ponits: ",
    "Do not point. ",
    "Do not point\n",
    "no pointing\n\n",
    "Answer without points: ",
    "Answer this question without pointing: ",
    "Answer without poiints. ",
    "answer without points: ",
    "answer with text only, do not points\n"
]
"""No-pointing requests templates, used for preprocessing"""


class PixMoCount(Dataset):
    @classmethod
    def download(cls, n_procs=1, check_sha=False, n_val=1024, cache_only=False):
        local_name = join(PIXMO_DATASETS, "count")
        if exists(local_name):
            return
        all_data = datasets.DatasetDict()
        for split in ["validation", "test", "train"]:
            ds = datasets.load_dataset("allenai/pixmo-count", split=split)
            url_to_filename = download_pixmo_urls(ds, n_procs, output_dir=PIXMO_IMAGES, check_sha=check_sha, cache_only=cache_only, verify=False)
            ds = ds.filter(lambda x: x in url_to_filename, input_columns=["image_url"])
            ds = ds.add_column("image", [url_to_filename[x] for x in ds["image_url"]])
            all_data[split] = ds
        save_local_dataset(all_data, local_name, n_procs)

    def __init__(self, split, sample=None, counting=False, keep_in_memory=False):
        self.dataset = datasets.load_from_disk(join(PIXMO_DATASETS, "count"), keep_in_memory=keep_in_memory)[split]
        self.counting = counting
        self.split = split

    def __len__(self):
        if self.counting == "both":
            return len(self.dataset) * 2
        else:
            return len(self.dataset)

    def get(self, item, rng):
        if self.counting == "both":
            mode = "point_count" if (item%2==0) else "pointing"
            item = item // 2
        else:
            mode = "point_count" if self.counting else "pointing"

        example = self.dataset[item]
        out = dict(
            style=mode,
            image=example["image"],
            label=example["label"],
            metadata=dict(
                image_url=example["image_url"],
                count=example["count"],
            )
        )
        if self.split == "train":
            points = example["points"]
            out["points"] = np.stack([points["x"], points["y"]], -1, dtype=np.float32)
        return out


class PixMoPoints(Dataset):

    @classmethod
    def download(cls, n_procs=1, check_sha=True, n_val=2048, cache_only=False, hold_out_pointing_eval=True):
        collection_method = ["pointing", "counting"]
        local_names = [join(PIXMO_DATASETS, f"points-{name}") for name in collection_method]
        if all(exists(x) for x in local_names):
            return
        ds = datasets.load_dataset("allenai/pixmo-points", split="train")
        filenames = download_pixmo_urls(ds, n_procs, output_dir=PIXMO_IMAGES, check_sha=check_sha, cache_only=cache_only, verify=VERIFY)
        if hold_out_pointing_eval:
            eval_ds = datasets.load_dataset("allenai/pixmo-points-eval", split="test")
            for url in eval_ds["image_url"]:
                if url in filenames:
                    del filenames[url]
        for method, local_name in zip(collection_method, local_names):
            logging.info(f"Building subset {method}")
            ds_for_method = ds.filter(lambda x: x == method, input_columns="collection_method")
            filtered_dataset = filter_and_group_data(ds_for_method, filenames, check_sha)
            name = "high_frequency" if method == "counting" else "basic"
            save_local_dataset(filtered_dataset, local_name, n_procs=n_procs, n_val=n_val)

    def __init__(self, split, kind="both", counting=False, keep_in_memory=False,
                 max_points=None, max_total_points_per_example=None):
        if kind not in ["high_frequency", "basic", "both"]:
            raise ValueError(kind)
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        self.counting = counting
        if counting == "both":
            self.mode = ["point_count", "pointing"]
        else:
            self.mode = "point_count" if counting else "pointing"
        self.split = split
        self.kind = kind
        if kind == "both":
            data1 = datasets.load_from_disk(
                join(PIXMO_DATASETS, "points-counting"), keep_in_memory=keep_in_memory)[split]
            data2 = datasets.load_from_disk(
                join(PIXMO_DATASETS, "points-pointing"), keep_in_memory=keep_in_memory)[split]
            self.data = datasets.concatenate_datasets([data1, data2])
        elif kind == "basic":
            self.data = datasets.load_from_disk(
                join(PIXMO_DATASETS, f"points-pointing"), keep_in_memory=keep_in_memory)[split]
        else:
            self.data = datasets.load_from_disk(
                join(PIXMO_DATASETS, f"points-counting"), keep_in_memory=keep_in_memory)[split]
        if max_total_points_per_example or max_points:
            n_points = self.data["count"][:]
            sub_index = []
            total_points = 0
            n_filtered = 0
            for image_idx, point_counts in enumerate(n_points):
                sub_batches = []
                on = []
                total_on = 0
                total_points += len(point_counts)
                for ix, n in enumerate(point_counts):
                    if max_points and n > max_points:
                        n_filtered += 1
                        continue
                    if max_total_points_per_example and (total_on + n > max_total_points_per_example):
                        if on:
                            sub_batches.append(on)
                            total_on = 0
                            on = []
                    on.append(ix)
                    total_on += n
                if on:
                    sub_batches.append(on)
                for ix in sub_batches:
                    sub_index.append((image_idx, ix))
            logging.info(f"Filtered {n_filtered} ({n_filtered}/{total_points}) points")
            logging.info(f"Split {len(self.data)} examples into {len(sub_index)} parts")
            self.sub_index = sub_index
        else:
            self.sub_index = None

    def __len__(self):
        n = len(self.sub_index) if self.sub_index else len(self.data)
        if self.counting == "both":
            n *= 2
        return n

    def get(self, item, rng):
        if self.counting == "both":
            mode = self.mode[item % 2]
            item = item // 2
        else:
            mode = self.mode

        if self.sub_index:
            image_idx, point_idx = self.sub_index[item]
            ex = dict(self.data[image_idx])
            ex["label"] = [ex["label"][i] for i in point_idx]
            ex["points"] = [ex["points"][i] for i in point_idx]
        else:
            ex = self.data[item]

        messages = []
        for label, points in zip(ex["label"], ex["points"]):
            messages.append(dict(
                label=label,
                points=np.stack([[x["x"] for x in points], [x["y"] for x in points]], -1),
                point_scale=100,
                clip_points=True,
                style=mode
            ))
        return dict(
            image=ex["image"],
            message_list=messages,
            metadata=dict(
                image_url=ex["image_url"],
            )
        )


class PixMoPointExplanations(Dataset):

    @classmethod
    def download(cls, n_procs=1, check_sha=True, n_val=1024, cache_only=False):
        local_name = join(PIXMO_DATASETS, "point-explanations")
        if exists(local_name):
            return
        ds = datasets.load_dataset("allenai/pixmo-point-explanations", split="train")
        ds = ds.filter(lambda x: x is not None, input_columns=["parsed_response"])
        filenames = download_pixmo_urls(ds, n_procs, output_dir=PIXMO_IMAGES, check_sha=check_sha, cache_only=cache_only, verify=VERIFY)
        filtered_dataset = filter_and_group_data(ds, filenames, check_sha)
        save_local_dataset(filtered_dataset, local_name, n_procs, n_val=n_val)

    def __init__(self, split, split_groups=True, keep_in_memory=False):
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        self.split = split
        self.split_groups = split_groups
        data = datasets.load_from_disk(
            join(PIXMO_DATASETS, "point-explanations"),
            keep_in_memory=keep_in_memory)[split]
        out = []
        for ex in data:
            molmo_ex = dict(
                image=ex["image"],
                metadata=dict(
                    image_url=ex["image_url"],
                )
            )
            msg_list = []
            for q, res, alt, inline, points in zip(
                ex["question"], ex["parsed_response"],
                ex["alt_text"], ex["inline_text"], ex["points"]
            ):
                msg_list.append(dict(
                    question=q,
                    answer=res,
                    answer_annotations=[dict(
                        points=p, inline_text=i, alt_text=a
                    ) for p, i, a in zip(points, inline, alt)],
                    style="point_qa"
                ))
            if self.split_groups and len(msg_list) > 1:
                n = len(msg_list) // 2 + len(msg_list) % 2
                out.append(dict(molmo_ex, message_list=msg_list[:n]))
                out.append(dict(molmo_ex, message_list=msg_list[n:]))
            else:
                out.append(dict(molmo_ex, message_list=msg_list))
        self.data = out

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        return dict(self.data[item])


class PixMoCapQa(Dataset):
    @classmethod
    def download(cls, n_procs=1, check_sha=False, n_val=2048, cache_only=False):
        local_name = join(PIXMO_DATASETS, "cap-qa")
        if exists(local_name):
            return
        ds = datasets.load_dataset("allenai/pixmo-cap-qa", split="train")
        filenames = download_pixmo_urls(ds, n_procs, output_dir=PIXMO_IMAGES, check_sha=check_sha, cache_only=cache_only, verify=VERIFY)
        filtered_dataset = filter_and_group_data(ds, filenames, check_sha)
        save_local_dataset(filtered_dataset, local_name, n_procs, n_val=n_val)

    def __init__(self, split, prefix_how_many=True, keep_in_memory=False, style="synthetic_qa"):
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        self.split = split
        self.prefix_how_many = prefix_how_many
        self.data = datasets.load_from_disk(
            join(PIXMO_DATASETS, "cap-qa"), keep_in_memory=keep_in_memory)[split]
        self.style = style

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        example = self.data[item]
        question = example["question"]
        answer = example["answer"]
        message_lists = []
        for qs, ans in zip(question, answer):
            parts = re.split(r"\s*(\[USER\]|\[ASSISTANT\])\s*", qs)
            assert parts[0] == ""
            assert parts[-1] == ""
            parts = parts[1:-1]
            assert len(parts) % 4 == 3
            messages = []
            for part_ix, part in enumerate(parts):
                if part_ix % 4 == 0:
                    assert part == "[USER]"
                elif part_ix % 4 == 1:
                    assert part
                    messages.append(part)
                elif part_ix % 4 == 2:
                    assert part == "[ASSISTANT]"
                else:
                    assert part
                    messages.append(part)
            messages.append(ans)
            message_lists.append(dict(messages=messages, style=self.style))

        example = dict(
            image=example["image"],
            message_list=message_lists,
            metadata=dict(
                image_url=example["image_url"],
            )
        )
        if self.prefix_how_many:
            for conv in example["message_list"]:
                messages = conv["messages"]
                for user_question_ix in range(0, len(messages), 2):
                    question = messages[user_question_ix]
                    answer = messages[user_question_ix+1]
                    if is_pixmo_point_and_count_question(question):
                        prefix = NO_POINT_PREFIX[rng.randint(0, len(NO_POINT_PREFIX))]
                        messages[user_question_ix] = prefix + messages[user_question_ix]
        return example


class PixMoCap(Dataset):
    @classmethod
    def download(cls, n_procs=1, check_sha=False, n_val=2048, cache_only=False, sample=None):
        local_name = join(PIXMO_DATASETS, "cap")
        if exists(local_name):
            return
        ds = datasets.load_dataset("allenai/pixmo-cap", split="train")
        if sample:
            ds = ds.take(sample)
        url_to_filename = download_pixmo_urls(ds, n_procs, output_dir=PIXMO_IMAGES, check_sha=check_sha, cache_only=cache_only, verify=VERIFY)
        logging.info("Preparing data...")
        filtered_dataset = ds.filter(lambda x: x in url_to_filename, input_columns=["image_url"])
        filtered_dataset = filtered_dataset.add_column(
            "image", [url_to_filename[x] for x in filtered_dataset["image_url"]])
        save_local_dataset(filtered_dataset, local_name, n_procs, n_val=n_val)

    def __init__(self, split, mode, prefix_how_many=True, keep_in_memory=False, flatten=False):
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        if mode not in ["transcript", "transcripts", "captions", "transcript_and_caption", "transcript1_and_caption"]:
            raise ValueError(mode)
        self.split = split
        self.mode = mode
        self.data = datasets.load_from_disk(
            join(PIXMO_DATASETS, "cap"), keep_in_memory=keep_in_memory)[split]

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        messages = []
        caption = ex.pop("caption")
        transcripts = ex.pop("transcripts")
        if self.mode in ["captions", "transcript_and_caption", "transcript1_and_caption"]:
            messages.append(dict(text=caption, style="long_caption"))
        if self.mode in ["transcript_and_caption", "transcript1_and_caption", "transcript"]:
            if self.mode == "transcript_and_caption":
                ix = rng.randint(0, len(transcripts))
            else:
                ix = 0
            messages.append(dict(text=transcripts[ix], style="transcript"))
        if self.mode == "transcripts":
            messages += [dict(text=tr, style="transcript") for tr in transcripts]
        out = dict(
            image=ex["image"],
            message_list=messages,
            metadata=dict(
                image_path=ex["image"],
                image_url=ex.pop("image_url"),
            )
        )
        return out


class PixMoAskModelAnything(Dataset):
    @classmethod
    def download(cls, n_procs=1, check_sha=True, n_val=2048, cache_only=False):
        local_name = join(PIXMO_DATASETS, "ask-model-anything")
        if exists(local_name):
            return
        ds = datasets.load_dataset("allenai/pixmo-ask-model-anything", split="train")
        filenames = download_pixmo_urls(ds, n_procs, output_dir=PIXMO_IMAGES, check_sha=check_sha, cache_only=cache_only, verify=VERIFY)
        filtered_dataset = filter_and_group_data(ds, filenames, check_sha)
        save_local_dataset(filtered_dataset, local_name, n_procs, n_val=n_val)

    def __init__(self, split, prefix_how_many=True, keep_in_memory=False, flat=False,
                 skip_counting=False, sample=None):
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        self.skip_counting = skip_counting
        self.split = split
        self.prefix_how_many = prefix_how_many
        self.flat = flat
        self.data = datasets.load_from_disk(
            join(PIXMO_DATASETS, "ask-model-anything"), keep_in_memory=keep_in_memory)[split]
        if self.flat:
            all_questions = self.data["question"]
            n_questions = [len(x) for x in all_questions]
            image_index = np.repeat(np.arange(len(self.data), dtype=np.int32), n_questions)
            question_index = np.concatenate([np.arange(x, dtype=np.int32) for x in n_questions], 0)
            self.flat_index = np.stack([image_index, question_index], 1)
            if self.skip_counting:
                is_counting = flatten_lists(
                    [re.fullmatch("how many.*", q.strip(), flags=re.IGNORECASE) is not None for q in questions]
                    for questions in all_questions)
                assert len(is_counting) == len(self.flat_index)
                self.flat_index = self.flat_index[~np.array(is_counting)]
            if sample:
                np.random.RandomState(872).shuffle(self.flat_index)
                self.flat_index = self.flat_index[:sample]
        else:
            if skip_counting or sample:
                raise NotImplementedError()

    def __len__(self):
        return len(self.flat_index) if self.flat else len(self.data)

    def get(self, item, rng):
        if self.flat:
            item, question_ix = self.flat_index[item]
            example = self.data[int(item)]
            q = example["question"][question_ix].strip()
            a = example["answer"][question_ix]
            metadata = dict(question=q, answer=a, image_file=example["image"])
            messages = [dict(question=q, answer=a, style="user_qa")]
        else:
            question_id = None
            example = self.data[item]
            messages = []
            for q, a in zip(example["question"], example["answer"]):
                messages.append(dict(question=q.strip(), answer=a, style="user_qa"))
            metadata = dict(image_url=example["image_url"])

        ex = dict(
            image=example["image"],
            message_list=messages,
            metadata=metadata
        )

        if self.prefix_how_many:
            for conv in ex["message_list"]:
                if is_pixmo_point_and_count_question(conv["question"]):
                    prefix = NO_POINT_PREFIX[rng.randint(0, len(NO_POINT_PREFIX))]
                    conv["question"] = prefix + conv["question"]
        return ex


class PixMoPointsEval(Dataset):
    # path = join(PIXMO_DATASETS, "pixmo-points-eval")
    path = "/data/chrisc/pixmo-points-eval-dbg"

    @classmethod
    def download(cls, n_procs=1, check_sha=True, cache_only=False):
        local_name = cls.path
        if exists(local_name):
            return
        ds = datasets.load_dataset("allenai/pixmo-points-eval", split="test")
        url_to_filename = download_pixmo_urls(ds, n_procs, output_dir=PIXMO_IMAGES, check_sha=check_sha, cache_only=cache_only, verify=VERIFY)
        ds = ds.filter(lambda x: x in url_to_filename, input_columns=["image_url"])
        ds = ds.add_column("image", [url_to_filename[x] for x in ds["image_url"]])
        save_local_dataset(ds, local_name, n_procs)

    def __init__(self, keep_in_memory=False):
        self.data = datasets.load_from_disk(self.path, keep_in_memory=keep_in_memory)

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        points = ex["points"]
        messages = []
        points = np.stack([[x["x"] for x in points], [x["y"] for x in points]], -1)
        mask = np.array(ex["masks"], dtype=bool)
        h, w = mask.shape[1:]
        gt_points = points * np.array([w, h])[None, :]/100

        return dict(
            image=ex["image"],
            label=ex["label"],
            points=points,
            point_scale=100,
            style="pointing",
            metadata=dict(
                label=ex["label"],
                masks=mask,
                image_url=ex["image_url"],
                gt_points=gt_points
            )
        )


class PixMoMultiPoints(Dataset):
    MULTI_IMAGE_POINTING_STYLES = [
        "multi_image_pointing",
        "multi_image_point_then_count",
    ]
    HOME = join(PIXMO_DATASETS, "pixmo-multi-points")

    @classmethod
    def download(cls, n_procs=1):
        if exists(cls.HOME):
            return
        PixMoPoints.download(n_procs=n_procs)
        dataset = datasets.load_dataset("allenai/molmo2-pixmo-multi-points")

        # Save a local copy with only images that were found downloaded
        def _check_exists(_fnames, cache={}):
            for _fname in _fnames:
                if _fname not in cache:
                    cache[_fname] = file_exists(join(DATA_HOME, _fname))
                if not cache[_fname]:
                    return False
            return True

        dataset = dataset.filter(
            _check_exists,
            input_columns=["images"],
            num_proc=n_procs
        )
        save_local_dataset(dataset, cls.HOME, n_procs)

    def __init__(self, split, keep_in_memory=False,
                 styles=("multi_image_pointing", "multi_image_point_then_count")):
        assert split in ["train", "validation"]
        assert all(x in self.MULTI_IMAGE_POINTING_STYLES for x in styles)
        self.styles = styles
        self.split = split
        self.dataset = datasets.load_from_disk(
            self.HOME, keep_in_memory=keep_in_memory)[split]

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        example = dict(self.dataset[item])
        images = example["images"]
        return dict(
            example,
            style=rng.choice(self.styles),
            image=images,
            point_scale=100,
            clip_points=True,
            metadata=dict(image_paths=images),
        )


class PixmoMultiImageQa(Dataset):
    home = join(PIXMO_DATASETS, "pixmo-multi-image-qa")

    @classmethod
    def download(cls, n_procs=1, check_sha=False, cache_only=False):
        if exists(cls.home):
            return
        dataset = DatasetDict()
        for split in ["train", "validation"]:
            ds = datasets.load_dataset("allenai/Molmo2-MultiImageQA", split=split)
            url_to_filename = download_pixmo_urls(ds, n_procs, output_dir=PIXMO_IMAGES, check_sha=check_sha, cache_only=cache_only, verify=False)
            ds = ds.filter(lambda x: all(u in url_to_filename for u in x), input_columns=["image_urls"])
            ds = ds.add_column("image", [
                [relpath(url_to_filename[u], PIXMO_IMAGES) for u in x]
                for x in ds["image_urls"]])
            dataset[split] = ds
        save_local_dataset(dataset, cls.home, n_procs)

    def __init__(self, split, multi_image_only=False, max_images=None, prefix_how_many=True):
        assert split in ["train", "validation"]
        self.split = split
        self.prefix_how_many = prefix_how_many
        self.max_images = max_images
        self.multi_image_only = multi_image_only
        self.dataset = datasets.load_from_disk(self.home, keep_in_memory=False)[split]
        if self.max_images is not None:
            if multi_image_only:
                self.dataset = self.dataset.filter(lambda x: 2 <= len(x) <= max_images, input_columns=["image_urls"])
            else:
                self.dataset = self.dataset.filter(lambda x: len(x) <= max_images, input_columns=["image_urls"])
        else:
            self.dataset = self.dataset.filter(lambda x: 2 <= len(x), input_columns=["image_urls"])

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng: np.random.RandomState):
        example = self.dataset[item]
        image = [join(PIXMO_IMAGES, x) for x in example["image"]]
        qa_pairs = example["qa_pairs"]
        messages = []
        for q, a in zip(qa_pairs["question"], qa_pairs["answer"]):
            if self.prefix_how_many:
                if is_pixmo_point_and_count_question(q):
                    prefix = NO_POINT_PREFIX[rng.randint(0, len(NO_POINT_PREFIX))]
                    q = prefix + q
            messages.append(dict(question=q, answer=a, style="correction_qa"))
        return dict(
            image=image,
            message_list=messages,
        )
