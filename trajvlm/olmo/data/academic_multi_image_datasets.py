import logging
import re
import string
import zipfile
from os.path import join, exists
from pathlib import Path
from typing import List
import numpy as np

import datasets
from huggingface_hub import snapshot_download

from olmo.data.dataset import Dataset, DATA_HOME, HfDataset


def replace_images(question, options, max_images=None):
    all_strings = [question] + options
    image_counter = 1

    total_images = sum(s.count("<image>") for s in all_strings)
    if max_images is not None:
        total_images = min(total_images, max_images)

    replaced = []

    for s in all_strings:
        def repl(match):
            nonlocal image_counter
            if image_counter > total_images:
                return match.group(0)
            replacement = f"Image {image_counter}"
            image_counter += 1
            return replacement

        replaced.append(re.sub(r"<image>", repl, s))

    return replaced[0], replaced[1:]


class MuirBench(Dataset):
    """
    This class loads the MuirBench dataset from HuggingFace (https://huggingface.co/datasets/MUIRBENCH/MUIRBENCH).
    VQA questions that each involve 2-9 images.
    """
    PATH = "MUIRBENCH/MUIRBENCH"

    def __init__(self, split: str, format: str = "multiple_choice", keep_in_memory=False):
        self.format = format
        self.dataset = datasets.load_dataset(self.PATH, keep_in_memory=keep_in_memory)[split]

    def qo_template(self, question, options, format: str, sep: str = "."):
        question, options = replace_images(question, options)
        option_text = "\n".join(
            f"{chr(ord('A') + idx)}{sep} {options[idx]}" for idx in range(len(options))
        )
        if format == "answer_first":
            question += " Choose the correct option and then explain your reasoning."
        elif format == "answer_last":
            question += " Explain your reasoning and then choose the correct option."
        prompts = [question, option_text]
        if format == "short_answer":
            prompts.append("Select the correct answer from the options above.")
        prompt = "\n".join(prompts)
        return question, prompt, options

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        example = self.dataset[item]
        question, prompt, options = self.qo_template(example['question'], example['options'], self.format)
        out = dict(
            image=example["image_list"],
            metadata=dict(
                example_id=example["idx"],
                task=example["task"],
                image_relation=example["image_relation"],
                image_type=example["image_type"],
                counterpart_id=example["counterpart_idx"],
            )
        )

        answer_idx = ord(example["answer"]) - ord("A")
        if self.format == "multiple_choice":
            out.update(
                question=question,
                options=options,
                answer_idx=answer_idx,
                style="eval_multiple_choice" if self.legacy else "mantis_instruct_mc",
            )
        else:
            out.update(
                question=prompt,
                answer=example["answer"],
                style=f"eval_multi_image_{self.format}",
            )
            out["metadata"]["options"] = options
            out["metadata"]["answer_idx"] = answer_idx

        return out


class MMIU(HfDataset):
    """
    MMIU (Multimodal Multi-image Understanding) benchmark
    7 types of multi-image relationships, 52 tasks, 77K images, and 11K meticulously curated multiple-choice questions
    VQA questions that each involve 1-62 images.
    Hugging Face repo: https://huggingface.co/datasets/FanqingM/MMIU-Benchmark
    Paper: arXiv 2408.027187
    """
    PATH = "FanqingM/MMIU-Benchmark"
    HOME = join(DATA_HOME, "academic_datasets", "tabwmp")

    # List of image ZIP files in the MMIU repository
    IMAGE_ZIP_FILES = [
        '2D-spatial.zip',
        '3D-spatial.zip',
        'Continuous-temporal.zip',
        'Discrete-temporal.zip',
        'High-level-obj-semantic.zip',
        'High-level-sub-semantic.zip',
        'Low-level-semantic.zip'
    ]

    @classmethod
    def download(cls, n_procs=1):
        local_name = cls.HOME
        if exists(local_name):
            return
        from huggingface_hub import hf_hub_download
        import zipfile
        from pathlib import Path

        # Download and unzip the image files
        logging.info("Downloading MMIU images...")
        for zip_file in cls.IMAGE_ZIP_FILES:
            local_zip_file = hf_hub_download(
                repo_id=cls.PATH,
                repo_type="dataset",
                filename=zip_file,
                revision="main",
                local_dir=join(DATA_HOME, "mmiu"),
                local_dir_use_symlinks=False,
            )
            extract_dir = join(DATA_HOME, "mmiu", zip_file.replace(".zip", ""))
            Path(extract_dir).mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(local_zip_file, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            Path(local_zip_file).unlink()

        datasets.load_dataset_builder(cls.PATH).download_and_prepare()

    def __init__(self, split: str, format: str = "multiple_choice", legacy=False, keep_in_memory=False):
        assert split in ["test"]
        self.format = format
        self.legacy = legacy
        super().__init__(split, keep_in_memory=keep_in_memory)

    def question_template(self, question, options, format: str, sep: str = "."):
        option_text = "\n".join(
            f"{chr(ord('A') + idx)}{sep} {options[idx]}" for idx in range(len(options))
        )
        if format == "answer_first":
            question += " Choose the correct option and then explain your reasoning."
        elif format == "answer_last":
            question += " Explain your reasoning and then choose the correct option."
        prompts = [question, option_text]
        if format == "short_answer":
            prompts.append("Select the correct answer from the options above.")
        prompt = "\n".join(prompts)
        return question, prompt

    def extract_options(self, option_string: str):
        matches = []
        for ix, (_, letter, answer) in enumerate(re.findall(r'(^|\n)([A-Z]):\s?([^\n]+)', option_string, flags=re.DOTALL | re.MULTILINE)):
            assert letter == string.ascii_uppercase[ix]
            matches.append(answer)
        return matches

    def get(self, item, rng):
        example_id = str(item)
        ex = self.dataset[item]
        images = [join(DATA_HOME, "mmiu", img[len("./"):]) for img in ex["input_image_path"]]
        relationship = list(set([img.split("/")[1] for img in ex["input_image_path"]]))
        assert len(relationship) == 1, "It should only have one relationship"
        relationship = relationship[0]
        options = self.extract_options(ex["options"])
        question, prompt = self.question_template(ex["question"], options, self.format)

        # Note the ex["options"] is sometimes an option not listed in ex["options"], for
        # example the output will be G but the options will only have A-D
        # Therefore we can't get a true ground-truth answer_idx reliably, although it is not
        # needed given this is an eval set
        # answer_idx = ord(ex["output"]) - ord("A")
        # if len(options) <= answer_idx:
        #     raise ValueError()

        format = "mc" if self.format == "multiple_choice" else self.format

        out = dict(
            image=images,
            question=prompt if self.legacy else question,
            answer=ex["output"],
            style=f"eval_multi_image_{format}" if self.legacy else f"mantis_instruct_{format}",
            metadata=dict(
                example_id=example_id,
                task=ex["task"],
                relationship=relationship,
                context=ex["context"],
                visual_input_component=ex["visual_input_component"],
                source=ex["source"],
                num_images=len(images),
            )
        )

        if self.legacy:
            out["metadata"]["options"] = options
        else:
            out["options"] = options
            out["content_in_mc"] = False
        return out


class BLINK(Dataset):
    """1-4 images per question"""
    NAMES = [
        'Art_Style', 'Functional_Correspondence', 'Multi-view_Reasoning',
        'Relative_Reflectance', 'Visual_Correspondence', 'Counting',
        'IQ_Test', 'Object_Localization', 'Semantic_Correspondence',
        'Visual_Similarity', 'Forensic_Detection', 'Jigsaw',
        'Relative_Depth', 'Spatial_Relation',
    ]

    @classmethod
    def download(cls, n_procs=1):
        for name in cls.NAMES:
            builder = datasets.load_dataset_builder("BLINK-Benchmark/BLINK", name=name)
            builder.download_and_prepare()

    def __init__(self, split: str):
        split = "val" if split == "validation" else split
        all_parts = []
        for name in self.NAMES:
            all_parts.append(datasets.load_dataset("BLINK-Benchmark/BLINK", name=name, split=split,
                                                   keep_in_memory=True))
        self.data = datasets.concatenate_datasets(all_parts)

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]

        images = [ex[f"image_{i}"] for i in range(1, 5) if ex[f"image_{i}"] is not None]
        if len(images) > 1:
            style = "mantis_instruct_mc"
        else:
            style = "eval_multiple_choice"

        answer = ex["answer"].replace("(", "").replace(")", "")

        out = dict(
            image=images if len(images) > 1 else images[0],
            question=ex["prompt"],
            answer=answer,
            style=style,
            metadata=dict(
                options=ex["choices"],
                answer_idx=ord(answer) - ord("A"),
                example_id=ex["idx"],
                sub_task=ex["sub_task"],
            )
        )
        return out


class MantisInstruct(Dataset):
    HOME = join(DATA_HOME, "mantis-instruct")
    NAMES = [
        "nlvr2",
        "llava_665k_multi",
        "spot-the-diff",
        "nextqa",
        "star",
    ]
    TRAIN_ONLY = [
        "llava_665k_multi",
        "spot-the-diff",
        "nextqa",
        "star",
    ]
    SPLITS = ["train", "validation"]
    PATH = "TIGER-Lab/Mantis-Instruct"

    @classmethod
    def download(cls, n_procs=1):
        for name in cls.NAMES:
            local_name = join(cls.HOME, name)
            if exists(local_name):
                continue

            # Download our pre-processed data
            datasets.load_dataset_builder(f"allenai/molmo2-mantis-instruct-{name}").download_and_prepare()

            # Download the dataset
            logging.info(f"Downloading Mantis-Instruct, {name} to {local_name}...")
            snapshot_download(
                repo_id=cls.PATH,
                repo_type="dataset",
                revision="main",
                local_dir=cls.HOME,
                local_dir_use_symlinks=False,
                allow_patterns=[f"{name}/*", f"{name}/**"],
            )

            splits = ["train"] if name in cls.TRAIN_ONLY else ["train", "val"]
            for split in splits:
                # Unzip the images and remove the zip file
                zip_path = join(DATA_HOME, "mantis-instruct", name, f"{split}_images.zip")
                extract_dir = join(DATA_HOME, "mantis-instruct", name, f"{split}_images")
                if exists(extract_dir):
                    continue
                logging.info(f"Unzipping Mantis-Instruct, {name} images...")
                Path(extract_dir).mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
                Path(zip_path).unlink()

    def __init__(self, name: str, split: str, direct_answer=False, multi_image_only=False, flat=False, sample=None, keep_in_memory=False):
        assert split in self.SPLITS
        self.split = split
        self.direct_answer = direct_answer
        self.style = "mantis_instruct_" + ("da" if direct_answer else "mc")
        self.data = datasets.load_dataset(
            f"allenai/molmo2-mantis-instruct-{name}", keep_in_memory=keep_in_memory
        )[split]
        if multi_image_only:
            self.data = self.data.filter(lambda images: len(images) > 1, input_columns="images")
        self.flat = flat
        if flat:
            flattened_data = []
            for item in self.data:
                for i in range(len(item["mc_question"])):
                    flattened_data.append(dict(
                        subset=item["subset"],
                        example_id=f"{item['example_id']}-{i:03d}",
                        images=item["images"],
                        mc_question=item["mc_question"][i],
                        oe_question=item["oe_question"][i],
                        direct_answer=item["direct_answer"][i],
                        choices=item["choices"][i],
                        correct_choice_idx=item["correct_choice_idx"][i],
                    ))
            if sample:
                logging.info(f"Sampling {sample} of {len(flattened_data)} ({100*sample/len(flattened_data)}:0.1f)")
                np.random.RandomState(9123).shuffle(flattened_data)
                flattened_data = flattened_data[:sample]
            self.data = flattened_data
        else:
            assert sample is None

    def __len__(self):
        return len(self.data)

    def shuffle_options(self, options: List[str], answer_idx: int, rng: np.random.RandomState):
        perm = rng.permutation(len(options))
        shuffled_options = [options[i] for i in perm]

        inverse_perm = np.empty_like(perm)
        inverse_perm[perm] = np.arange(len(perm))

        shuffled_answer_idx = int(inverse_perm[answer_idx])
        return shuffled_options, shuffled_answer_idx

    def get(self, item, rng: np.random.RandomState):
        ex = self.data[item]
        images = [join(DATA_HOME, x) for x in ex["images"]]
        if self.flat:
            question = ex["oe_question"] if self.direct_answer else ex["mc_question"]
            out = dict(
                image=images,
                question=question,
                metadata=dict(
                    example_id=ex["example_id"],
                    subset=ex["subset"],
                ),
                style=self.style,
            )
            if self.direct_answer:
                out["answer"] = ex["direct_answer"]
            else:
                out["options"], out["answer_idx"] = self.shuffle_options(
                    ex["choices"], ex["correct_choice_idx"], rng
                )
        else:
            questions = ex["oe_question"] if self.direct_answer else ex["mc_question"]
            messages = []
            for i, question in enumerate(questions):
                if self.direct_answer:
                    messages.append(dict(question=question, answer=ex["direct_answer"][i], style=self.style))
                else:
                    options, answer_idx = self.shuffle_options(
                        ex["choices"][i], ex["correct_choice_idx"][i], rng
                    )
                    messages.append(
                        dict(
                            question=question,
                            options=options,
                            answer_idx=answer_idx,
                            style=self.style,
                        )
                    )
            out = dict(
                image=images,
                message_list=messages,
                metadata=dict(
                    example_id=ex["example_id"],
                    subset=ex["subset"],
                ),
            )

        return out
