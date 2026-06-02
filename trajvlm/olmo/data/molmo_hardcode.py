import json
import logging
from os.path import join, basename, relpath
import numpy as np
import datasets

from olmo.data.dataset import Dataset, DATA_HOME
from olmo.data.pixmo_datasets import PIXMO_IMAGES
from olmo.data.molmo2_datasets import Molmo2HumanQA, Molmo2SynCaptionsQA, VIDEO_HOME
from olmo.io import file_exists, list_directory, write_file, read_file

log = logging.getLogger(__name__)


class Molmo2HardCodes(Dataset):
    """Hardocded responses so the model understands it is Molmo"""
    HOME = join(DATA_HOME, "molmo2-hardcodes")
    HF_SOURCE = "allenai/molmo2-hardcodes"

    @classmethod
    def download(cls, n_procs=1):
        datasets.load_dataset_builder(cls.HF_SOURCE).download_and_prepare()

        # Collect images/videos to pair with the hardcodes so we can train them in the presence
        # of multi-modal input. It doesn't matter much what images/videos we use, so we just
        # grab some from other datasets
        if not file_exists(join(cls.HOME, "images.json")):
            logging.info("Getting image list")
            images = sorted(basename(x) for x in list_directory(PIXMO_IMAGES))
            if len(images) == 0:
                raise ValueError("Download pixmo images before molmo2hardcodes")
            write_file(cls.HOME, "images.json", json.dumps(images), True)
        if not file_exists(join(cls.HOME, "videos.json")):
            logging.info("Getting video list")
            videos = set()
            for ds in [
                lambda: Molmo2SynCaptionsQA("train"),
                lambda: Molmo2HumanQA("train")
            ]:
                for ex in ds():
                    videos.add(relpath(ex["video"], DATA_HOME))
            write_file(cls.HOME, "videos.json", json.dumps(sorted(videos)), True)

    def __init__(self, p_video=0.25):
        self.p_video = p_video
        hf_data = datasets.load_dataset(self.HF_SOURCE, keep_in_memory=True)["train"]
        data = []
        for hardcode in hf_data:
            for question in hardcode["questions"]:
                if hardcode["images"]:
                    for image in hardcode["images"]:
                        data.append(dict(
                            question=question,
                            image=image,
                            answer=hardcode["response"]
                        ))
                else:
                    data.append(dict(
                        question=question,
                        answer=hardcode["response"]
                    ))
        self.data = data
        self.images = json.loads(read_file(join(self.HOME, "images.json")))
        self.videos = json.loads(read_file(join(self.HOME, "videos.json")))
        self.options = ["image", "video", "multi-image", "none"]
        self.probs = [0.25, self.p_video, 0.15, 0.35]
        self.probs = np.array(self.probs) / sum(self.probs)

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = dict(self.data[item], style="user_qa")
        if "image" not in ex:
            src = rng.choice(self.options, p=self.probs)
            if src == "image":
                ex["image"] = join(PIXMO_IMAGES, rng.choice(self.images))
            elif src == "multi-image":
                n = rng.randint(2, 6)
                ex["image"] = [join(PIXMO_IMAGES, rng.choice(self.images)) for _ in range(n)]
            elif src == "video":
                ex["video"] = join(DATA_HOME, rng.choice(self.videos))
            elif src == "none":
                pass
            else:
                raise RuntimeError()
        return ex




