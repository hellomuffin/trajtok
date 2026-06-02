import dataclasses
import hashlib
import io
import json
import logging
import multiprocessing
import os
import pickle
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from os import rename, makedirs
from os.path import join, exists, relpath
from typing import Union, Dict, Optional

import PIL.Image
import datasets
import httpx
import numpy as np
import requests
import urllib3
from PIL import ImageFile
from olmo.util import flatten_list
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

from tqdm import tqdm

from olmo.data.dataset import DATA_HOME
from olmo.io import _s3_get_bytes_range



def setup_pil():
    PIL.Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclasses.dataclass
class DownloadError:
    url: str
    exception: Union[str, Exception]


@dataclasses.dataclass
class ImageError:
    url: str
    exception: Optional[Exception] = None


def compute_hash(string: Union[str, bytes]) -> str:
    if isinstance(string, str):
        return hashlib.sha256(string.encode("utf-8")).hexdigest()
    else:
        return hashlib.sha256(string).hexdigest()


client = None


def init_worker():
    global client
    client = httpx.Client(timeout=httpx.Timeout(60, connect=5))
    setup_pil()


def _download_images(args):
    url, image_sha, check_sha, cache_only, src_dir, kwargs = args
    global client
    internal_url = None
    if isinstance(image_sha, tuple):
        image_sha, internal_url = image_sha
    image_id = compute_hash(url)
    cache_file = join(src_dir, image_id)

    if exists(cache_file):
        with open(cache_file, "rb") as f:
            image_bytes = f.read()
    elif cache_only:
        return DownloadError(url, ValueError('Not in cache'))
    else:
        max_attempts = 3
        try:
            for attempt in range(max_attempts):
                if attempt > 0:
                    time.sleep(2 ** attempt)
                try:
                    response = client.get(url)
                except httpx.TransportError:
                    # Retry connection errors
                    if attempt == max_attempts-1:
                        raise
                    else:
                        continue
                if response.status_code == 429 and attempt < max_attempts-1:
                    # Too many requests error, try and back off
                    continue
                response.raise_for_status()
                image_bytes = response.content
                break
        except Exception as e:
            with open(cache_file, 'w') as f:
                f.write(str(e))
            # httpx errors sometimes cannot be sent between processes, so just send the string
            return DownloadError(url, str(e))
        # Else write the file bytes even though we have not confirmed the result is an image
        # Write to a tmp file and rename to ensure we don't only partially write an image if
        # we crash mid-write
        with open(cache_file + ".tmp", 'wb') as f:
            f.write(image_bytes)
        rename(cache_file + ".tmp", cache_file)

    if check_sha:
        downloaded_hash = compute_hash(image_bytes)
        assert image_sha is not None
        if downloaded_hash != image_sha:
            return ImageError(url, ValueError("Mismatched image hash"))
    else:
        # Else make sure we actually got an image, and it can be parsed by PIL
        try:
            # Avoid annoying palette transparency warnings filling up the logs
            with warnings.catch_warnings(record=True) as w:
                img = PIL.Image.open(io.BytesIO(image_bytes))
                if min(img.size) == 0:
                    raise ValueError("Zero dimensional image")
        except Exception as e:
            return ImageError(url, e)

    return url, cache_file


def download_pixmo_urls(
    data: datasets.Dataset,
    n_processes,
    check_sha,
    output_dir,
    request_kwargs=None,
    cache_only=False,
    verify=True
) -> Dict[str, str]:
    """Download urls from a PixMo dataset, return a map of urls->filename"""
    if "image_urls" in data.features:
        assert not check_sha
        urls = set(flatten_list(data["image_urls"]))
        urls_and_shas = [(url, None) for url in urls]
    elif check_sha:
        urls_and_shas = list(dict(zip(data["image_url"], data["image_sha256"])).items())
    else:
        urls_and_shas = [(url, None) for url in list(set(data["image_url"]))]

    # Randomize order so resuming is more convenient, speed is more predictable,
    # and to distribute requests across different domains
    urls_and_shas.sort(key=lambda x: x[0])
    np.random.RandomState(58713).shuffle(urls_and_shas)

    logging.info(f"Getting files for {len(urls_and_shas)} image URLs")
    makedirs(output_dir, exist_ok=True)
    if request_kwargs is None:
        request_kwargs = dict(timeout=60)
    if not verify:
        request_kwargs["verify"] = False
        urllib3.disable_warnings()

    images = []
    to_save = [(url, image_sha, check_sha, cache_only, output_dir, request_kwargs) for url, image_sha in urls_and_shas]
    pbar = tqdm(total=len(to_save), desc=f"{0}/{len(to_save)}")
    image_error, download_err, success = 0, 0, 0

    if n_processes != 1:
        def _iter():
            with multiprocessing.Pool(processes=n_processes, initializer=init_worker) as pool:
                for val in pool.imap_unordered(_download_images, to_save):
                    yield val
    else:
        init_worker()
        def _iter():
            for val in to_save:
                yield _download_images(val)

    found_urls = {}
    for val in _iter():
        if isinstance(val, ImageError):
            image_error += 1
        elif isinstance(val, DownloadError):
            download_err += 1
        else:
            url, filename = val
            found_urls[url] = filename
            success += 1
        pbar.update(1)
        pbar.set_description(
            f"dl_er={download_err} file_err={image_error}",
            refresh=False)
    pbar.close()
    logging.info(f"Got images for {len(found_urls)}/{len(urls_and_shas)} ({len(found_urls)/len(urls_and_shas)*100:0.2f}%) image URLs")
    return found_urls


def filter_and_group_data(data: datasets.Dataset, url_to_path: Dict, check_sha: bool) -> datasets.Dataset:
    """
    Groups a pixmo datasets so each row contains all annotation for one image, and add
    images path using `url_to_path`, removing rows that do not exist in `url_to_path`
    """
    grouped_by_url = defaultdict(list)
    for example in data:
        if example["image_url"] not in url_to_path:
            continue
        grouped_by_url[example["image_url"]].append(example)

    grouped_examples = []
    for image_url, examples in grouped_by_url.items():
        grouped = dict(
            image_url=image_url,
            image=url_to_path[image_url],
        )
        if "image_sha256" in examples[0] and not check_sha:
            assert all(examples[0]["image_sha256"] == ex["image_sha256"] for ex in examples)
            grouped["original_sha256"] = examples[0]["image_sha256"]
        annotations = defaultdict(list)
        for ex in examples:
            for k, v in ex.items():
                if k not in ["image_url", "image_sha256"]:
                    annotations[k].append(v)
        grouped.update(annotations)
        grouped_examples.append(grouped)
    return datasets.Dataset.from_list(grouped_examples)
