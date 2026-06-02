import logging
import multiprocessing
import os
import shutil
import tarfile
import tempfile
import uuid
import zipfile
from io import BytesIO
from os.path import dirname, join, exists
from pathlib import Path
from typing import Tuple, Iterable, List, Set

import PIL
import datasets
import numpy as np
import requests
from PIL.ImageFile import ImageFile
from tqdm import tqdm

from olmo.io import write_file, dir_is_empty

log = logging.getLogger(__name__)


def save_local_dataset(dataset: datasets.Dataset, location: str, n_procs, n_val=None):
    if len(dataset) == 0:
        raise ValueError("Given an empty dataset")
    if n_val:
        split = dataset.train_test_split(test_size=n_val, seed=96817)
        dataset = datasets.DatasetDict(train=split["train"], validation=split["test"])
    logging.info(f"Preparing local HF dataset at {location}...")
    if exists(location):
        logging.info(f"{location} already exists, it will be removed")
        shutil.rmtree(location)
    dataset.save_to_disk(location)
    logging.info("Done")


def maybe_download_and_unzip(location, url, expected_dir=None):
    expected_dir = join(location, expected_dir) if expected_dir else location
    expected_dir = Path(expected_dir)
    if exists(expected_dir) and not dir_is_empty(expected_dir):
        log.info(f"Skip downloading {url} since {expected_dir} exists")
    else:
        download_and_unzip(location, url)


def download_and_unzip(location, url):
    unique_id = str(uuid.uuid4())

    os.makedirs(location, exist_ok=True)
    download_path = join(location, unique_id)

    try:
        # Download the file
        log.info(f"Downloading from {url} to {download_path}...")
        _download_file(url, download_path)

        # Extract the zip file
        log.info(f"Extracting to {location}...")
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(location)
        log.info("Extraction complete!")

    finally:
        # Always clean up the zip file, even if extraction fails
        if os.path.exists(download_path):
            os.remove(download_path)
            log.info(f"Removed {download_path}")


def maybe_download_and_untar(location, url, expected_dir=None):
    expected_dir = join(location, expected_dir) if expected_dir else location
    if exists(expected_dir) and not dir_is_empty(expected_dir):
        log.info(f"Skip downloading {url} since {expected_dir} exists")
    else:
        log.info(f"Starting download and extraction process for {url}")
        download_and_untar(location, url)
        log.info(f"Successfully completed download and extraction to {expected_dir}")


def download_and_untar(location, url):
    unique_id = str(uuid.uuid4())

    os.makedirs(location, exist_ok=True)
    download_path = join(location, unique_id)

    try:
        # Overall progress: 2 main phases (download + extract)
        with tqdm(total=2, desc="Download & Extract", unit="phase") as overall_pbar:
            # Download the file
            overall_pbar.set_description("Downloading...")
            log.info(f"Downloading from {url} to {download_path}...")
            _download_file(url, download_path)
            overall_pbar.update(1)

            # Extract the tar file
            overall_pbar.set_description("Extracting...")
            log.info(f"Extracting to {location}...")
            with tarfile.open(download_path, 'r:*') as tar_ref:
                # Get list of members to extract for progress tracking
                members = tar_ref.getmembers()

                # Extract with progress bar
                with tqdm(total=len(members), desc="  Files", unit="file", leave=False) as extract_pbar:
                    for member in members:
                        tar_ref.extract(member, location)
                        extract_pbar.update(1)
            overall_pbar.update(1)

        log.info("Download and extraction complete!")

    finally:
        # Always clean up the tar file, even if extraction fails
        if os.path.exists(download_path):
            os.remove(download_path)
            log.info(f"Removed {download_path}")


def maybe_download_file(url, filename):
    if exists(filename):
        log.info(f"{filename} already exists")
        return
    target_dir = dirname(filename)
    os.makedirs(target_dir, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(dir=target_dir, prefix='.tmp_download_')
    log.info(f"Downloading {url}...")
    _download_file(url, temp_path)
    os.replace(temp_path, filename)


def _download_file(url, filename):
    if url.startswith("https://drive.google.com/"):
        try:
            import gdown
        except ImportError:
            raise ImportError("Install gdown to download gdrive files")
        gdown.download(url, filename, quiet=False, fuzzy=True)
        return

    # Send a GET request to the URL
    response = requests.get(url, stream=True)
    # Get the total file size
    total_size = int(response.headers.get('content-length', 0))

    # Open the local file to write the downloaded content
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)


def setup_pil():
    PIL.Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True


def save_image(args) -> Tuple[str, bool]:
    image: PIL.Image.Image = args[0]
    filename: str = args[1]

    if isinstance(image, bytes):
        image_bytes = image
    else:
        assert isinstance(image, PIL.Image.Image), \
            f"{filename}: Expected a PIL image, got {type(image)}"
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
    write_file(
        os.path.dirname(filename),
        os.path.basename(filename),
        image_bytes,
        save_overwrite=True
    )
    return filename


def save_images(
    pil_images: Iterable[PIL.Image.Image],
    filenames: List[str],
    n_procs: int = 1,
) -> Set[str]:
    if n_procs != 1:
        def _iter():
            with multiprocessing.Pool(processes=n_procs, initializer=setup_pil) as pool:
                for val in pool.imap_unordered(save_image, zip(pil_images, filenames)):
                    yield val
    else:
        setup_pil()
        def _iter():
            for val in zip(pil_images, filenames):
                yield save_image(val)

    pbar = tqdm(total=len(filenames), desc="Saving images")
    saved_images = set()
    for val in _iter():
        saved_images.add(val[0])
        pbar.update(1)
    pbar.close()
    logging.info(
        f"Saved {len(saved_images)}/{len(filenames)} ({len(saved_images)/len(filenames) * 100:0.2f}%) images")
    return saved_images


def make_random_state(seed: int, *seeds: int) -> np.random.RandomState:
    ss = np.random.SeedSequence(seed, spawn_key=seeds)
    return np.random.RandomState(np.random.MT19937(ss))