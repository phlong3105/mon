import io
import math
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xxhash
from packaging.version import InvalidVersion, Version
from PIL import Image

from hafnia.log import user_logger


def is_valid_version_string(version: Optional[str], allow_none: bool = False, allow_latest: bool = False) -> bool:
    if allow_none and version is None:
        return True
    if allow_latest and version == "latest":
        return True
    return version_from_string(version, raise_error=False) is not None


def version_from_string(version: Optional[str], raise_error: bool = True) -> Optional[Version]:
    if version is None:
        if raise_error:
            raise ValueError("Version is 'None'. A valid version string is required e.g '1.0.0'")
        return None

    try:
        version_casted = Version(version)
    except (InvalidVersion, TypeError) as e:
        if raise_error:
            raise ValueError(f"Invalid version string/type: {version}") from e
        return None

    # Check if version is semantic versioning (MAJOR.MINOR.PATCH)
    if len(version_casted.release) < 3:
        if raise_error:
            raise ValueError(f"Version string '{version}' is not semantic versioning (MAJOR.MINOR.PATCH)")
        return None
    return version_casted


def dataset_name_and_version_from_string(
    string: str,
    resolve_missing_version: bool = True,
) -> Tuple[str, Optional[str]]:
    if not isinstance(string, str):
        raise TypeError(f"'{type(string)}' for '{string}' is an unsupported type. Expected 'str' e.g 'mnist:1.0.0'")

    parts = string.split(":")
    if len(parts) == 1:
        dataset_name = parts[0]
        if resolve_missing_version:
            version = "latest"  # Default to 'latest' if version is missing. This will be resolved to a specific version later.
            user_logger.info(f"Version is missing in dataset name: {string}. Defaulting to version='latest'.")
        else:
            raise ValueError(f"Version is missing in dataset name: {string}. Use 'name:version'")
    elif len(parts) == 2:
        dataset_name, version = parts
    else:
        raise ValueError(f"Invalid dataset name format: {string}. Use 'name' or 'name:version' ")

    if not is_valid_version_string(version, allow_none=True, allow_latest=True):
        raise ValueError(f"Invalid version string: {version}. Use semantic versioning e.g. '1.0.0' or 'latest'")

    return dataset_name, version


def create_split_name_list_from_ratios(split_ratios: Dict[str, float], n_items: int, seed: int = 42) -> List[str]:
    samples_per_split = split_sizes_from_ratios(split_ratios=split_ratios, n_items=n_items)

    split_name_column = []
    for split_name, n_split_samples in samples_per_split.items():
        split_name_column.extend([split_name] * n_split_samples)
    random.Random(seed).shuffle(split_name_column)  # Shuffle the split names

    return split_name_column


def hash_file_xxhash(path: Path, chunk_size: int = 262144) -> str:
    hasher = xxhash.xxh3_128()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):  # 8192, 16384, 32768, 65536
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_from_bytes(data: bytes) -> str:
    hasher = xxhash.xxh3_128()
    hasher.update(data)
    return hasher.hexdigest()


def save_image_with_hash_name(image: np.ndarray, path_folder: Path, allow_skip: bool = True) -> Path:
    pil_image = Image.fromarray(image)
    path_image = save_pil_image_with_hash_name(pil_image, path_folder, allow_skip=allow_skip)
    return path_image


def save_pil_image_with_hash_name(
    image: Image.Image,
    path_folder: Path,
    allow_skip: bool = True,
    compress_level: int = 1,
) -> Path:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=compress_level)
    png_bytes = buffer.getvalue()
    hash_value = hash_from_bytes(png_bytes)
    path_image = Path(path_folder) / relative_path_from_hash(hash=hash_value, suffix=".png")
    if allow_skip and path_image.exists():
        return path_image
    path_image.parent.mkdir(parents=True, exist_ok=True)
    path_image.write_bytes(png_bytes)
    return path_image


def copy_and_rename_file_to_hash_value(
    path_source: Path,
    path_dataset_root: Path,
    allow_skip: bool = True,
) -> Path:
    """
    Copies a file to a dataset root directory with a hash-based name and sub-directory structure.
    """

    if not path_source.exists():
        raise FileNotFoundError(f"Source file {path_source} does not exist.")

    hash_value = hash_file_xxhash(path_source)
    path_file = path_dataset_root / relative_path_from_hash(hash=hash_value, suffix=path_source.suffix)
    path_file.parent.mkdir(parents=True, exist_ok=True)

    if allow_skip and path_file.exists():
        return path_file

    shutil.copy2(path_source, path_file)

    return path_file


def relative_path_from_hash(hash: str, suffix: str) -> str:
    return f"{hash}{suffix}"


def split_sizes_from_ratios(n_items: int, split_ratios: Dict[str, float]) -> Dict[str, int]:
    summed_ratios = sum(split_ratios.values())
    abs_tols = 0.0011  # Allow some tolerance for floating point errors {"test": 0.333, "val": 0.333, "train": 0.333}
    if not math.isclose(summed_ratios, 1.0, abs_tol=abs_tols):  # Allow tolerance to allow e.g. (0.333, 0.333, 0.333)
        raise ValueError(f"Split ratios must sum to 1.0. The summed values of {split_ratios} is {summed_ratios}")

    # recaculate split sizes
    split_ratios = {split_name: split_ratio / summed_ratios for split_name, split_ratio in split_ratios.items()}
    split_sizes = {split_name: int(n_items * split_ratio) for split_name, split_ratio in split_ratios.items()}

    remaining_items = n_items - sum(split_sizes.values())
    if remaining_items > 0:  # Distribute remaining items evenly across splits
        for _ in range(remaining_items):
            # Select name by the largest error from the expected distribution
            total_size = sum(split_sizes.values())
            distribution_error = {
                split_name: abs(split_ratios[split_name] - (size / total_size))
                for split_name, size in split_sizes.items()
            }

            split_with_largest_error = sorted(distribution_error.items(), key=lambda x: x[1], reverse=True)[0][0]
            split_sizes[split_with_largest_error] += 1

    if sum(split_sizes.values()) != n_items:
        raise ValueError("Something is wrong. The split sizes do not match the number of items.")

    return split_sizes
