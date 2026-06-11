from pathlib import Path

import pytest

from hafnia.dataset.dataset_names import SampleField
from hafnia.dataset.dataset_recipe.dataset_recipe import DatasetRecipe
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.operations.dataset_s3_storage import sync_files_from_platform
from hafnia.utils import is_hafnia_configured
from tests.helper_testing_datasets import DATASET_SPEC_MNIST


def test_recipe_download_files_with_write(tmp_path):
    """
    Verifies the logic used in the "fetching data" step. The naive approach would be to
    first build the recipe and then write the dataset to a local path:

    ```python
    dataset: HafniaDataset = recipe.build()  # This would download all files to a temp location
    dataset.write(path_write)  # This would copy and hash all files to path.
    ```
    Above approach would both download, copy and hash all files which requires storage and is time consuming.

    Instead we can do the following:
    ```python
    dataset: HafniaDataset = recipe.build(download_files=False)
    dataset = sync_files_from_platform(dataset, path_write)
    dataset.write_annotations(path_write)
    ```
    This approach approach has multiple performance advantages
    1) We download only files after (potentially) filtering done in the recipe.
    2) We avoid the write() operation that would both copy and hash all images/video files.
    """
    if not is_hafnia_configured():
        pytest.skip("Hafnia platform not configured. Skipping recipe download logic test.")

    recipe = DatasetRecipe.from_merge(
        recipe0=DatasetRecipe.from_name(DATASET_SPEC_MNIST.name, version=DATASET_SPEC_MNIST.version),
        recipe1=DatasetRecipe.from_name(DATASET_SPEC_MNIST.name, version=DATASET_SPEC_MNIST.version),
    )

    path_write = tmp_path / "write"

    dataset: HafniaDataset = recipe.build(download_files=False)  # Images/videos are not downloaded yet
    dataset = sync_files_from_platform(dataset, path_write)

    # All file paths should now exist locally
    file_paths = dataset.samples[SampleField.FILE_PATH].to_list()
    assert len(file_paths) > 0
    for path in file_paths:
        assert Path(path).exists(), f"Expected downloaded file to exist: {path}"

    assert len(list(path_write.iterdir())) == 1, "Expect only 'data' folder in the download path"
    dataset.write_annotations(path_write)
    assert len(list(path_write.iterdir())) == 4, "Expect remaining files"

    # Written files should preserve original filenames
    written_dataset = HafniaDataset.from_path(path_write)
    written_paths = written_dataset.samples[SampleField.FILE_PATH].to_list()
    assert len(written_paths) == len(file_paths)
    for written_path in written_paths:
        assert Path(written_path).exists(), f"Expected written file to exist: {written_path}"

    dataset_new = HafniaDataset.from_path(path_write)
    dataset_new.check_dataset()
