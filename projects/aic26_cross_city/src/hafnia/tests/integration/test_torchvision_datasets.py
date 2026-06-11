from pathlib import Path

import pytest

from hafnia.dataset.format_conversions.torchvision_datasets import torchvision_to_hafnia_converters
from hafnia.dataset.hafnia_dataset import HafniaDataset
from tests.helper_testing import is_github_actions_pipeline


@pytest.mark.parametrize("dataset_name", torchvision_to_hafnia_converters())
def test_torchvision_datasets(dataset_name: str, tmp_path: Path) -> None:
    FORCE_DOWNLOAD = False

    if is_github_actions_pipeline():
        pytest.skip("Skipping torchvision dataset tests in GitHub Actions to avoid large downloads.")
    dataset = HafniaDataset.from_name_public_dataset(dataset_name, n_samples=20, force_redownload=FORCE_DOWNLOAD)
    assert len(dataset) == 20
