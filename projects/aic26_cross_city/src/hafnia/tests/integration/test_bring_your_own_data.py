from datetime import datetime

import pytest

from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import Sample
from hafnia.platform.datasets import get_dataset_by_name, get_datasets
from hafnia.utils import is_hafnia_configured
from tests.helper_testing import get_micro_hafnia_dataset, is_github_actions_pipeline

INTEGRATION_TEST_DATASET_NAME_PREFIX = "integration-test-dataset-fg3dh-"
DATETIME_FORMAT = "%Y-%m-%dT%H-%M-%S"


def test_bring_your_own_data():
    """Placeholder test for bring your own data functionality."""
    if not is_hafnia_configured():
        pytest.skip("Hafnia platform not configured. Skipping bring your own data test.")

    version = "1.0.0"
    user_dataset = get_micro_hafnia_dataset("micro-tiny-dataset")
    user_dataset_name = get_integration_test_dataset_name()
    user_dataset.info.dataset_name = user_dataset_name
    user_dataset.info.version = version
    user_dataset.info.description = "Integration test for BYOD functionality. This dataset can be deleted."

    # Test case 1: Upload dataset to platform
    user_dataset.upload_to_platform(interactive=False, allow_version_overwrite=False)

    # Check that dataset is now available on the platform
    get_dataset_by_name(user_dataset_name)

    # Check that the sample dataset can be downloaded
    user_dataset_reloaded = HafniaDataset.from_name(user_dataset_name, version=version)
    sample = Sample(**user_dataset_reloaded[0])
    sample.draw_annotations()  # Smoke test for sample functionality

    # Test case 2: Dataset upload is blocked when version already exists
    with pytest.raises(ValueError, match="Upload cancelled"):
        user_dataset.upload_to_platform(interactive=False, allow_version_overwrite=False)

    # Test case 3: Overwrite existing version is allowed when flag is set
    user_dataset.upload_to_platform(interactive=False, allow_version_overwrite=True)

    # Test case 4: Upload new version of the dataset
    user_dataset.info.version = "1.0.1"
    user_dataset.upload_to_platform(interactive=False, allow_version_overwrite=False)

    # Test case 5: Delete dataset from platform
    user_dataset.delete_on_platform(interactive=False)

    # Check that dataset is no longer available on the platform
    dataset_response = get_dataset_by_name(user_dataset_name)
    assert dataset_response is None, "Dataset was not deleted from the platform."


def get_integration_test_dataset_name() -> str:
    date_str = datetime.now().strftime(DATETIME_FORMAT)
    dataset_name = f"{INTEGRATION_TEST_DATASET_NAME_PREFIX}{date_str}"  # Adding timestamp to ensure uniqueness
    return dataset_name


def test_remove_leftover_integration_test_datasets():
    """Cleans up old integration test datasets from the platform."""
    if is_github_actions_pipeline():
        pytest.skip("Skipping cleanup of integration test datasets in GitHub Actions.")

    date_now = datetime.now()
    dataset_responses = get_datasets()
    delete_test_datasets = []
    for dataset_response in dataset_responses:
        _dataset_name = dataset_response["name"]
        is_integration_test_dataset = _dataset_name.startswith(INTEGRATION_TEST_DATASET_NAME_PREFIX)

        if not is_integration_test_dataset:
            continue

        _date_str = datetime.strptime(_dataset_name.replace(INTEGRATION_TEST_DATASET_NAME_PREFIX, ""), DATETIME_FORMAT)
        is_old = (date_now - _date_str).total_seconds() > 60  # 1 minute old

        if not is_old:
            continue

        print(f"Clean up - deleting old integration tests: {_dataset_name}")
        delete_test_datasets.append(_dataset_name)

    if len(delete_test_datasets) > 0:
        delete_cmds = [f"uv run hafnia dataset delete {_name} --no-interactive" for _name in delete_test_datasets]
        pytest.fail(
            f"Found ({len(delete_test_datasets)}) leftover integration test datasets: "
            f"{delete_test_datasets}. "
            "Please clean them up manually. Run the following commands in terminal to delete them \n\t"
            + " && \n\t".join(delete_cmds)
        )
