from pathlib import Path
from unittest.mock import MagicMock
from zipfile import ZipFile

import pytest

from hafnia.platform.builder import check_registry, validate_trainer_package_format


@pytest.fixture
def valid_trainer_package(tmp_path: Path) -> Path:
    from zipfile import ZipFile

    zip_path = tmp_path / "valid_trainer_package.zip"
    with ZipFile(zip_path, "w") as zipf:
        zipf.writestr("src/lib/example.py", "# Example lib")
        zipf.writestr("scripts/run.py", "print('Running training.')")
        zipf.writestr("Dockerfile", "FROM python:3.9")
    return zip_path


def test_valid_trainer_package_structure(valid_trainer_package: Path) -> None:
    """Test validation with a correctly structured zip file."""
    validate_trainer_package_format(valid_trainer_package)


def test_validate_trainer_package_no_scripts(tmp_path: Path) -> None:
    """Test validation fails when no Python scripts are present."""
    from zipfile import ZipFile

    zip_path = tmp_path / "no_scripts.zip"
    with ZipFile(zip_path, "w") as zipf:
        zipf.writestr("src/lib/example.py", "# Example lib")
        zipf.writestr("Dockerfile", "FROM python:3.9")

    with pytest.raises(FileNotFoundError, match="Wrong trainer package structure"):
        validate_trainer_package_format(zip_path)


def test_invalid_trainer_package_structure(tmp_path: Path) -> None:
    """Test validation with an incorrectly structured zip file."""
    zip_path = tmp_path / "invalid_trainer_package.zip"
    with ZipFile(zip_path, "w") as zipf:
        zipf.writestr("README.md", "# Example readme")

    with pytest.raises(FileNotFoundError, match="Wrong trainer package structure"):
        validate_trainer_package_format(zip_path)


def test_successful_trainer_package_extraction(valid_trainer_package: Path, tmp_path: Path) -> None:
    """Test successful trainer package download and extraction."""

    from hashlib import sha256

    from hafnia.platform.builder import prepare_trainer_package

    state_file = "state.json"
    expected_hash = sha256(valid_trainer_package.read_bytes()).hexdigest()[:8]

    with pytest.MonkeyPatch.context() as mp:
        mock_download = MagicMock(return_value={"status": "success", "downloaded_files": [valid_trainer_package]})

        mp.setattr("hafnia.platform.builder.download_resource", mock_download)
        result = prepare_trainer_package("s3://bucket/trainer.zip", tmp_path, "api-key-123", Path(state_file))
        mock_download.assert_called_once_with("s3://bucket/trainer.zip", tmp_path.as_posix(), "api-key-123")

        assert result["digest"] == expected_hash


def test_ecr_image_exist() -> None:
    """Test when image exists in ECR."""
    mock_ecr_client = MagicMock()
    mock_ecr_client.describe_images.return_value = {"imageDetails": [{"imageTags": ["v1.0"], "imageDigest": "1234a"}]}

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("AWS_REGION", "us-west-1")
        mp.setattr("boto3.client", lambda service, **kwargs: mock_ecr_client)
        result = check_registry("my-repo:v1.0")
        assert result == "1234a"


def test_ecr_image_not_found() -> None:
    """Test when ECR client raises ImageNotFoundException."""

    from botocore.exceptions import ClientError

    mock_ecr_client = MagicMock()
    mock_ecr_client.describe_images.side_effect = ClientError(
        {"Error": {"Code": "ImageNotFoundException"}}, "describe_images"
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("AWS_REGION", "us-west-2")
        mp.setattr("boto3.client", lambda service, **kwargs: mock_ecr_client)
        result = check_registry("my-repo:v1.0")
        assert result is None
