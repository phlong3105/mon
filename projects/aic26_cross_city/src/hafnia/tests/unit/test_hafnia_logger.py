import json
import os
from pathlib import Path

import polars as pl
import pytest

from hafnia.experiment.hafnia_logger import EntityType, HafniaLogger


@pytest.fixture(scope="function")
def logger(tmpdir: Path) -> HafniaLogger:
    """Create a logger instance for testing."""
    if "HAFNIA_LOCAL_SCRIPT" not in os.environ:
        os.environ["HAFNIA_LOCAL_SCRIPT"] = "true"
    return HafniaLogger(project_name="test_project", log_dir=Path(tmpdir))


def test_basic_scalar_logging(logger: HafniaLogger) -> None:
    """Test that basic scalar logging works."""

    assert not logger.log_file.exists()

    logger.log_scalar("test_scalar", 42.0, 1)

    assert logger.log_file.exists()

    logger.log_scalar("test_scalar", 43.0, 2)

    df = pl.read_parquet(logger.log_file)
    assert len(df) == 2
    assert df[0]["value"].item() == 42.0
    assert df[1]["value"].item() == 43.0


def test_metric_logging(logger: HafniaLogger) -> None:
    """Test that metric logging works."""
    logger.log_metric("accuracy", 0.95, 100)
    logger.log_metric("loss", 0.05, 100)
    assert logger.log_file.exists()

    # Verify the data
    df = pl.read_parquet(logger.log_file)

    metrics_df = df.filter(df["ent_type"].eq(EntityType.METRIC.value))
    assert len(metrics_df) == 2
    assert "accuracy" in metrics_df["name"].to_list()
    assert "loss" in metrics_df["name"].to_list()


def test_config_logging(logger: HafniaLogger):
    """Test configuration logging."""
    config = {"learning_rate": 0.001, "batch_size": 32, "model_type": "resnet50", "project_name": "test_project"}
    logger.log_configuration(config)
    config_file = logger._path_artifacts() / "configuration.json"

    assert config_file.exists()
    loaded_config = json.loads(config_file.read_text())

    assert loaded_config["learning_rate"] == 0.001
    assert loaded_config["batch_size"] == 32
    assert loaded_config["model_type"] == "resnet50"
    assert loaded_config == config

    extra_config = {"dropout": 0.5, "batch_size": 64}
    logger.log_configuration(extra_config)

    loaded_config_with_extra = json.loads(config_file.read_text())

    expected_config = {**config, **extra_config}
    assert loaded_config_with_extra["dropout"] == 0.5
    assert loaded_config_with_extra["batch_size"] == 64
    assert loaded_config_with_extra == expected_config
