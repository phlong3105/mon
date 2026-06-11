#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hashlib
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pytest
import torch
from hafnia.experiment.command_builder import (
    auto_save_command_builder_schema,
    CommandBuilderSchema,
    DEFAULT_ORDER,
    path_of_function,
    simulate_form_data,
)

from trainer_object_detection.utils import CLI_TOOL


def file_hash(zip_file, name):
    """Get hash of the uncompressed file content inside a zip archive."""
    with zip_file.open(name) as f:
        return hashlib.md5(f.read()).hexdigest()


def compare_zip_files(zip_path1, zip_path2):
    files_changed = []
    with zipfile.ZipFile(zip_path1, "r") as z1, zipfile.ZipFile(zip_path2, "r") as z2:
        z1_files = sorted(z1.namelist())
        z2_files = sorted(z2.namelist())

        if z1_files != z2_files:
            print("The new trainer package contain new files")
            return False

        for name in z1_files:
            if file_hash(z1, name) != file_hash(z2, name):
                print(f"File content differs: {name}")
                files_changed.append(name)

    if len(files_changed) > 0:
        print(f"The following files have changed: {files_changed}")
        return False

    return True


def test_train_script():
    from scripts.train import main

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available. Skipping integration test.")
    main(project_name="test_project", epochs=1, samples=40)


def test_benchmark_script():
    from scripts.benchmark import main

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available. Skipping integration test.")
    main(samples=2, model_class_mapping="COCO2OnlyVehicle", dataset_class_mapping="Midwest2OnlyVehicle")


def test_predict_script():
    from scripts.visualize import main

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available. Skipping integration test.")
    main(samples=2)


class _StubLogger:
    """Minimal stand-in for ``HafniaLogger`` exposing only the checkpoints path."""

    def __init__(self, checkpoints_path):
        self._checkpoints_path = Path(checkpoints_path)

    def path_model_checkpoints(self):
        return self._checkpoints_path


def _make_checkpoint_zip(archive_path, class_names=("car", "truck")):
    """Build a checkpoint archive (model config + dummy weights) the way ``train.py`` does."""
    from hafnia.dataset.hafnia_dataset_types import TaskInfo
    from hafnia.dataset.primitives import Bbox

    from trainer_object_detection.wrapped_model import InitModelConfig

    archive_path = Path(archive_path)
    with tempfile.TemporaryDirectory() as source_dir:
        weights_path = Path(source_dir) / f"{archive_path.stem}.pth"
        weights_path.write_bytes(b"dummy-weights")
        model_config = InitModelConfig(
            name="RFDETRNano",
            task=TaskInfo.from_class_names(primitive=Bbox, class_names=list(class_names)),
            model_weight_path=str(weights_path),
        )
        model_config.save_model(archive_path)
    return archive_path


def test_get_checkpoint_if_available(tmp_path):
    """A checkpoint is discovered only when a ``*.zip`` archive is present, deterministically."""
    from trainer_object_detection.utils import get_checkpoint_if_available

    checkpoints_dir = tmp_path / "checkpoints"
    logger = _StubLogger(checkpoints_dir)

    # Missing checkpoints directory -> no checkpoint
    assert get_checkpoint_if_available(logger) is None

    # Empty directory -> no checkpoint
    checkpoints_dir.mkdir()
    assert get_checkpoint_if_available(logger) is None

    # Non-archive files are ignored
    (checkpoints_dir / "state.json").write_text("{}")
    assert get_checkpoint_if_available(logger) is None

    # A single checkpoint archive is returned
    _make_checkpoint_zip(checkpoints_dir / "checkpoint_best_ema.zip")
    assert get_checkpoint_if_available(logger) == checkpoints_dir / "checkpoint_best_ema.zip"

    # With multiple archives the selection is deterministic (sorted by name)
    _make_checkpoint_zip(checkpoints_dir / "checkpoint_best_regular.zip")
    assert get_checkpoint_if_available(logger) == checkpoints_dir / "checkpoint_best_ema.zip"


def test_checkpoint_is_loaded(tmp_path):
    """An available checkpoint is discovered and can be loaded back into a model config."""
    from trainer_object_detection.utils import get_checkpoint_if_available
    from trainer_object_detection.wrapped_model import InitModelConfig

    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    archive_path = _make_checkpoint_zip(
        checkpoints_dir / "checkpoint_best_ema.zip", class_names=["car", "bus", "truck"]
    )
    logger = _StubLogger(checkpoints_dir)

    checkpoint_model_path = get_checkpoint_if_available(logger)
    assert checkpoint_model_path == archive_path

    # The discovered checkpoint loads, with its weights extracted to an existing file on disk.
    model_config = InitModelConfig.load_model(checkpoint_model_path, use_weights=True)
    assert model_config.name == "RFDETRNano"
    assert [c.name for c in model_config.task.classes] == ["car", "bus", "truck"]
    assert Path(model_config.model_weight_path).exists()


def _script_main(script_name: str):
    """Import the ``main`` function from a script module by name."""
    import importlib

    module = importlib.import_module(f"scripts.{script_name}")
    return module.main


@pytest.mark.parametrize("script_name", ["train", "benchmark"])
def test_command_builder_schema(script_name: str):
    """Test that the launch schema is up-to-date for each script."""
    main = _script_main(script_name)

    path_function = path_of_function(main)
    path_function_schema = path_function.with_suffix(".schema.json")

    if script_name == "train":
        order = 0
    else:
        order = DEFAULT_ORDER
    if not path_function_schema.exists():
        auto_save_command_builder_schema(main, cli_tool=CLI_TOOL, order=order)
        pytest.fail("Launch schema file not found. Schema file have been generated. Please run the test again.")

    actual_schema = CommandBuilderSchema.from_function(main, cli_tool=CLI_TOOL, order=order)
    current_schema = CommandBuilderSchema.from_json_file(path_function_schema)

    schema_is_up_to_date = current_schema == actual_schema
    assert schema_is_up_to_date, (
        f"Launch schema in '{path_function_schema}' is outdated. Please delete the schema file "
        f"({path_function_schema}) and rerun this test to regenerate it."
    )


def test_train_command_runs():
    """Test that the train script can be invoked end-to-end via the generated CLI args."""
    from scripts.train import main

    actual_schema = CommandBuilderSchema.from_function(main, cli_tool=CLI_TOOL)
    form_data = simulate_form_data(main, user_args={"stop_early": "True"})
    cmd_args = actual_schema.command_args_from_form_data(form_data)
    subprocess.run(cmd_args, shell=True, check=True)
