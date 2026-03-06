#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Runners.

This module provides prediction runner classes.
"""

from __future__ import annotations

__all__ = [
    "Predictor",
]

from abc import ABC, abstractmethod

import cv2
from numpy import ndarray
from rich.progress import Progress
from sympy.printing.pytorch import torch
from torch import nn, Tensor

from mon.core import (
    Config,
    ConfigContext,
    create_progress_bar,
    K,
    log,
    Path,
    RunMode,
    Size,
    Split,
    sys_ctx,
    TensorOrArray,
    TimeProfiler,
)
from mon.dataset import build_dataloader, transform as T
from mon.metrics import benchmark
from mon.ops import to_image_array, write_image

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class Predictor(ABC):
    """Base class for all predictors."""

    # --- Lifecycle & Initialization ---
    def __init__(self, config: Config):
        """Initialize a new instance.

        Args:
            config (Config): The configuration object containing all necessary
                parameters for training.
        """
        # Assign attributes
        self._config = config

        # Allocate resources
        # These attributes will be initialized later to avoid a long
        # initialization time
        self._model: nn.Module | None = None
        self._transforms: T.Compose | None = None

    # --- Properties ---
    @property
    def config(self) -> Config:
        """Return the config object."""
        return self._config

    @property
    def device(self) -> torch.device:
        """Return the device to use."""
        return self.config.device

    @property
    def benchmark(self) -> bool:
        """Return the benchmark flag."""
        return self.config.benchmark

    @property
    def verbose(self) -> bool:
        """Return the verbose flag."""
        return self.config.verbose

    @property
    def model(self) -> nn.Module:
        """Return the model object."""
        return self._model

    @abstractmethod
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        pass

    @property
    def transforms(self) -> T.Compose | None:
        """Return the transforms object."""
        return self._transforms

    @abstractmethod
    def _init_transforms(self):
        """Initialize ``self._transforms`` attribute."""
        pass

    # --- Creation ---
    @classmethod
    def from_cli(cls, *args, **kwargs) -> "Predictor":
        """Create an instance of Predictor from command-line arguments."""
        config_ctx = ConfigContext.from_cli(*args, **kwargs)
        config = config_ctx.config_for(RunMode.PREDICT)
        return cls(config)

    # --- Control ---
    def predict(self):
        """Predict the output of the model."""
        config = self.config

        # 1. Summarize the current run
        if config.verbose:
            config.log_summary()

        # 2. Setup environment
        sys_ctx.set_random_seed(config.seed)

        # 3. Define model
        self._init_model()
        if self.model is None:
            raise RuntimeError(f"'model' is not initialized.")

        # 4. Define transforms
        self._init_transforms()
        if self.transforms is None:
            if self.verbose:
                log(f"'transforms' is not initialized.")

        # 5. Run benchmark
        self.benchmark()

        # 6. Main loop
        with create_progress_bar() as pbar:
            for data in pbar.track(
                sequence=config.data,
                total=len(config.data),
                description=f"[bright_yellow]Data"
            ):
                # 6.1. Predict data
                timers = TimeProfiler()
                self._predict_data(data=data, pbar=pbar, timers=timers)
                timers.total.tock()

                # 6.2. Finish
                timers.print()

    # --- Prediction ---
    def _predict_data(self, data: Path | str, pbar: Progress, timers: TimeProfiler):
        """Predict the output of the model for a single data source.

        Args:
            data (Path | str): The path to the data point to predict.
            pbar (Progress): The progress bar to update during prediction.
            timers (TimeProfiler): The time profiler to record timing information
                during prediction.
        """
        config = self.config
        save_debug = config.save_debug

        # 1. Build dataset
        data_name, dataloader = build_dataloader(
            src=data,
            dataset_dir=config.data_dir,
            split=Split.TEST,
            transforms=self.transforms,
        )

        # 2. Main processing loop
        task = pbar.add_task(
            description=f"[bright_yellow]Predicting {data_name}",
            total=len(dataloader),
        )
        for i, datapoint in enumerate(dataloader):
            # 2.1. Predict step
            outputs = self._predict_step(datapoint=datapoint, timers=timers)

            # 2.2. Post-process
            timers.postprocess.tick()
            meta = datapoint["meta"]
            self._save(outputs, meta)
            if save_debug:
                self._save_debug(outputs, meta)
            timers.postprocess.tock()

            pbar.update(task, advance=1)
        pbar.remove_task(task)

    @abstractmethod
    def _predict_step(self, datapoint: dict, timers: TimeProfiler) -> dict:
        """Predict the output of the model for a single data point.

        Args:
            datapoint (dict): The dictionary containing the data point to predict.
            timers (TimeProfiler): The time profiler to record timing information
                during prediction.

        Returns:
            dict: The dictionary containing the prediction results.
        """
        pass

    # --- Utilities ---
    def _benchmark(self):
        """Run the benchmark for the model."""
        config = self.config
        imgsz = Size.from_value(config.eval_imgsz)

        if self.benchmark:
            benchmark(self.model, imgsz=imgsz)

    @abstractmethod
    def _save(self, outputs: dict, meta: dict):
        """Save the main prediction results to a file.

        Args:
            outputs (dict): The dictionary containing the main prediction results.
            meta (dict): The dictionary containing the metadata.
        """
        pass

    @abstractmethod
    def _save_debug(self, outputs: dict, meta: dict):
        """Save debugging results for visualization.

        Args:
            outputs (dict): The dictionary containing the debugging results.
            meta (dict): The dictionary containing the metadata.
        """
        pass

    def _save_image(
        self,
        image: TensorOrArray,
        size: Size,
        src_path: Path,
        stem: str | None = None
    ):
        """Save a debug image for visualization.

        Args:
            image (TensorOrArray): The image to be saved, which can be a tensor
                or an array.
            size (Size): The original size of the input image, used for resizing
                the image if necessary.
            src_path (Path): The source path, used to determine the output file path.
            stem (str, optional): An optional string to be appended to the
                output file name for differentiation. If None, the original
                file name will be used.
        """
        config = self.config

        # Convert the input image to an array
        if isinstance(image, Tensor):
            image = to_image_array(image)
        if not isinstance(image, ndarray):
            raise TypeError(
                f"Expected 'image' to be an array, "
                f"but got {type(image).__name__}."
            )

        # Resize the image if necessary
        imgsz = Size.from_value(image)
        if imgsz != size:
            image = cv2.resize(image, size.wh)

        # Save the image
        if stem:
            save_dir = config.resolve_save_dir(dirname=K.PRED_DIR, src_path=src_path)
            save_path = save_dir / f"{src_path.stem}_{stem}{K.IMAGE_EXT}"
        else:
            save_path = config.resolve_save_file(dirname=K.PRED_DIR, src_path=src_path)
        write_image(image=image, path=save_path)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
