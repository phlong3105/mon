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
from torch import nn, Tensor

from mon.core import (
    Config,
    ConfigContext,
    create_progress_bar,
    K,
    Path,
    RunMode,
    Size,
    sys_ctx,
    TensorOrArray,
    TimeProfiler,
    log,
)
from mon.dataset import transform as T
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
        self.config = config

        # Extract commonly used attributes for convenience
        self.device = config.device
        self.benchmark = config.benchmark
        self.verbose = config.verbose

        # Allocate resources
        # We will initialize these attributes later to avoid a long
        # initialization time
        self.model: nn.Module | None = None
        self.transforms: T.Compose | None = None

    # --- Properties ---
    @abstractmethod
    def init_model(self):
        """Initialize ``self.model`` attribute."""
        pass

    @abstractmethod
    def init_transforms(self):
        """Initialize ``self.transforms`` attribute."""
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
        self.init_model()
        if self.model is None:
            raise ValueError(f"'model' is not initialized.")

        # 4. Run benchmark
        self.benchmark()

        # 5. Define transforms
        self.init_transforms()
        if self.transforms is None:
            if self.verbose:
                log(f"'transforms' is not initialized.")

        # 5. Main loop
        with create_progress_bar() as pbar:
            for data in pbar.track(
                sequence=config.data,
                total=len(config.data),
                description=f"[bright_yellow]Data"
            ):
                # 5.1. Predict data
                timers = TimeProfiler()
                self.predict_data(data=data, pbar=pbar, timers=timers)
                timers.total.tock()

                # 5.2. Finish
                timers.print()

    # --- Prediction ---
    @abstractmethod
    def predict_data(self, data: Path | str, pbar: Progress, timers: TimeProfiler):
        """Predict the output of the model for a single data source.

        Args:
            data (Path | str): The path to the data point to predict.
            pbar (Progress): The progress bar to update during prediction.
            timers (TimeProfiler): The time profiler to record timing information
                during prediction.
        """
        pass

    # --- Utilities ---
    def benchmark(self):
        """Run the benchmark for the model."""
        config = self.config
        imgsz = Size.from_value(config.eval_imgsz)

        if self.benchmark:
            benchmark(self.model, imgsz=imgsz)

    @abstractmethod
    def save(self, path: Path, outputs: dict):
        """Save the main prediction results to a file.

        Args:
            path (Path): The source path used to determine the output file path.
            outputs (dict): The dictionary containing the main prediction results.
        """
        pass

    @abstractmethod
    def save_debug(self, path: Path, outputs: dict):
        """Save debugging results for visualization.

        Args:
            path (Path): The source path used to determine the output file path.
            outputs (dict): The dictionary containing the debugging results.
        """
        pass

    def save_image(self, path: Path, image: TensorOrArray, size: Size):
        """Save a debug image for visualization.

        Args:
            path (Path): The source path used to determine the output file path.
            image (TensorOrArray): The image to be saved, which can be a tensor
                or an array.
            size (Size): The original size of the input image, used for resizing
                the image if necessary.
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
        save_path = config.resolve_save_file(dirname=K.PRED_DIR, src_path=path)
        write_image(image=image, path=save_path)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
