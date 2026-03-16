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
from typing import Any

import cv2
from numpy import ndarray
from rich.progress import Progress
from torch import Tensor

from mon.core import (
    Config,
    ConfigContext,
    create_progress_bar,
    DictLike,
    K,
    log,
    Path,
    PathLike,
    RunMode,
    Size,
    Split,
    SplitLike,
    sys_ctx,
    TensorOrArray,
    TimeProfiler,
)
from mon.dataset import build_dataloader, DataLoader, Dataset, transform as T
from mon.ops import to_image_array, write_image
from .base import Runner

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class Predictor(Runner, ABC):
    """Base class for all predictors."""

    # --- Lifecycle & Initialization ---
    def __init__(self, config: Config):
        """Initialize a new instance.

        Args:
            config (Config): The configuration object containing all necessary
                parameters for training.
        """
        super().__init__(config=config)
        # Allocate resources
        # These attributes will be initialized later to avoid a long
        # initialization time
        self._transforms: T.Compose | None = None

    @abstractmethod
    def _init_transforms(self):
        """Initialize ``self._transforms`` attribute."""
        pass

    def _init_data(
        self,
        source: DictLike | PathLike,
        split: SplitLike = Split.TEST,
        transforms: T.Compose | None = None,
    ) -> tuple[str, Dataset | DataLoader]:
        """Initialize and return a dataset or dataloader.

        Args:
            source (DictLike | PathLike): A dataset/dataloader configuration
                dictionary or a source path.
            split (SplitLike, optional): The data split to use.
                Defaults to Split.TEST.
            transforms (T.Compose, optional): The data transformations to apply.
                Defaults to None.

        Returns:
            tuple[str, Dataset | DataLoader]: A tuple containing the name of the
                dataset/dataloader and the dataset/dataloader object itself.
        """
        return build_dataloader(
            src=source,
            dataset_dir=self.config.data_dir,
            split=split,
            transforms=transforms,
        )

    # --- Properties ---
    @property
    def transforms(self) -> T.Compose | None:
        """Return the transforms object."""
        return self._transforms

    # --- Creation ---
    @classmethod
    def from_cli(cls, prompt: bool = False, *args, **kwargs) -> "Predictor":
        """Create an instance of Predictor from command-line arguments.

        Args:
            prompt (bool, optional): Whether to prompt the user for input if
                necessary. Defaults to False.
        """
        config_ctx = ConfigContext.from_cli(*args, **kwargs)
        config = config_ctx.config_for(RunMode.PREDICT, prompt=prompt)
        return cls(config)

    # --- Control ---
    def predict(self):
        """Predict the output of the model."""
        config = self.config

        # 1. Summarize the current run
        if config.verbose:
            config.log_summary()

        # 2. Setup environment
        config.output_dir.mkdir(exist_ok=True, parents=True)
        sys_ctx.set_random_seed(config.seed)

        # 3. Define model
        self._init_model()
        if self.model is None:
            log(f"'model' is not initialized.")

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
                # 6.1. Update config
                config.infer_data = data

                # 6.2. Predict data
                timers = TimeProfiler()
                self._predict_data(data=data, pbar=pbar, timers=timers)
                timers.total.tock()

                # 6.3. Clean up
                config.infer_data = None

                # 6.4. Finish
                timers.print()

    # --- Prediction ---
    def _predict_data(self, data: PathLike, pbar: Progress, timers: TimeProfiler):
        """Predict the output of the model for a single data source.

        Args:
            data (PathLike): The path to the data point to predict.
            pbar (Progress): The progress bar to update during prediction.
            timers (TimeProfiler): The time profiler to record timing information
                during prediction.
        """
        config = self.config

        # 1. Build dataset
        data_name, dataloader = self._init_data(
            source=data, split=Split.TEST, transforms=self.transforms,
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
            if self.save:
                self._save(outputs=outputs, meta=meta)
            if self.save_debug:
                self._save_debug(outputs=outputs, meta=meta)
            timers.postprocess.tock()

            pbar.update(task, advance=1)
        pbar.remove_task(task)

    @abstractmethod
    def _predict_step(
        self, datapoint: dict[str, Any], timers: TimeProfiler
    ) -> dict[str, Any]:
        """Predict the output of the model for a single data point.

        Args:
            datapoint (dict): The dictionary containing the data point to predict.
            timers (TimeProfiler): The time profiler to record timing information
                during prediction.

        Returns:
            dict: The dictionary containing the prediction results.
        """
        pass

    # --- Output ---
    @abstractmethod
    def _save(self, outputs: dict[str, Any], meta: list[dict[str, Any]]):
        """Save the main prediction results to a file.

        Args:
            outputs (dict): The dictionary containing the main prediction results.
                Each key in the dictionary is a batched of prediction results.
            meta (list[dict]): The list of dictionaries containing the metadata
                for each data point.
        """
        pass

    @abstractmethod
    def _save_debug(self, outputs: dict[str, Any], meta: list[dict[str, Any]]):
        """Save debugging results for visualization.

        Args:
            outputs (dict): The dictionary containing the main prediction results.
                Each key in the dictionary is a batched of prediction results.
            meta (list[dict]): The list of dictionaries containing the metadata
                for each data point.
        """
        pass

    def _save_image(
        self,
        image: TensorOrArray,
        size: Size,
        src_path: Path,
        dirname: str = K.PRED_DIR,
        subdirname: str = "",
        stem: str = ""
    ):
        """Save a debug image for visualization.

        Args:
            image (TensorOrArray): The image to be saved, which can be a tensor
                or an array.
            size (Size): The original size of the input image, used for resizing
                the image if necessary.
            src_path (Path): The source path, used to determine the output file
                path.
            dirname (str, optional): The directory name for the output file.
                Defaults to K.PRED_DIR.
            subdirname (str): Subdirectory name to append to the output path
                (e.g., 'debug'/'mask'). Defaults to "".
            stem (str, optional): An optional string to be appended to the
                output file name for differentiation. If not provided, the
                output file name will be the same as the source file name.
                Defaults to "".
        """
        config = self.config

        # Convert the input image to an array
        if isinstance(image, Tensor):
            image = to_image_array(image)
        if not isinstance(image, ndarray):
            raise TypeError(
                f"Expected 'image' to be an array, but got {type(image).__name__}."
            )

        # Resize the image if necessary
        imgsz = Size.from_value(image)
        if imgsz != size:
            image = cv2.resize(image, size.wh, interpolation=cv2.INTER_LINEAR)

        # Save the image
        if stem:
            save_dir = config.resolve_save_dir(dirname=dirname, subdirname=subdirname, src_path=src_path)
            save_path = save_dir / f"{src_path.stem}_{stem}{K.IMAGE_EXT}"
        else:
            save_path = config.resolve_save_file(dirname=dirname, subdirname=subdirname, src_path=src_path)

        write_image(image=image, path=save_path)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
