#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Runners.

This module provides prediction runner classes.
"""

from __future__ import annotations

__all__ = [
    "Predictor",
]

import gc
from abc import ABC, abstractmethod
from typing import override

from numpy import ndarray
from rich.progress import Progress
from tensordict import TensorDict
from torch import Tensor

from mon.core import (
    BBoxes,
    Config,
    BBoxFormat,
    ConfigContext,
    create_progress_bar,
    K,
    log,
    Path,
    RunMode,
    Size,
    Split,
    Strategy,
    sys_ctx,
    TensorOrArray,
    TimeProfiler,
    UPSAMPLERS,
)
from mon.dataset import build_dataloader, DataLoader, transform as T
from mon.ops import ImageUpsampler, to_image_array, write_image, write_bbox
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
        # These attributes will be initialized later to avoid a long initialization time
        self._transforms: T.Compose = None
        self._upsampler: ImageUpsampler = None

    @override
    def _setup(self):
        """Setup the runner ready for inference."""
        config = self.config

        # Setup environment
        config.output_dir.mkdir(exist_ok=True, parents=True)
        sys_ctx.set_random_seed(config.seed)

        # Define model
        self._init_model()

        # Define transforms & upsampler
        self._init_transforms()
        self._init_upsampler()

    @abstractmethod
    def _init_transforms(self):
        """Initialize ``self._transforms`` attribute for pre-processing the
        input data.
        """
        pass

    def _init_upsampler(self):
        """Initialize ``self._upsampler`` attribute for upsampling the output
        images if necessary.
        """
        config = self.config

        if (
            config.strategy in [Strategy.RESIZE]
            and config.upscale
            and config.upsampler
        ):
            # Only build the upsampler if the strategy is RESIZE and upscaling
            # is requested, since other strategies (e.g., NATIVE, PATCH) do not
            # require upsampling
            upsampler: ImageUpsampler = UPSAMPLERS.build(**self.config.upsampler)
        else:
            upsampler = None

        self._upsampler = upsampler

    # --- Creation ---
    @classmethod
    def from_cli(cls, prompt: bool = False, **kwargs) -> "Predictor":
        """Create an instance of Predictor from command-line arguments.

        Args:
            prompt (bool, optional): Whether to prompt the user for input if
                necessary. Defaults to False.
        """
        config_ctx = ConfigContext.from_cli(**kwargs)
        config = config_ctx.config_for(RunMode.PREDICT, prompt=prompt)
        return cls(config)

    # --- Control ---
    def predict(self):
        """Predict the output of the model."""
        config = self.config

        # 1. Setup
        self._setup()
        # Validate that all necessary components are initialized
        if self._model is None and self.verbose:
            log(f"model has not been initialized.")
        if self._transforms is None and self.verbose:
            log(f"transforms have not been initialized.")

        # 2. Summarize the current run
        if config.verbose:
            self._log_summary()

        # 3. Run benchmark (if requested)
        if config.benchmark:
            self._benchmark()

        # 4. Main loop
        with create_progress_bar() as pbar:
            for data in pbar.track(
                sequence=config.data,
                total=len(config.data),
                description=f"[bright_yellow]Data"
            ):
                # 4.1. Update config
                self.config.infer_data = data

                # 4.2. Predict data
                timers = TimeProfiler()
                timers.total.tick()
                self._predict_data(data=data, pbar=pbar, timers=timers)
                timers.total.tock()

                # 4.3. Clean up
                self.config.infer_data = None

                # 4.4. Finish
                timers.print()

    # --- Prediction ---
    def _predict_data(self, data: Path, pbar: Progress, timers: TimeProfiler):
        """Predict the output of the model for a single data source.

        Args:
            data (Path): The path to the data point to predict.
            pbar (Progress): The progress bar to update during prediction.
            timers (TimeProfiler): The time profiler to record timing information
                during prediction.
        """
        # 1. Build dataset
        data_name, dataloader = self._build_dataloader(source=data)

        # 2. Main processing loop
        task = pbar.add_task(
            description=f"[bright_yellow]Predicting {data_name}",
            total=len(dataloader),
        )
        for i, datapoint in enumerate(dataloader):
            # 2.1. Predict step
            outputs = self._predict_step(datapoint=datapoint, timers=timers)

            # 2.2. Save results
            timers.postprocess.tick()
            if self.save:
                self._save(datapoint=datapoint, outputs=outputs)
            if self.save_debug:
                self._save_debug(datapoint=datapoint, outputs=outputs)
            timers.postprocess.tock()

            # 2.3 Clean up
            gc.collect()
            pbar.update(task, advance=1)
        pbar.remove_task(task)

    @abstractmethod
    def _predict_step(self, datapoint: TensorDict, timers: TimeProfiler) -> TensorDict:
        """Predict the output of the model for a single data point.

        Args:
            datapoint (TensorDict): The dictionary containing the data point
                to predict.
            timers (TimeProfiler): The time profiler to record timing information
                during prediction.

        Returns:
            TensorDict: The dictionary containing the prediction results.
        """
        pass

    def _build_dataloader(self, source: dict | Path) -> tuple[str, DataLoader]:
        """Initialize and return a dataset or dataloader.

        Args:
            source (dict | Path): A dataloader configuration dictionary or a
                source path.

        Returns:
            tuple[str, DataLoader]: A tuple containing the name of the
                dataloader and the dataset/dataloader object itself.
        """
        return build_dataloader(
            src=source,
            dataset_dir=self.config.data_dir,
            split=Split.TEST,
            transforms=self._transforms,
            keep_original=True,
            batch_size=1,
            num_workers=1,
        )

    # --- Output ---
    @abstractmethod
    def _save(self, datapoint: TensorDict, outputs: TensorDict):
        """Save the main prediction results to a file.

        Args:
            datapoint (TensorDict): The dictionary containing the input data.
            outputs (TensorDict): The dictionary containing the main results.
                Each key in the dictionary is a batch of results.
        """
        pass

    @abstractmethod
    def _save_debug(self, datapoint: TensorDict, outputs: TensorDict):
        """Save debugging results for visualization.

        Args:
            datapoint (TensorDict): The dictionary containing the input data.
            outputs (TensorDict): The dictionary containing the main results.
                Each key in the dictionary is a batch of results.
        """
        pass

    # --- Logging ---
    @override
    def _log_summary(self):
        """Log a summary of the current run."""
        self.config.log_summary()

    # --- Utilities ---
    def _save_batch_image(
        self,
        keys: list[str],
        datapoint: TensorDict,
        outputs: TensorDict,
        dirname: str = K.IMAGE_DIR,
        subdirname: str = "",
        use_stem: bool = False
    ):
        """Save a batch of image-based outputs.

        Args:
            keys (list[str]): The list of keys in the output dictionary that
                correspond to the images to be post-processed.
            datapoint (TensorDict): The dictionary containing the input data.
            outputs (TensorDict): The dictionary containing the main results.
                Each key in the dictionary is a batch of results.
            dirname (str, optional): The directory name for the output files.
                Defaults to K.IMAGE_DIR.
            subdirname (str, optional): Subdirectory name to append to the
                output path (e.g., 'debug'/'mask'). Defaults to "".
            use_stem (bool, optional): Whether to use the source file name stem
                for the output file name. If False, the output file name will
                be the same as the source file name. Defaults to False.
        """
        device = self.device
        upsampler = self._upsampler

        # Pre-extract the batches for the requested keys to avoid dict lookups
        # in the loop
        batch_metas = datapoint["meta"]
        batch_y_hr = datapoint[f"image_{K.ORIGINAL}"].to(device)  # For upsampler that needs high-res image (e.g., guided filter)
        batch_images_dict = {k: outputs[k] for k in keys if k in outputs}

        for i, meta in enumerate(batch_metas):
            path = meta["path"]
            y_hr = batch_y_hr[i:i + 1]
            size = Size.from_any(meta["imgsz"])

            for k, images in batch_images_dict.items():
                # Slice once per key per item
                image = images[i:i + 1]

                # Resize the image if needed
                imgsz = Size.from_any(image)
                if upsampler and (imgsz != size):
                    image = upsampler(x_lr=image, y_hr=y_hr, imgsz=size)

                # Convert to array
                if isinstance(image, Tensor):
                    image = to_image_array(image)
                if not isinstance(image, ndarray):
                    raise TypeError(f"expected image to be an array, "
                                    f"got {type(image).__name__}.")

                # Use the key as the stem only if requested (for debug)
                stem = k if use_stem else ""
                self._save_image(
                    image=image,
                    src_path=path,
                    dirname=dirname,
                    subdirname=subdirname,
                    stem=stem,
                )

    def _save_image(
        self,
        image: TensorOrArray,
        src_path: Path,
        dirname: str = K.IMAGE_DIR,
        subdirname: str = "",
        stem: str = "",
    ):
        """Save a single image for visualization.

        Args:
            image (TensorOrArray): The image to be saved, which can be a tensor
                or an array.
            src_path (Path): The source path, used to determine the output file
                path.
            dirname (str, optional): The directory name for the output file.
                Defaults to K.IMAGE_DIR.
            subdirname (str, optional): Subdirectory name to append to the
                output path (e.g., 'debug'/'mask'). Defaults to "".
            stem (str, optional): An optional string to be appended to the
                output file name for differentiation. If not provided, the
                output file name will be the same as the source file name.
                Defaults to "".
        """
        config = self.config

        # Determine the save path based on the source path and the provided parameters
        if stem:
            save_dir = config.resolve_save_dir(
                dirname=dirname,
                subdirname=subdirname,
                src_path=src_path,
            )
            save_path = save_dir / f"{src_path.stem}_{stem}{K.IMAGE_EXT}"
        else:
            save_path = config.resolve_save_file(
                dirname=dirname,
                subdirname=subdirname,
                src_path=src_path,
            )

        # Save the image
        write_image(image=image, path=save_path)

    def _save_batch_bboxes(
        self,
        bboxes: list[BBoxes],
        fmt: BBoxFormat,
        datapoint: TensorDict,
        outputs: TensorDict,
        dirname: str = K.LABEL_DIR,
        subdirname: str = "",
        use_stem: bool = False,
    ):
        """Save a batch of bounding box outputs.

        Args:
            bboxes (list[BBoxes]): The list of bounding boxes for each image in
                the batch.
            fmt (BBoxFormat): The format of the bounding boxes to be saved.
            datapoint (TensorDict): The dictionary containing the input data.
            outputs (TensorDict): The dictionary containing the main results.
                Each key in the dictionary is a batch of results.
            dirname (str, optional): The directory name for the output files.
                Defaults to K.ANN_DIR.
            subdirname (str, optional): Subdirectory name to append to the
                output path (e.g., 'debug'/'mask'). Defaults to "".
            use_stem (bool, optional): Whether to use the source file name stem
                for the output file name. If False, the output file name will
                be the same as the source file name. Defaults to False.
        """
        # Pre-extract the batches for the requested keys to avoid dict lookups
        # in the loop
        batch_metas = datapoint["meta"]

        # Process each item in the batch
        for i, meta in enumerate(batch_metas):
            path = meta["path"]
            bboxes = bboxes[i]
            stem = "label" if use_stem else ""

            self._save_bboxes(
                bboxes=bboxes,
                fmt=fmt,
                src_path=path,
                dirname=dirname,
                subdirname=subdirname,
                stem=stem,
            )

    def _save_bboxes(
        self,
        bboxes: BBoxes,
        fmt: BBoxFormat,
        src_path: Path,
        dirname: str = K.LABEL_DIR,
        subdirname: str = "",
        stem: str = "",
    ):
        """Save all bounding boxes for a single image.

        Args:
            bboxes (BBoxes): The predicted bounding boxes.
            fmt (BBoxFormat): The format of the bounding boxes to be saved.
            src_path (Path): The source path, used to determine the output file
                path.
            dirname (str, optional): The directory name for the output file.
                Defaults to K.ANN_DIR.
            subdirname (str, optional): Subdirectory name to append to the
                output path (e.g., 'debug'/'mask'). Defaults to "".
            stem (str, optional): An optional string to be appended to the
                output file name for differentiation. If not provided, the
                output file name will be the same as the source file name.
                Defaults to "".
        """
        config = self.config

        # Determine the save path based on the source path and the provided parameters
        save_dir = config.resolve_save_dir(
            dirname=dirname,
            subdirname=subdirname,
            src_path=src_path,
        )
        if stem:
            save_path = save_dir / f"{src_path.stem}_{stem}{K.ANN_EXT}"
        else:
            save_path = save_dir / f"{src_path.stem}{K.ANN_EXT}"

        # Save the bbox
        write_bbox(bbox=bboxes, path=save_path, fmt=fmt)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
