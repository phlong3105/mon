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

from numpy import ndarray
from rich.progress import Progress
from torch import nn, Tensor

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
    UPSAMPLERS,
)
from mon.dataset import (
    build_dataloader,
    DataLoader,
    Dataset,
    transform as T,
)
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
        # These attributes will be initialized later to avoid a long initialization time
        self._transforms: T.Compose | None = None
        self._upsampler: nn.Module | None = None

    @abstractmethod
    def _init_transforms(self):
        """Initialize ``self._transforms`` attribute."""
        pass

    def _init_upsampler(self):
        """Initialize ``self._upsampler`` attribute for upsampling the output
        images if necessary.
        """
        config = self.config

        if config.eval_resize and config.upsampler:
            # Only build the upsampler if ``eval_resize`` is True and an
            # upsampler config is provided
            upsampler = UPSAMPLERS.build(**self.config.upsampler).to(self.device)
        else:
            upsampler = None

        self._upsampler = upsampler

    def _init_data(
        self,
        source: DictLike | PathLike,
        split: SplitLike = Split.TEST,
    ) -> tuple[str, Dataset | DataLoader]:
        """Initialize and return a dataset or dataloader.

        Args:
            source (DictLike | PathLike): A dataset/dataloader configuration
                dictionary or a source path.
            split (SplitLike, optional): The data split to use.
                Defaults to Split.TEST.

        Returns:
            tuple[str, Dataset | DataLoader]: A tuple containing the name of the
                dataset/dataloader and the dataset/dataloader object itself.
        """
        # Apply pre-processing transforms inside the dataset or dataloader
        name, dataloader = build_dataloader(
            src=source,
            dataset_dir=self.config.data_dir,
            split=split,
            transforms=self.transforms,
            keep_original=self.keep_original,
            batch_size=1,
        )

        # Validate
        if name is None:
            raise RuntimeError(f"Failed to build dataset/dataloader from source: {source}.")
        if dataloader is None:
            raise RuntimeError(f"Failed to build dataloader from source: {source}.")

        # Return the name and dataloader
        return name, dataloader

    # --- Properties ---
    @property
    def keep_original(self) -> bool:
        """Whether to keep the original data alongside the transformed data."""
        return True

    @property
    def transforms(self) -> T.Compose | None:
        """Return the transforms object."""
        return self._transforms

    @property
    def upsampler(self) -> nn.Module | None:
        """Return the upsampler object."""
        return self._upsampler

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

        # 4. Define transforms & upsampler (if eval_resize is True)
        self._init_transforms()
        self._init_upsampler()
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
        data_name, dataloader = self._init_data(source=data, split=Split.TEST)

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

            pbar.update(task, advance=1)
        pbar.remove_task(task)

    @abstractmethod
    def _predict_step(self, datapoint: dict[str, Any], timers: TimeProfiler) -> dict[str, Any]:
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
    def _save(self, datapoint: dict[str, Any], outputs: dict[str, Any]):
        """Save the main prediction results to a file.

        Args:
            datapoint (dict): The dictionary containing the input data.
            outputs (dict): The dictionary containing the main prediction results.
                Each key in the dictionary is a batched of prediction results.
        """
        pass

    @abstractmethod
    def _save_debug(self, datapoint: dict[str, Any], outputs: dict[str, Any]):
        """Save debugging results for visualization.

        Args:
            datapoint (dict): The dictionary containing the input data.
            outputs (dict): The dictionary containing the main prediction results.
                Each key in the dictionary is a batched of prediction results.
        """
        pass

    # --- Utilities ---
    def _save_batch_image(
        self,
        keys: list[str],
        datapoint: dict[str, Any],
        outputs: dict[str, Any],
        dirname: str = K.PRED_DIR,
        subdirname: str = "",
        use_stem: bool = False
    ):
        """Save a batch of image-based outputs.

        Args:
            keys (list[str]): The list of keys in the output dictionary that
                correspond to the images to be post-processed.
            datapoint (dict): The dictionary containing the input data.
            outputs (dict): The dictionary containing the main prediction results.
                Each key in the dictionary is a batched of prediction results.
            dirname (str): The directory name for the output files.
                Defaults to K.PRED_DIR.
            subdirname (str, optional): Subdirectory name to append to the
                output path (e.g., 'debug'/'mask'). Defaults to "".
            use_stem (bool, optional): Whether to use the source file name stem
                for the output file name. If False, the output file name will
                be the same as the source file name. Defaults to False.
        """
        device = self.device
        upsampler = self.upsampler

        # Pre-extract the batches for the requested keys to avoid dict lookups
        # in the loop
        metas = datapoint.get("meta", [])
        batch_y_hr = datapoint[f"image_{K.ORIGINAL}"].to(device)  # For upsampler that needs high-res image (e.g., guided filter)
        batch_images_dict = {k: outputs[k] for k in keys if k in outputs}

        for i, meta_i in enumerate(metas):
            path = Path(meta_i["path"])
            y_hr = batch_y_hr[i:i + 1]
            size = Size.from_value(meta_i["imgsz"])

            for k, images in batch_images_dict.items():
                # Slice once per key per item
                image = images[i:i + 1]

                # Resize the image if needed
                imgsz = Size.from_value(image)
                if (upsampler is not None) and (imgsz != size):
                    image = upsampler(x_lr=image, y_hr=y_hr, imgsz=size)["x_hr"]

                # Convert to array
                if isinstance(image, Tensor):
                    image = to_image_array(image)
                if not isinstance(image, ndarray):
                    raise TypeError(
                        f"Expected 'image' to be an array, "
                        f"but got {type(image).__name__}."
                    )

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
        dirname: str = K.PRED_DIR,
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
                Defaults to K.PRED_DIR.
            subdirname (str): Subdirectory name to append to the output path
                (e.g., 'debug'/'mask'). Defaults to "".
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

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
