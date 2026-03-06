#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Runners.

This module provides prediction runner classes for Zero-DCE and Zero-DCE++ models.
"""

from __future__ import annotations

__all__ = [
    "ZeroDCE_Predictor",
    "ZeroDCEPP_Predictor",
]

from rich.progress import Progress
from typing_extensions import override

from mon.core import Path, Size, Split, TimeProfiler
from mon.dataset import build_dataloader, transform as T
from mon.runners import Predictor
from .model import zero_dce, zero_dce_pp

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region PREDICTOR
# ==============================================================================

class ZeroDCE_Predictor(Predictor):
    """Predictor for Zero-DCE models."""

    # --- Properties ---
    @override
    def init_model(self):
        """Initialize ``self.model`` attribute."""
        config = self.config
        device = self.device
        weights = config.weights or config.finetune

        model = zero_dce(**config.model | { "weights": weights})
        model = model.to(device)
        model.eval()
        self.model = model

    @override
    def init_transforms(self) -> T.Compose | None:
        """Initialize and return the transforms to be used for prediction."""
        config = self.config
        imgsz = Size.from_value(config.eval_imgsz)

        scale_factor = config.model.get("scale_factor")
        if scale_factor:
            imgsz = Size(height=imgsz.h // scale_factor, width=imgsz.w // scale_factor)

        self.transforms = T.Compose([
            T.ResizeDivisibleBy(height=imgsz.h, width=imgsz.w, divisor=32),
            T.Normalize(normalization="min_max"),
            T.ToTensorV2(transpose_mask=True),
        ])

    # --- Prediction ---
    @override
    def predict_data(self, data: Path | str, pbar: Progress, timers: TimeProfiler):
        """Predict the output of the model for a single data source.

        Args:
            data (Path | str): The path to the data point to predict.
            pbar (Progress): The progress bar to update during prediction.
            timers (TimeProfiler): The time profiler to record timing information
                during prediction.
        """
        config = self.config
        device = self.device
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
            # 2.1. Prepare inputs
            timers.preprocess.tick()
            meta = datapoint["meta"]
            path = Path(meta["path"])
            size0 = Size.from_value(meta["imgsz"])
            image = datapoint["image"]
            image = image.unsqueeze(0).to(device)
            timers.preprocess.tock()

            # 2.2. Inference
            timers.infer.tick()
            outputs = self.model(image, save_debug=save_debug)
            timers.infer.tock()

            # 2.3. Postprocess
            timers.postprocess.tick()
            self.save(path, outputs, size0)
            if save_debug:
                self.save_debug(path, outputs, size0)
            timers.postprocess.tock()

            pbar.update(task, advance=1)
        pbar.remove_task(task)

    # --- Utilities ---
    @override
    def save(self, path: Path, outputs: dict, size: Size):
        """Save the main prediction results to a file.

        Args:
            path (Path): The source path used to determine the output file path.
            outputs (dict): The dictionary containing the main prediction results.
            size (Size): The original size of the input image.
        """
        self.save_image(path, outputs["enhanced"], size)

    @override
    def save_debug(self, path: Path, outputs: dict, size: Size):
        """Save debugging results for visualization.

        Args:
            path (Path): The source path used to determine the output file path.
            outputs (dict): The dictionary containing the debugging results.
            size (Size): The original size of the input image.
        """
        pass


class ZeroDCEPP_Predictor(ZeroDCE_Predictor):
    """Predictor for Zero-DCE++ models."""

    # --- Properties ---
    @override
    def init_model(self):
        """Initialize ``self.model`` attribute."""
        config = self.config
        device = self.device
        weights = config.weights or config.finetune

        model = zero_dce_pp(**config.model | { "weights": weights})
        model = model.to(device)
        model.eval()
        self.model = model

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
