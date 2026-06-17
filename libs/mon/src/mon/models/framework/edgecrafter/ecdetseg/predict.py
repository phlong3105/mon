#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Runners.

This module provides prediction runner classes for EdgeCrafter models.
"""

from __future__ import annotations

__all__ = [
    "ECDet_Predictor",
]

import argparse

import numpy as np
import torch
from tensordict import TensorDict
from typing_extensions import override

from mon.core import (
    BBoxes,
    BBoxFormat,
    ConfigContext,
    K,
    MODELS,
    Path,
    PREDICTORS,
    RunMode,
    TimeProfiler,
    Weights,
)
from mon.dataset import transform as T
from mon.runners import Predictor

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region PREDICTOR
# ==============================================================================

@PREDICTORS.register(name="ecdet")
class ECDet_Predictor(Predictor):
    """Predictor for EdgeCrafter models."""

    # --- Lifecycle & Initialization ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        config = self.config
        device = self.device

        args = config.extra
        args.cfg_path = config.config_file.parent / args.cfg_paths

        # Resolve model's weights
        if config.get("resume"):
            args.resume = Weights(path=Path(config.resume)).rectify_path(root=config.root)
        else:
            args.resume = None
        self.config.extra = args

        weights = args.resume or config.weights or config.finetune

        model = MODELS.build(**config.model | {"weights": weights})
        model = model.to(device)
        model.eval()
        self._model = model

    @override
    def _init_transforms(self):
        """Initialize ``self._transforms`` attribute for pre-processing the
        input data.
        """
        config = self.config

        transforms = T.Compose(
            [
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    normalization="min_max",
                ),
                T.ToTensorV2(transpose_mask=True),
            ],
            is_check_shapes=False,
        )

        if config is not None:
            if config.use_resize:
                imgsz = config.imgsz
                resize = T.ResizeDivisibleBy(height=imgsz.h, width=imgsz.w, divisor=32)
                transforms = resize + transforms

        self._transforms = transforms

    # --- Creation ---
    @classmethod
    def from_cli(cls, prompt: bool = False, **kwargs) -> "Predictor":
        """Create an instance of Predictor from command-line arguments.

        Args:
            prompt (bool, optional): Whether to prompt the user for input if
                necessary. Defaults to False.
        """
        # Additional model's arguments for training
        parser = argparse.ArgumentParser(description="edgecrafter")
        parser.add_argument("-r", "--resume", type=str, help="Resume from checkpoint")
        parser.add_argument("-t", "--thresh", type=float, default=0.4)
        args = vars(parser.parse_args())
        kwargs |= args

        config_ctx = ConfigContext.from_cli(**kwargs)
        config = config_ctx.config_for(RunMode.PREDICT, prompt=prompt)
        return cls(config)

    # --- Prediction ---
    @override
    @torch.inference_mode()
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
        config = self.config
        device = self.device

        # 1. Prepare inputs
        timers.preprocess.tick()
        datapoint = datapoint.to(device)
        meta = datapoint["meta"][0]
        imgsz = meta["imgsz"]
        timers.preprocess.tock()

        # 2. Inference
        timers.infer.tick()
        outputs = self.model(data=datapoint, imgsz=imgsz, save_debug=self.save_debug)
        timers.infer.tock()

        return outputs

    # --- Output ---
    @override
    def _save(self, datapoint: TensorDict, outputs: TensorDict):
        """Save the main prediction results to a file.

        Args:
            datapoint (TensorDict): The dictionary containing the input data.
            outputs (TensorDict): The dictionary containing the main results.
                Each key in the dictionary is a batch of results.
        """
        bboxes = self._postprocess(datapoint, outputs)

        self._save_batch_bboxes(
            bboxes=bboxes,
            fmt=BBoxFormat.CXCYWHN,
            datapoint=datapoint,
            outputs=outputs,
            dirname=K.LABELS_DIR,
            subdirname="",
        )

    @override
    def _save_debug(self, datapoint: TensorDict, outputs: TensorDict):
        """Save debugging results for visualization.

        Args:
            datapoint (TensorDict): The dictionary containing the input data.
            outputs (TensorDict): The dictionary containing the main results.
                Each key in the dictionary is a batch of results.
        """
        pass

    # --- Utilities ---
    def _postprocess(self, datapoint: TensorDict, outputs: TensorDict) -> list[BBoxes]:
        """Filter the raw outputs of the model based on the confidence threshold.

        Args:
            datapoint (TensorDict): The dictionary containing the input data.
            outputs (TensorDict): The dictionary containing the raw prediction
                results. Each key in the dictionary is a batch of prediction
                results.

        Returns:
            list[BBoxes]: A list of ``BBoxes`` instances containing the
                post-processed bounding boxes for each image in the batch.
        """
        thresh = self.config.thresh

        # Extract raw outputs
        outputs = outputs.cpu().numpy()
        batch_labels = outputs["labels"]
        batch_boxes = outputs["boxes"]
        batch_scores = outputs["scores"]

        # Process batch
        bboxes = []
        batch_metas = datapoint["meta"]
        for i, meta in enumerate(batch_metas):
            # Filter outputs by confidence threshold
            keep = batch_scores[i] > thresh
            labels = batch_labels[i][keep]
            boxes = batch_boxes[i][keep]
            scores = batch_scores[i][keep]

            # Merge all outputs to a single array of format:
            # [cx, cy, w, h, angle, class_id, score, track_id]
            imgsz = meta["imgsz"]
            bbox = []
            for j in range(len(labels)):
                x1, y1, x2, y2 = boxes[j]
                angle = 0.0
                class_id = labels[j]
                score = scores[j]
                track_id = -1.0
                bbox.append([x1, y1, x2, y2, angle, class_id, score, track_id])
            bbox = np.array(bbox, dtype=np.float32)
            bboxes.append(BBoxes.from_any(bbox=bbox, imgsz=imgsz, fmt=BBoxFormat.XYXY))

        return bboxes

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
