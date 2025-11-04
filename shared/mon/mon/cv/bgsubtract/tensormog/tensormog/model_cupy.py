#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""TensorMoG model for background subtraction.

References:
    - Paper: "TensorMoG: A Tensor-Driven Gaussian Mixture Model with Dynamic Scene
      Adaptation for Background Modeling," Sensors 2020.
"""

__all__ = [
    "TensorMOGCuPy"
]

from typing import Literal

import cupy as cp
import numpy as np
import torch
import torch.nn as nn
from mon.core.model import ModelMixin

from mon.constants import MLType, MODELS, Task
from mon.core.enum import Enum
from mon.core.pathlib import Path

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Module -----
class MOG:
    """Mixture of Gaussians model for background subtraction.

    Args:
        height: Height of the input image.
        width: Width of the input image.
        num_gaussians: Number of Gaussian components per pixel.
        learning_rate: Rate at which model parameters are updated.
        matching_thres: Threshold for matching a pixel to a Gaussian.
        background_thres: Threshold for background classification.

    Attributes:
        min_var: Minimum variance for Gaussian components.
        max_var: Maximum variance for Gaussian components.
        mix_mu: Mean values of Gaussians ``[height, width, 3, num_gaussians]``.
        mix_var: Variance of Gaussians ``[height, width, 1, num_gaussians]``.
        mix_weight: Weights of Gaussians ``[height, width, 1, num_gaussians]``.
        mix_key: Keys for sorting Gaussians ``[height, width, 1, num_gaussians]``.
    """

    def __init__(
        self,
        height          : int,
        width           : int,
        num_gaussians   : int   = 3,
        learning_rate   : float = 0.02,
        matching_thres  : float = 2 * 2,
        background_thres: float = 0.6,
    ):
        self.height = height
        self.width  = width

        # Initialize constant parameters
        self.num_gaussians    = num_gaussians
        self.learning_rate    = learning_rate
        self.matching_thres   = matching_thres
        self.background_thres = background_thres
        self.min_var          = 4 * 4 * 3
        self.max_var          = 8 * 8 * 3

        # Initialize the variables
        self.mix_mu     = cp.zeros((self.height, self.width, 3, self.num_gaussians))
        self.mix_var    = cp.zeros((self.height, self.width, 1, self.num_gaussians))
        self.mix_weight = cp.zeros((self.height, self.width, 1, self.num_gaussians))
        self.mix_key    = cp.zeros((self.height, self.width, 1, self.num_gaussians))

    def update(self, image: np.ndarray):
        """Updates the background model."""
        frame = cp.asarray(image, dtype=cp.float32)[..., None]

        # Compute difference and matching mask
        d     = frame - self.mix_mu
        d_sum = cp.sum(d * d, axis=2, keepdims=True)
        mask  = d_sum < (self.matching_thres * self.mix_var)

        # Identify matched and unmatched components
        taken       = cp.any(mask, axis=-1, keepdims=True)
        match_idx   = cp.argmax(mask * self.mix_key, axis=-1)
        replace_idx = cp.argmin(self.mix_key, axis=-1)
        u_mask      = cp.logical_and(taken, self._one_hot(match_idx, self.num_gaussians))
        r_mask      = cp.logical_and(~taken, self._one_hot(replace_idx, self.num_gaussians))
        nr_mask     = ~r_mask

        # Update weights
        w_tmp = self.mix_weight * (1.0 - self.learning_rate) + u_mask * self.learning_rate
        w_tmp = nr_mask * w_tmp + r_mask * self.learning_rate
        self.mix_weight = w_tmp / cp.sum(w_tmp, axis=-1, keepdims=True)

        # Update means
        self.mix_mu  = nr_mask * (self.mix_mu + u_mask * d * self.learning_rate) + r_mask * frame

        # Update variances
        dv           = u_mask * (d_sum - self.mix_var * 3)
        self.mix_var = cp.clip(self.mix_var + dv * self.learning_rate, self.min_var, self.max_var)

        # Update keys
        self.mix_key = self.mix_weight / cp.sqrt(self.mix_var)

    def get_background(self, mode: Literal["numpy", "cupy"] = "numpy") -> np.ndarray:
        """Generate the background image based on the model.

        Args:
            mode: Output format, either ``"numpy"`` or ``"cupy"``.
                Defaults to ``"numpy"``.

        Returns:
            cp.ndarray or np.ndarray: Background image of shape ``[height, width, 3]``.
        """
        wmax_ind = cp.argmax(self.mix_key, axis=-1)
        wmax_hot = self._one_hot(wmax_ind, self.num_gaussians)
        bg       = cp.sum(wmax_hot * self.mix_mu, axis=-1).astype(cp.uint8)
        return bg if mode == "cupy" else cp.asnumpy(bg)

    def get_foreground(self, image: np.ndarray, background: np.ndarray) -> np.ndarray:
        """Generate a foreground mask by comparing the image to the background.

        Args:
            image: Input image of shape ``[height, width, 3]``.
            background: Background image of shape ``[height, width, 3]``.

        Returns:
            np.ndarray: Foreground mask of shape ``[height, width]`` with values ``0`` or ``255``.
        """
        diff = cp.abs(cp.asarray(image, dtype=cp.uint8) - cp.asarray(background, dtype=cp.uint8))
        mask = cp.any(diff > 80, axis=-1)
        return cp.asnumpy(mask * 255).astype(np.uint8)

    def estimate_entropy(self) -> float:
        """Estimate the model's uncertainty (entropy).

        Returns:
            Average entropy across all pixels.
        """
        tmp     = self.mix_weight + (cp.abs(self.mix_weight) < 1e-12)
        entropy = -cp.sum(tmp * cp.log(tmp)) / (self.height * self.width)
        return float(cp.asnumpy(entropy))

    def estimate_entropy_mask(self) -> cp.ndarray:
        """Estimate per-pixel entropy of the model.

        Returns:
            cp.ndarray: Entropy mask of shape ``[height, width, 1]``.
        """
        weights = cp.max(self.mix_weight, axis=-1)
        tmp = weights + (cp.abs(weights) < 1e-9)
        return -tmp * cp.log(tmp)

    def estimate_noise_ratio(self) -> float:
        """Estimate the noise ratio based on entropy.

        Returns:
            float: Average noise ratio across all pixels.
        """
        return float(cp.sum(self.estimate_entropy_mask()) / (self.height * self.width))

    def _one_hot(self, indices: cp.ndarray, depth: int) -> cp.ndarray:
        """Create a one-hot encoded array from indices.

        Args:
            indices: Array of indices to encode.
            depth: Number of classes for one-hot encoding.

        Returns:
            cp.ndarray: One-hot encoded array.
        """
        shape        = indices.shape + (depth,)
        one_hot      = cp.zeros(shape, dtype=cp.float32)
        indices_flat = indices.ravel()
        one_hot.reshape(-1, depth)[cp.arange(indices_flat.size), indices_flat] = 1.0
        return one_hot


class HVR:
    """High-Variation Removal background subtraction model.

    Manages a Mixture of Gaussians (MOG) model with state-based updates to handle
    background stability and anomalies.

    Attributes:
        height: Height of the input image.
        width: Width of the input image.
        num_gaussians: Number of Gaussian components in the MOG model.
        num_updates: Number of frames required to trigger state transitions.
        state: Current state of the model (UPDATE, SUSPENSE, or TRAIN).
        counter: Frame counter for state transitions.
        tau: Average entropy (stability line).
        tau_rate: Entropy threshold rate for anomaly detection.
        tau_updating_rate: Rate for updating the stability line.
        background: Current background image.
        background_model: Underlying MOG model for background subtraction.

    Args:
        height: Height of the input image.
        width: Width of the input image.
        num_gaussians: Number of Gaussian components. Defaults to ``3``.
        learning_rate: Learning rate for the MOG model. Defaults to ``0.02``.
        matching_thres: Threshold for matching a pixel to a Gaussian. Defaults to ``2 * 2``.
        background_thres: Threshold for background classification. Defaults to ``0.6``.
        num_updates: Frames for state transitions. Defaults to ``30``.
        tau_rate: Entropy threshold rate. Defaults to ``0.01``.
        tau_updating_rate: Rate for updating tau. Defaults to ``0.025``.
    """

    class State(Enum):
        UPDATE   = "update"
        SUSPENSE = "suspense"
        TRAIN    = "train"

    def __init__(
        self,
        height           : int,
        width            : int,
        num_gaussians    : int   = 3,
        learning_rate    : float = 0.02,
        matching_thres   : float = 2 * 2,
        background_thres : float = 0.6,
        num_updates      : int   = 30,
        tau_rate         : float = 0.01,
        tau_updating_rate: float = 0.025,
    ):
        self.height            = height
        self.width             = width
        self.num_gaussians     = num_gaussians
        self.learning_rate     = learning_rate
        self.matching_thres    = matching_thres
        self.background_thres  = background_thres
        self.num_updates       = num_updates
        self.tau_rate          = tau_rate
        self.tau_updating_rate = tau_updating_rate

        # Initialize state and variables
        self.state      = self.State.UPDATE
        self.counter    = 0
        self.tau        = 0.0
        self.background = np.zeros((self.height, self.width , 3))
        self.background_model = MOG(
            height           = self.height,
            width            = self.width,
            num_gaussians    = self.num_gaussians,
            learning_rate    = self.learning_rate,
            matching_thres   = self.matching_thres,
            background_thres = self.background_thres,
        )

    def update(self, image: np.ndarray):
        """Update the model with a new image.

        Args:
            image: Input image of shape ``[height, width, 3]``.

        Raises:
            ValueError: If image shape does not match expected dimensions.
        """
        if image.shape != (self.height, self.width, 3):
            raise ValueError(f"Expected [image]'s shape ({self.height}, {self.width}, 3), got {image.shape}")

        if self.state == self.State.TRAIN:
            return  # TRAIN state is currently unused

        # Update MOG model and compute entropy
        self.background_model.update(image)
        entropy = self.background_model.estimate_entropy()

        # Update stability line (tau)
        self.tau = min(entropy, self.tau * (1 - self.tau_updating_rate) + entropy * self.tau_updating_rate)

        # Handle state transitions
        is_anomaly = entropy - self.tau >= self.tau_rate * self.tau
        self._handle_update_state(is_anomaly)
        if self.state == self.State.UPDATE:
            self._handle_update_state(is_anomaly)
        elif self.state == self.State.SUSPENSE:
            self._handle_suspense_state(is_anomaly)

    def _handle_update_state(self, is_anomaly: bool):
        """Handle logic for UPDATE state.

        Args:
            is_anomaly: Whether an anomaly is detected based on entropy.
        """
        if is_anomaly:
            self.counter += 1
            if self.counter >= self.num_updates:
                self.state      = self.State.SUSPENSE
                self.counter    = 0
                self.background = self.background_model.get_background(mode="cupy")
                self.background_model.learning_rate = max(self.background_model.learning_rate / 2, 0.001)
        else:
            self.counter = 0

    def _handle_suspense_state(self, is_anomaly: bool):
        """Handle logic for SUSPENSE state.

        Args:
            is_anomaly: Whether an anomaly is detected based on entropy.
        """
        if not is_anomaly:
            self.counter += 1
            if self.counter >= self.num_updates:
                self.state   = self.State.UPDATE
                self.counter = 0
                self.background_model.learning_rate = min(self.background_model.learning_rate * 2, 0.1)
        else:
            self.counter = 0

    def get_background(self) -> np.ndarray:
        """Get the current background image.

        Returns:
            np.ndarray: Background image of shape ``[height, width, 3]`` with dtype uint8.
        """
        if self.state == self.State.UPDATE:
            self.background = self.background_model.get_background(mode="numpy")
        if isinstance(self.background, cp.ndarray):
            self.background = self.background.get().astype(np.uint8)
        return self.background.astype(np.uint8)

    def get_foreground(self, image: np.ndarray) -> np.ndarray:
        """Generate a foreground mask by comparing the image to the background.

        Args:
            image: Input image of shape ``[height, width, 3]``.

        Returns:
            np.ndarray: Foreground mask of shape ``[height, width]`` with dtype ``uint8``.

        Raises:
            ValueError: If image shape does not match expected dimensions.
        """
        if image.shape != (self.height, self.width, 3):
            raise ValueError(f"Expected [image]'s shape ({self.height}, {self.width}, 3), got {image.shape}")
        return self.background_model.get_foreground(image, self.get_background())

    def estimate_entropy(self) -> float:
        """Estimate the model's uncertainty.

        Returns:
            float: Average entropy across all pixels.
        """
        return self.background_model.estimate_entropy()


# ----- Model -----
@MODELS.register(name="tensormog_cupy", arch="tensormog")
class TensorMOGCuPy(nn.Module, ModelMixin):
    """TensorMoG model for background subtraction.

    Args:
        height: Height of the input image. Defaults to ``512``.
        width: Width of the input image. Defaults to ``512``.
        num_gaussians: Number of Gaussian components. Defaults to ``3``.
        learning_rate: Learning rate for the MOG model. Defaults to ``0.02``.
        matching_thres: Threshold for matching a pixel to a Gaussian. Defaults to ``2 * 2``.
        background_thres: Threshold for background classification. Defaults to ``0.6``.
        num_updates: Frames for state transitions. Defaults to ``30``.
        tau_rate: Entropy threshold rate. Defaults to ``0.01``.
        tau_updating_rate: Rate for updating tau. Defaults to ``0.025``.
        
    References:
        - Paper: "TensorMoG: A Tensor-Driven Gaussian Mixture Model with Dynamic Scene
          Adaptation for Background Modeling," Sensors 2020.
    """

    arch     : str          = "tensormog"
    name     : str          = "tensormog_cupy"
    tasks    : list[Task]   = [Task.BGSUBTRACT, Task.VIDEO]
    mltypes  : list[MLType] = [MLType.INFERENCE]
    model_dir: Path         = root_dir
    zoo      : dict         = {}
    
    def __init__(
        self,
        height           : int   = 512,
        width            : int   = 512,
        num_gaussians    : int   = 3,
        learning_rate    : float = 0.02,
        matching_thres   : float = 2 * 2,
        background_thres : float = 0.6,
        num_updates      : int   = 30,
        tau_rate         : float = 0.01,
        tau_updating_rate: float = 0.025,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.height = height
        self.width  = width
        self.hvr    = HVR(
            height            = self.height,
            width             = self.width,
            num_gaussians     = num_gaussians,
            learning_rate     = learning_rate,
            matching_thres    = matching_thres,
            background_thres  = background_thres,
            num_updates       = num_updates,
            tau_rate          = tau_rate,
            tau_updating_rate = tau_updating_rate,
        )

    # ----- Initialize -----
    def init_weights(self, m: nn.Module):
        """Initializes the model's weights.
    
        Args:
            m: ``nn.Module`` to initialize weights for.
        """
        pass
    
    # ----- Forward Pass -----
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        """Performs forward pass of the model.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
    
        Returns:
            ``dict`` of predictions with ``"enhanced"`` keys.
        """
        x = datapoint["image"]
        self.hvr.update(x)
        background = self.hvr.get_background()
        foreground = self.hvr.get_foreground(x)
        return {
            "foreground": foreground,
            "background": background,
        }
    
    # ----- Predict -----
    def infer(self, datapoint: dict, timers: core.TimeProfiler = None, *args, **kwargs) -> dict:
        """Infers model output with optional processing.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
            timers: ``TimeProfiler`` for measuring time.

        Returns:
            ``dict`` of model predictions.
    
        Notes:
            Override for custom pre/post-processing; defaults to ``self.forward()``.
        """
        # Preprocess
        timers.preprocess.tick() if timers is not None else None
        image = datapoint["image"]
        if isinstance(image, torch.Tensor):
            image = types.image_to_array(image, True)

        h0, w0 = types.image_size(image)
        if h0 != self.height or w0 != self.width:
            image = geometry.resize(image, (self.height, self.width))
        timers.preprocess.tock() if timers is not None else None

        # Forward pass
        timers.infer.tick() if timers is not None else None
        outputs = self.forward(datapoint={"image": image})
        timers.infer.tock() if timers is not None else None
        
        # Postprocess
        timers.postprocess.tick() if timers is not None else None
        foreground = outputs["foreground"]
        background = outputs["background"]
        if h0 != self.height or w0 != self.width:
            foreground = geometry.resize(foreground, (h0, w0))
            background = geometry.resize(background, (h0, w0))
        timers.postprocess.tock() if timers is not None else None

        # Return
        return outputs | {
            "foreground": foreground,
            "background": background,
        }
