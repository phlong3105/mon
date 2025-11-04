#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements TensorMoG model for background subtraction.

References:
    - Paper: "TensorMoG: A Tensor-Driven Gaussian Mixture Model with Dynamic Scene
      Adaptation for Background Modeling," Sensors 2020.
"""

__all__ = [
    "TensorMOG",
]

import torch

from mon.constants import MODELS
from mon.core import Enum, MLType, ModelMixin, nn, Path, Task

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


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
        device: Device to run the model on (default is CPU).

    Attributes:
        min_var: Minimum variance for Gaussian components.
        max_var: Maximum variance for Gaussian components.
        mix_mu: Mean values of Gaussians ``[1, 3, height, width, num_gaussians]``.
        mix_var: Variance of Gaussians ``[1, 1, height, width, num_gaussians]``.
        mix_weight: Weights of Gaussians ``[1, 1, height, width, num_gaussians]``.
        mix_key: Keys for sorting Gaussians ``[1, 1, height, width, num_gaussians]``.
    """

    def __init__(
        self,
        height          : int,
        width           : int,
        num_gaussians   : int   = 3,
        learning_rate   : float = 0.02,
        matching_thres  : float = 2 * 2,
        background_thres: float = 0.6,
        device          : torch.device = torch.device("cpu"),
    ):
        self.height = height
        self.width  = width
        self.device = device

        # Initialize constant parameters
        self.num_gaussians    = num_gaussians
        self.learning_rate    = learning_rate
        self.matching_thres   = matching_thres
        self.background_thres = background_thres
        self.min_var          = 4 * 4 * 3
        self.max_var          = 8 * 8 * 3

        # Initialize the variables
        self.mix_mu     = torch.zeros((1, 3, self.height, self.width, self.num_gaussians), device=self.device)
        self.mix_var    = torch.zeros((1, 1, self.height, self.width, self.num_gaussians), device=self.device)
        self.mix_weight = torch.zeros((1, 1, self.height, self.width, self.num_gaussians), device=self.device)
        self.mix_key    = torch.zeros((1, 1, self.height, self.width, self.num_gaussians), device=self.device)

    def update(self, image: torch.Tensor):
        """Updates the background model."""
        # Ensure input is in BCHW format and normalized
        frame = image.to(self.device).float().unsqueeze(-1)      # [1, 3, H, W, 1]
                                                                 
        # Compute difference and matching mask                   
        d     = frame - self.mix_mu                              # [1, 3, H, W, K]
        d_sum = torch.sum(d * d, dim=1, keepdim=True)            # [1, 1, H, W, K]
        mask  = d_sum < (self.matching_thres * self.mix_var)     # [1, 1, H, W, K]

        # Identify matched and unmatched components
        taken       = torch.any(mask, dim=-1, keepdim=True)      # [1, 1, H, W, 1]
        match_idx   = torch.argmax(mask * self.mix_key, dim=-1)  # [1, 1, H, W]
        replace_idx = torch.argmin(self.mix_key, dim=-1)         # [1, 1, H, W]

        u_mask  = self._one_hot(match_idx, self.num_gaussians)   * taken.float()     # [1, 1, H, W, K]
        r_mask  = self._one_hot(replace_idx, self.num_gaussians) * (~taken).float()  # [1, 1, H, W, K]
        nr_mask = 1.0 - r_mask  # [1, 1, H, W, K]

        # Update weights
        w_tmp = self.mix_weight * (1.0 - self.learning_rate) + u_mask * self.learning_rate
        w_tmp = nr_mask * w_tmp + r_mask * self.learning_rate
        self.mix_weight = w_tmp / torch.sum(w_tmp, dim=-1, keepdim=True)

        # Update means
        self.mix_mu = nr_mask * (self.mix_mu + u_mask * d * self.learning_rate) + r_mask * frame

        # Update variances
        dv = u_mask * (d_sum - self.mix_var * 3)
        self.mix_var = torch.clamp(self.mix_var + dv * self.learning_rate, self.min_var, self.max_var)

        # Update keys
        self.mix_key = self.mix_weight / torch.sqrt(self.mix_var)

    def get_background(self) -> torch.Tensor:
        """Generate the background image based on the model.

        Returns:
            Background image of shape ``[1, 3, height, width]``, normalized.
        """
        wmax_ind = torch.argmax(self.mix_key, dim=-1)           # [1, 1, H, W]
        wmax_hot = self._one_hot(wmax_ind, self.num_gaussians)  # [1, 1, H, W, K]
        bg       = torch.sum(wmax_hot * self.mix_mu, dim=-1)    # [1, 3, H, W]
        return bg.clamp(0, 1)

    def get_foreground(self, image: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
        """Generate a foreground mask by comparing the image to the background.

        Args:
            image: Input image of shape ``[1, 3, height, width]``, normalized.
            background: Background image of shape ``[1, 3, height, width]``, normalized.

        Returns:
            torch.Tensor: Foreground mask of shape ``[1, 1, height, width]``, normalized.
        """
        diff = torch.abs(image - background)                        # [1, 3, H, W]
        mask = torch.any(diff > (80 / 255.0), dim=1, keepdim=True)  # [1, 1, H, W]
        return mask.float()

    def estimate_entropy(self) -> float:
        """Estimate the model's uncertainty (entropy).

        Returns:
            Average entropy across all pixels.
        """
        tmp     = self.mix_weight + (torch.abs(self.mix_weight) < 1e-12).float()
        entropy = -torch.sum(tmp * torch.log(tmp)) / (self.height * self.width)
        return entropy.item()

    def estimate_entropy_mask(self) -> torch.Tensor:
        """Estimate per-pixel entropy of the model.

        Returns:
            Entropy mask of shape ``[1, 1, height, width]``.
        """
        weights = torch.max(self.mix_weight, dim=-1)[0]  # [1, 1, H, W]
        tmp     = weights + (torch.abs(weights) < 1e-9).float()
        return -tmp * torch.log(tmp)

    def estimate_noise_ratio(self) -> float:
        """Estimate the noise ratio based on entropy.

        Returns:
            float: Average noise ratio across all pixels.
        """
        return torch.sum(self.estimate_entropy_mask()) / (self.height * self.width).item()

    def _one_hot(self, indices: torch.Tensor, depth: int) -> torch.Tensor:
        """Create a one-hot encoded tensor from indices.

        Args:
            indices: Tensor of indices to encode ``[1, 1, H, W]``.
            depth: Number of classes for one-hot encoding.

        Returns:
            One-hot encoded tensor ``[1, 1, H, W, depth]``.
        """
        shape        = indices.shape + (depth,)
        one_hot      = torch.zeros(shape, device=self.device)
        indices_flat = indices.flatten()
        one_hot      = one_hot.reshape(-1, depth).scatter_(1, indices_flat.unsqueeze(-1), 1.0)
        return one_hot.reshape(shape)


class HVR:
    """High-Variation Removal background subtraction model using PyTorch.

    Manages a Mixture of Gaussians (MOG) model with state-based updates to handle
    background stability and anomalies. All image tensors are in BCHW format
    and normalized to ``[0, 1]``.

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
        background: Current background image ``[1, 3, height, width]``.
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
        device: Device to run the model on (default is CPU).
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
        matching_thres   : float = 4.0,
        background_thres : float = 0.6,
        num_updates      : int   = 30,
        tau_rate         : float = 0.01,
        tau_updating_rate: float = 0.025,
        device           : torch.device = torch.device("cpu"),
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
        self.device            = device

        # Initialize state and variables
        self.state      = self.State.UPDATE
        self.counter    = 0
        self.tau        = 0.0
        self.background = torch.zeros((1, 3, height, width), device=self.device)
        self.background_model = MOG(
            height           = height,
            width            = width,
            num_gaussians    = num_gaussians,
            learning_rate    = learning_rate,
            matching_thres   = matching_thres,
            background_thres = background_thres,
        )

    def update(self, image: torch.Tensor):
        """Update the model with a new image.

        Args:
            image: Input image of shape ``[1, 3, height, width]``, normalized.

        Raises:
            ValueError: If image shape does not match expected dimensions.
        """
        if image.shape != (1, 3, self.height, self.width):
            raise ValueError(f"Expected [image]'s shape (1, 3, {self.height}, {self.width}), got {image.shape}")

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
                self.background = self.background_model.get_background()
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

    def get_background(self) -> torch.Tensor:
        """Get the current background image.

        Returns:
            Background image of shape ``[1, 3, height, width]``, normalized.
        """
        if self.state == self.State.UPDATE:
            self.background = self.background_model.get_background()
        return self.background

    def get_foreground(self, image: torch.Tensor) -> torch.Tensor:
        """Generate a foreground mask by comparing the image to the background.

        Args:
            image: Input image of shape ``[1, 3, height, width]``, normalized.

        Returns:
            Foreground mask of shape ``[1, 1, height, width]``, normalized.

        Raises:
            ValueError: If image shape does not match expected dimensions.
        """
        if image.shape != (1, 3, self.height, self.width):
            raise ValueError(f"Expected [image]'s shape (1, 3, {self.height}, {self.width}), got {image.shape}")
        return self.background_model.get_foreground(image, self.background)

    def estimate_entropy(self) -> float:
        """Estimate the model's uncertainty.

        Returns:
            float: Average entropy across all pixels.
        """
        return self.background_model.estimate_entropy()


# ----- Model -----
@MODELS.register(name="tensormog", arch="tensormog")
class TensorMOG(nn.Module, ModelMixin):
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
        - Paper: "TensorMoG: A Tensor-Driven Gaussian Mixture Model with Dynamic
          Scene Adaptation for Background Modeling," Sensors 2020.
    """

    arch     : str          = "tensormog"
    name     : str          = "tensormog"
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
    
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.hvr.update(image)
        background = self.hvr.get_background()
        foreground = self.hvr.get_foreground(image)
        return foreground, background
