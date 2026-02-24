#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ZS-N2N Models.

References:
    - Paper: "Zero-Shot Noise2Noise: Efficient Image Denoising without any
      Data," CVPR 2023.
    - Code: https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing#scrollTo=Srf0GQTYrkxA
"""

from __future__ import annotations

__all__ = [
    "ZSN2N",
]

import sys

import torch
from torch import nn, Tensor

from mon.core import MODELS, Path, Task
from mon.cv.ops import pair_downsample
from mon.nn import ModelRegisterMixin

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]
if str(current_dir) not in sys.path:
    # Add the project root to sys.path so 'import zero_dce' works
    # even if you run this script from inside the folder
    sys.path.append(str(current_dir))

try:
    # Works when running as a module: python -m zero_dce.predict
    from .module import DenoiseNetwork
except ImportError:
    # Works when running as a script: python predict.py
    from module import DenoiseNetwork


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

@MODELS.register(name="zsn2n")
class ZSN2N(ModelRegisterMixin, nn.Module):
    """ZS-N2N model for zero-shot image denoising.

    References:
        - Paper: "Zero-Shot Noise2Noise: Efficient Image Denoising without any
          Data," CVPR 2023.
        - Code: https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing#scrollTo=Srf0GQTYrkxA
    """

    arch: str = "zsn2n"
    name: str = "zsn2n"
    tasks: list[Task] = [Task.RESTORE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 48,
        epochs: int = 3000,
        device: torch.device = torch.device("cpu"),
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            in_channels (int, optional): Number of input channels.
                Defaults to 3.
            hidden_dim (int): Number of channels in the hidden layers.
                Defaults to 48.
            epochs (int, optional): Number of optimization epochs for the network.
                Defaults to 3,000.
            device (torch.device, optional): Device to use for computation.
                Defaults to torch.device("cpu").
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        # Satisfy PyTorch's empty signature first.
        super().__init__()
        # Initialize RegistrableMixin
        # ModelRegisterMixin.__init__(self, name=name)

        # Assign attributes
        self.verbose = verbose
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.device = device

     # --- Callable & Context Manager ---
    def forward(
        self,
        image: Tensor,
        epochs: int | None = None,
        save_debug: bool = False,
    ) -> dict:
        """Forward the input through the network.

        Args:
            image (Tensor): Image tensor of shape (B, 3, H, W) and values
                ranging from 0.0 to 1.0.
            epochs (int, optional): Number of optimization epochs for the network.
                Defaults to None
            save_debug (bool, optional): If True, return intermediate results
                for debugging. Defaults to False.

        Returns:
            dict: Dictionary containing the enhanced image tensor and
                intermediate results for debugging.
        """
        noisy_image = image
        epochs = epochs or self.epochs

        # 1. Create the denoising network
        model = DenoiseNetwork(
            in_channels=self.in_channels,
            hidden_dim=self.hidden_dim
        ).to(self.device)

        # 2. Move inputs to the corresponding device
        image = image.to(self.device)

        # 3. Define optimizer & losses
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
        L = nn.MSELoss().to(self.device)

        # 4. Optimize the network
        noisy1 = None
        noisy2 = None
        pred1 = None
        pred2 = None
        denoised1 = None
        denoised2 = None

        model.train()
        for i in range(epochs):
            noisy1, noisy2 = pair_downsample(noisy_image)
            pred1 = noisy1 - model(noisy1)
            pred2 = noisy2 - model(noisy2)
            l_res = 0.5 * (L(noisy1, pred2) + L(noisy2, pred1))

            noisy_denoised = noisy_image - model(noisy_image)
            denoised1, denoised2 = pair_downsample(noisy_denoised)
            l_cons = 0.5 * (L(pred1, denoised1) + L(pred2, denoised2))
            loss = l_res + l_cons

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # 5. Final denoising step
        model.eval()
        with torch.no_grad():
            restored = torch.clamp(noisy_image - model(noisy_image),0,1)
            restored = restored.detach().cpu()

        # 6. Return final and intermediate results for debugging
        outputs = { "restored": restored }
        if save_debug:
            outputs |= {
                "noisy1": noisy1.detach().cpu(),
                "noisy2": noisy2.detach().cpu(),
                "pred1": pred1.detach().cpu(),
                "pred2": pred2.detach().cpu(),
                "denoised1": denoised1.detach().cpu(),
                "denoised2": denoised2.detach().cpu(),
            }
        return outputs

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
