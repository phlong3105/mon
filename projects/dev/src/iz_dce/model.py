#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""IZ-DCE Models.

This module provides the IZ-DCE definition and pre-trained weights.

References:
    - Paper: "IZ-DCE: Implicit Zero-Reference Deep Curve Estimation"
    - Code: https://github.com/phlong3105/izdce
"""

from __future__ import annotations

__all__ = [
    "IZ_DCE",
    "IZ_DCE_ODE",
    "iz_dce",
    "iz_dce_ode",
]

import torch
from torch import nn, Tensor

from mon.core import is_weights_type, log, MODELS, Path, Task, WeightsLike
from mon.nn import ModelRegisterMixin
from .module import EnhanceFunction, EnhanceFunctionTime, ODEBlock

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class IZ_DCE(ModelRegisterMixin, nn.Module):
    """IZ-DCE model.

    References:
        - Paper: "IZ-DCE: Implicit Zero-Reference Deep Curve Estimation"
        - Code: https://github.com/phlong3105/izdce
    """

    arch: str = "iz_dce"
    name: str = "iz_dce"
    tasks: list[Task] = [Task.ENHANCE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        in_channels: int = 3,
        hidden_dim: int = 32,
        imgsz: int = 256,
        chunk_size: int = 100000,
        num_iter: int = 8,
        use_depth: bool = False,
        use_anscombe: bool = False,
        weights: WeightsLike | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model to use.
            in_channels (int, optional): Number of input channels. Defaults to 3.
            hidden_dim (int, optional): Hidden dimension. Defaults to 32.
            imgsz (int, optional): Downsample the input image to this size for
                encoding. Defaults to 256.
            chunk_size (int): Number of pixels to process at once.
                Defaults to 100,000.
            num_iter (int, optional): Number of iterations for curve estimation.
                Defaults to 8.
            use_depth (bool, optional): Whether to use depth as an additional
                input channel. Defaults to False.
            use_anscombe (bool, optional): Whether to apply the Anscombe
                transform to the input before denoising. Defaults to False.
            weights (WeightsLike, optional): Pre-trained weights to load.
                Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        # Satisfy PyTorch's empty signature first.
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose
        self.imgsz = imgsz

        # Define network
        self.enhance_func = EnhanceFunction(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            imgsz=imgsz,
            chunk_size=chunk_size,
            num_iter=num_iter,
            use_depth=use_depth,
            use_anscombe=use_anscombe,
            *args, **kwargs
        )

        # Load weights
        if is_weights_type(weights):
            self.load_state_dict(weights.state_dict())
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

    # --- Callable & Context Manager ---
    def forward(self, image: Tensor, depth: Tensor | None = None, *args, **kwargs) -> dict:
        """Forward the input through the network.

        Args:
            image (Tensor): Image tensor of shape (B, 3, H, W) and values
                ranging from 0.0 to 1.0.
            depth (Tensor, optional): Depth tensor of shape (B, 1, H, W) and
                values ranging from 0.0 to 1.0. Defaults to None.
        """
        return self.enhance_func(image, depth, *args, **kwargs)


class IZ_DCE_ODE(ModelRegisterMixin, nn.Module):
    """IZ-DCE-ODE model.

    References:
        - Paper: "IZ-DCE: Implicit Zero-Reference Deep Curve Estimation"
        - Code: https://github.com/phlong3105/izdce
    """

    arch: str = "iz_dce"
    name: str = "iz_dce_ode"
    tasks: list[Task] = [Task.ENHANCE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        in_channels: int = 3,
        hidden_dim: int = 32,
        imgsz: int = 128,
        chunk_size: int = 100000,
        use_depth: bool = False,
        use_anscombe: bool = False,
        tol: float = 1e-5,
        adjoint: bool = True,
        ode_options: dict | None = None,
        weights: WeightsLike | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model to use.
            in_channels (int, optional): Number of input channels. Defaults to 3.
            hidden_dim (int, optional): Hidden dimension. Defaults to 32.
            imgsz (int, optional): Downsample the input image to this size for
                encoding. Defaults to 128.
            chunk_size (int): Number of pixels to process at once.
                Defaults to 100,000.
            use_depth (bool, optional): Whether to use depth as input.
                Defaults to False.
            use_anscombe (bool, optional): Whether to apply the Anscombe
                transform to the input before denoising. Defaults to False.
            use_dopri5 (bool, optional): Whether to use the Dopri5 solver.
                Defaults to False.
            tol (float, optional): Tolerance for solver. Defaults to 1e-5.
            adjoint (bool, optional): Whether to use the adjoint method for
                backpropagation. Defaults to False.
            ode_options (dict, optional): Additional options to pass to the ODE
                solver. Defaults to None.
            weights (WeightsLike, optional): Pre-trained weights to load.
                Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        # Satisfy PyTorch's empty signature first.
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose
        self.imgsz = imgsz

        # Define network
        self.enhance_func = EnhanceFunctionTime(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            imgsz=imgsz,
            chunk_size=chunk_size,
            use_depth=use_depth,
            use_anscombe=use_anscombe,
            *args, **kwargs
        )
        self.ode_block = ODEBlock(self.enhance_func, tol, adjoint, ode_options)

        # Load weights
        if is_weights_type(weights):
            self.load_state_dict(weights.state_dict())
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

    # --- Callable & Context Manager ---
    def forward(
        self,
        image: Tensor,
        depth: Tensor | None = None,
        eval_time: Tensor | None = None,
        *args, **kwargs
    ) -> dict:
        """Forward the input through the network.

        Args:
            image (Tensor): Image tensor of shape (B, 3, H, W) and values
                ranging from 0.0 to 1.0.
            depth (Tensor, optional): Depth tensor of shape (B, 1, H, W) and
                values ranging from 0.0 to 1.0. Defaults to None.
            eval_time (Tensor, optional): Time steps at which to evaluate the
                ODE solution. If None, defaults to [0, 1]. Defaults to None.
        """
        # 1. Pre-process
        b, c, h, w = image.shape
        depth = depth if depth is not None else torch.zeros(b, 1, h, w, device=image.device)
        x = torch.cat([
            image,
            depth,
            torch.zeros_like(image),
            torch.zeros_like(image)
        ], dim=1)

        # 2. Forward pass
        preds = self.ode_block(x, eval_time=eval_time)

        # 3. Post-process
        pred = preds[-1]

        return {
            "enhanced": pred[:, :3, :, :],
            "curve_map": self.enhance_func.last_curve_map,
            "denoised": self.enhance_func.last_denoised,
            "noise_map": self.enhance_func.last_noise_map,
            "l_tv": pred[:, 4:7, :, :],
            "l_denoise": pred[:, 7:10, :, :],
        }

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---


# --- Model Variants ---

@MODELS.register(name="iz_dce", metaclass=IZ_DCE)
def iz_dce(*args, **kwargs):
    """Create a IZ-DCE model."""
    _ = kwargs.pop("name", "iz_dce")
    in_channels = kwargs.pop("in_channels", 3)
    hidden_dim = kwargs.pop("hidden_dim", 32)
    imgsz = kwargs.pop("imgsz", 256)
    chunk_size = kwargs.pop("chunk_size", 100000)
    num_iter = kwargs.pop("num_iter", 8)
    use_depth = kwargs.pop("use_depth", False)
    use_anscombe = kwargs.pop("use_anscombe", False)
    return IZ_DCE(
        name="iz_dce",
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        imgsz=imgsz,
        chunk_size=chunk_size,
        num_iter=num_iter,
        use_depth=use_depth,
        use_anscombe=use_anscombe,
        *args, **kwargs
    )


@MODELS.register(name="iz_dce_ode", metaclass=IZ_DCE_ODE)
def iz_dce_ode(*args, **kwargs):
    """Create a IZ-DCE-ODE model."""
    _ = kwargs.pop("name", "iz_dce_ode")
    in_channels = kwargs.pop("in_channels", 3)
    hidden_dim = kwargs.pop("hidden_dim", 32)
    imgsz = kwargs.pop("imgsz", 128)
    chunk_size = kwargs.pop("chunk_size", 100000)
    use_depth = kwargs.pop("use_depth", False)
    use_anscombe = kwargs.pop("use_anscombe", False)
    tol = kwargs.pop("tol", 1e-5)
    adjoint = kwargs.pop("adjoint", True)
    return IZ_DCE_ODE(
        name="iz_dce_ode",
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        imgsz=imgsz,
        chunk_size=chunk_size,
        use_depth=use_depth,
        use_anscombe=use_anscombe,
        tol=tol,
        adjoint=adjoint,
        *args, **kwargs
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
