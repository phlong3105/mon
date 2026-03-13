#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SLICE Models.

This module provides the SLICE definition and pre-trained weights.

References:
    - Paper: "SLICE: Scale-Arbitrary Low-Light Enhancement via Depth-Aware
      Implicit Curve Estimation"
    - Code: https://github.com/phlong3105/slice
"""

from __future__ import annotations

__all__ = [
    "SLICE",
    "slice",
]

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchdiffeq import odeint

from mon.core import (
    is_weights_type,
    log,
    MODELS,
    Path,
    Size,
    SizeLike,
    Task,
    WeightsLike,
)
from mon.nn import ModelRegisterMixin
from .module import DecoderSIREN, Denoiser, Encoder, EnhancementCurveODE
from .utils import get_coords

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class SLICE(ModelRegisterMixin, nn.Module):
    r"""SLICE model.

    "What exactly is SLICE?":

    - Self-Supervised Frontend: Takes the noisy raw tensor and extracts a
      mathematically pristine, structurally sound image (bypassing the Anscombe
      clipping trap and structure leakage).
    - Contextual Encoding: Downsamples the clean image and its corresponding
      depth map to generate a rich, globally aware feature latent space.
    - Scale-Arbitrary Decoding: Uses a Fourier-encoded SIREN MLP to map
      continuous high-resolution spatial coordinates $(x, y)$ against the global
      features, instantly generating the spatial curve parameters $\mathcal{A}$
      in memory-safe chunks.
    - Iterative/Continuous Enhancement: Applies the predicted curve to the
      pristine image using either discrete iterations or a continuous ODE solver,
      yielding a 4K/8K image with perfect contrast and zero noise amplification.

    References:
        - Paper: "SLICE: Scale-Arbitrary Low-Light Enhancement via Depth-Aware
          Implicit Curve Estimation"
        - Code: https://github.com/phlong3105/slice
    """

    arch: str = "slice"
    name: str = "slice"
    tasks: list[Task] = [Task.ENHANCE]
    model_dir: Path = current_dir
    methods = [
        "iter8", "iter5", "iter4", "dopri8", "dopri5", "bosh3", "fehlberg2",
        "adaptive_heun", "euler", "midpoint", "heun2", "heun3", "rk4",
        "explicit_adams", "implicit_adams", "fixed_adams", "scipy_solver"
    ]

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        in_channels: int = 3,
        hidden_dim: int = 32,
        imgsz: SizeLike = 256,
        method: str = "dopri5",
        tol: float = 1e-5,
        ode_options: dict | None = None,
        noise_level: float | None = None,
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
            imgsz (SizeLike, optional): Downsample the input image to this size
                for encoding. Defaults to 256.
            method (str, optional): ODE solver method. Defaults to "dopri5".
            tol (float, optional): Tolerance for solver. Defaults to 1e-5.
            ode_options (dict, optional): Additional options to pass to the ODE
                solver. Defaults to None.
            noise_level (float | None, optional): The noise level to add to the
                input. If None, no noise is added. Defaults to None.
            use_depth (bool, optional): Whether to use depth as an additional
                input channel. Defaults to False.
            use_anscombe (bool, optional): Whether to apply the Anscombe
                transform to the input before denoising. Defaults to False.
            weights (WeightsLike, optional): Pre-trained weights to load.
                Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)
        # Validate inputs
        if method not in self.methods:
            raise ValueError(
                f"Invalid ODE solver method: '{method}'. "
                f"Must be one of {self.methods}."
            )

        # Assign attributes
        self.verbose = verbose
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.imgsz = Size.from_value(imgsz)
        self.method = method
        self.tol = tol
        self.ode_options = ode_options

        # Define network
        self.denoiser = Denoiser(
            in_channels=in_channels,
            use_anscombe=use_anscombe,
            noise_level=noise_level,
        )
        self.encoder = Encoder(
            in_channels=in_channels * 2 + 1 if use_depth else in_channels * 2,
            hidden_dim=hidden_dim,
        )
        # Implicit decoder (Siren/Continuous MLP)
        self.decoder = DecoderSIREN(
            in_channels=hidden_dim,
            out_channels=self.out_channels,
            hidden_dim=hidden_dim,
            pos_encode=True,
            mapping_size=imgsz
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
    def forward(
        self,
        image: Tensor,
        depth: Tensor | None = None,
        t: Tensor | None = None,
        chunk_size: int = 100000,
        save_debug: bool = False,
        *args, **kwargs
    ) -> dict:
        """Forward the input through the network.

        Args:
            image (Tensor): Image tensor of shape (B, 3, H, W) and values
                ranging from 0.0 to 1.0.
            depth (Tensor, optional): Depth tensor of shape (B, 1, H, W) and
                values ranging from 0.0 to 1.0. Defaults to None.
            t (Tensor, optional): Time tensor. If None, use the original
                Zero-DCE iteration scheme. Defaults to None.
            chunk_size (int): Number of pixels to process at once at inference.
                Defaults to 100,000.
            save_debug (bool, optional): Whether to save intermediate results for
                debugging. Defaults to False.
        """
        # 1. Prepare inputs
        x = image
        d = depth
        size0 = Size.from_value(x)
        size1 = self.imgsz
        chunk_size = chunk_size or self.chunk_size

        # We downsample the input to 512x512 so the CNN doesn't cause an OOM error
        if size0 != size1:
            x = F.interpolate(x, size=size1.hw, mode="bilinear", align_corners=True)
            d = F.interpolate(d, size=size1.hw, mode="bilinear", align_corners=True) if d is not None else None

        # 2. Denoise
        l_denoise, noise, p_x = self.denoiser(x)

        # 3. Fusion
        if d is not None:
            x_in = torch.cat([x, p_x, d], dim=1)
        else:
            x_in = torch.cat([x, p_x], dim=1)

        # 4. Encode
        features = self.encoder(x_in)

        # 5. Predict curve parameters
        if size0 == size1:
            A = self.predict_curve_map(features, size1)
        else:
            A = self.predict_curve_map_chunk(features, size0, chunk_size=chunk_size)

        # 6. Enhance
        if "iter" in self.method:
            num_iters = int(self.method.split("iter")[-1])
            y = self.enhance_iter(image, A, num_iters)
        else:
            y = self.enhance(image, A, t=t)

        # 7. Return final and intermediate results for debugging
        outputs = { "enhanced": y }
        if self.training or save_debug:
            outputs |= {
                "curve_map": A,
                "noise_map": noise,
                "denoised": p_x,
                "l_denoise": l_denoise,
            }
        return outputs

    # --- Curve Map ---
    def predict_curve_map(self, features: Tensor, size: Size) -> Tensor:
        b = features.shape[0]

        coords = get_coords(features, size)
        sampled_feat = F.grid_sample(features, coords.unsqueeze(1), mode="bilinear", align_corners=True)
        sampled_feat = sampled_feat.squeeze(2).permute(0, 2, 1)  # [B, N, 32]

        A = self.decoder(sampled_feat, coords)
        A = A.view(b, size.h, size.w, 3).permute(0, 3, 1, 2)

        return A

    def predict_curve_map_chunk(self, features: Tensor, size: Size, chunk_size: int) -> Tensor:
        b = features.shape[0]

        coords = get_coords(features, size)
        total_points = size.area
        A_list = []

        # 1. Chunked MLP Inference
        # Process the points in batches of `chunk_size` to cap VRAM usage.
        for i in range(0, total_points, chunk_size):
            coords_chunk = coords[:, i:i+chunk_size, :]  # [B, chunk, 2]

            # Sample from the 512x512 feature map at the exact target coordinates
            sampled_feat = F.grid_sample(features, coords_chunk.unsqueeze(1), mode="bilinear", align_corners=True)
            sampled_feat = sampled_feat.squeeze(2).permute(0, 2, 1)  # [B, chunk, 32]

            # Predict the curve parameters for this chunk
            A_chunk = self.decoder(sampled_feat, coords_chunk)
            A_list.append(A_chunk)

        # 2. Reconstruct the spatial curve parameter map
        A_flat = torch.cat(A_list, dim=1)  # [B, H*W, 24]
        A = A_flat.view(b, size.h, size.w, 3).permute(0, 3, 1, 2)  # [B, 3, H, W]

        return A

    # --- Enhance ---
    def enhance(self, image: Tensor, A: Tensor, t: Tensor | None = None) -> Tensor:
        """Apply the continuous enhancement via Neural ODE."""
        # 1. Initialize the derivative function with our predicted curve
        ode_func = EnhancementCurveODE(A)

        # 2. Define the continuous integration time span
        # t=0.0 is the dark image, t=1.0 is the fully enhanced image
        if t is None:
            t_span = torch.tensor([0.0, 3.0], device=image.device)
        else:
            t_span = t

        # 3. Solve the ODE
        trajectory = odeint(
            func=ode_func,
            y0=image,
            t=t_span,
            rtol=self.tol,
            atol=self.tol,
            method=self.method,
            options=self.ode_options,
        )
        # trajectory contains the image at t=0.0 and t=1.0. We want the final state.
        y = trajectory[-1]

        return y

    # noinspection PyMethodMayBeStatic
    def enhance_iter(self, image: Tensor, A: Tensor, num_iters: int = 8) -> Tensor:
        """Apply the original iterative enhancement scheme."""
        y = image
        for _ in range(num_iters):
            y = y + A * (torch.pow(y, 2) - y)
        return y

    # --- Extension for IEEE TIP ---
    def loss_equi_A(self, image: Tensor, depth: Tensor | None = None) -> Tensor:
        """Calculate the equivariance loss on the curve map A (inspired by P2N paper).
        """
        # 1. The Original Pass
        x = image
        d = depth
        _, _, p_x = self.denoiser(x)
        if d is not None:
            x_in_orig = torch.cat([x, p_x, d], dim=1)
        else:
            x_in_orig = torch.cat([x, p_x], dim=1)
        feat_orig = self.encoder(x_in_orig)

        # Predict standard curve map (assuming training size, e.g., 256x256)
        A_orig = self.predict_curve_map(feat_orig, self.imgsz)

        # 2. The Flipped Pass
        # Apply a horizontal flip (dim 3 is the width dimension in B, C, H, W)
        x_flipped = torch.flip(x, dims=[3])
        d_flipped = torch.flip(d, dims=[3]) if d is not None else None
        _, _, p_x_flipped = self.denoiser(x_flipped)
        if d_flipped is not None:
            x_in_flipped = torch.cat([x_flipped, p_x_flipped, d_flipped], dim=1)
        else:
            x_in_flipped = torch.cat([x_flipped, p_x_flipped], dim=1)
        feat_flipped = self.encoder(x_in_flipped)

        # Predict curve map from the flipped inputs
        A_pred_flip = self.predict_curve_map(feat_flipped, self.imgsz)

        # 3. The Equivariance Penalty
        # Manually flip the original prediction to see if they match
        A_orig_manually_flipped = torch.flip(A_orig, dims=[3])

        # Calculate the L1 loss between the two
        loss_equi = F.l1_loss(A_pred_flip, A_orig_manually_flipped)

        return loss_equi

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---


# --- Model Variants ---

@MODELS.register(name="slice", metaclass=SLICE)
def slice(*args, **kwargs):
    """Create a SLICE model."""
    _ = kwargs.pop("name", "slice")
    in_channels = kwargs.pop("in_channels", 3)
    hidden_dim = kwargs.pop("hidden_dim", 32)
    imgsz = kwargs.pop("imgsz", 256)
    method = kwargs.pop("method", "dopri5")
    tol = kwargs.pop("tol", 1e-5)
    ode_options = kwargs.pop("ode_options", None)
    noise_level = kwargs.pop("noise_level", None)
    use_depth = kwargs.pop("use_depth", False)
    use_anscombe = kwargs.pop("use_anscombe", False)
    return SLICE(
        name="slice",
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        imgsz=imgsz,
        method=method,
        tol=tol,
        ode_options=ode_options,
        noise_level=noise_level,
        use_depth=use_depth,
        use_anscombe=use_anscombe,
        *args, **kwargs
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
