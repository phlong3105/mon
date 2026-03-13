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
    "iz_dce",
]

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchdiffeq import odeint

from mon.core import is_weights_type, log, MODELS, Path, Task, WeightsLike
from mon.nn import ModelRegisterMixin
from .module import DecoderSIREN, Denoiser, Encoder, EnhancementCurveODE

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
        imgsz: int = 256,
        chunk_size: int = 100000,
        method: str = "dopri5",
        tol: float = 1e-5,
        ode_options: dict | None = None,
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
            method (str, optional): ODE solver method. Defaults to "dopri5".
            tol (float, optional): Tolerance for solver. Defaults to 1e-5.
            ode_options (dict, optional): Additional options to pass to the ODE
                solver. Defaults to None.
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
        self.imgsz = imgsz
        self.chunk_size = chunk_size
        self.method = method
        self.tol = tol
        self.ode_options = ode_options

        # Define network
        self.denoiser = Denoiser(
            in_channels=in_channels,
            use_anscombe=use_anscombe,
        )
        self.encoder = Encoder(
            in_channels=in_channels * 2 + 1 if use_depth else in_channels * 2,
            hidden_dim=hidden_dim,
        )
        # Implicit decoder (Siren/Continuous MLP)
        """
        self.decoder = Decoder(
            in_channels=hidden_dim + 2,
            out_channels=self.out_channels,
            hidden_dim=hidden_dim
        )
        """
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
            save_debug (bool, optional): Whether to save intermediate results for
                debugging. Defaults to False.
        """
        # 1. Prepare inputs
        x = image
        d = depth
        b, c, h, w = x.shape
        size = (self.imgsz, self.imgsz)

        # We downsample the input to 512x512 so the CNN doesn't cause an OOM error
        if (h, w) != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
            d = F.interpolate(d, size=size, mode="bilinear", align_corners=True) if d is not None else None

        # 2. Denoise
        l_denoise, noise, p_x = self.denoiser(x)

        # 3. Fusion
        if d is not None:
            x_in = torch.cat([x, p_x, d], dim=1)
        else:
            x_in = torch.cat([x, p_x], dim=1)

        # 4. Encode
        feat = self.encoder(x_in)

        # 5. Predict curve parameters
        if (h, w) == size:
            A = self.predict_curve_map(feat, self.imgsz, self.imgsz)
        else:
            A = self.predict_curve_map_chunk(feat, h, w)

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
    def predict_curve_map(self, feat: Tensor, h: int, w: int) -> Tensor:
        b = feat.shape[0]
        device = feat.device

        # We map the massive target resolution to the [-1, 1] continuous space.
        h_coords = torch.linspace(-1, 1, steps=h, device=device)
        w_coords = torch.linspace(-1, 1, steps=w, device=device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
        coords = torch.stack([grid_w, grid_h], dim=-1).view(1, -1, 2).repeat(b, 1, 1)  # [B, H*W, 2]

        sampled_feat = F.grid_sample(feat, coords.unsqueeze(1), mode="bilinear", align_corners=True)
        sampled_feat = sampled_feat.squeeze(2).permute(0, 2, 1)  # [B, N, 32]

        A = self.decoder(sampled_feat, coords)
        A = A.view(b, h, w, 3).permute(0, 3, 1, 2)

        return A

    def predict_curve_map_chunk(self, feat: Tensor, h: int, w: int) -> Tensor:
        chunk_size = self.chunk_size
        b = feat.shape[0]
        device = feat.device

        # We map the massive target resolution to the [-1, 1] continuous space.
        h_coords = torch.linspace(-1, 1, steps=h, device=device)
        w_coords = torch.linspace(-1, 1, steps=w, device=device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")

        # coords = torch.stack([grid_w, grid_h], dim=-1).view(b, -1, 2)  # [B, H*W, 2]
        # FIX: View as 1 batch, then expand/repeat to match actual batch size B
        coords = torch.stack([grid_w, grid_h], dim=-1).view(1, -1, 2).expand(b, -1, -1)

        total_points = h * w
        A_list = []

        # 1 Chunked MLP Inference
        # Process the points in batches of `chunk_size` to cap VRAM usage.
        for i in range(0, total_points, chunk_size):
            coords_chunk = coords[:, i:i+chunk_size, :]  # [B, chunk, 2]

            # Sample from the 512x512 feature map at the exact target coordinates
            sampled_feat = F.grid_sample(feat, coords_chunk.unsqueeze(1), mode="bilinear", align_corners=True)
            sampled_feat = sampled_feat.squeeze(2).permute(0, 2, 1)  # [B, chunk, 32]

            # Predict the curve parameters for this chunk
            A_chunk = self.decoder(sampled_feat, coords_chunk)
            A_list.append(A_chunk)

        # 2. Reconstruct the spatial curve parameter map
        A_flat = torch.cat(A_list, dim=1)  # [B, H*W, 24]
        A = A_flat.view(b, h, w, 3).permute(0, 3, 1, 2)  # [B, 3, H, W]

        return A

    # --- Enhance ---
    def enhance(self, image: Tensor, A: Tensor, t: Tensor | None = None) -> Tensor:
        """Apply the continuous enhancement via Neural ODE."""
        # 1. Initialize the derivative function with our predicted curve
        ode_func = EnhancementCurveODE(A)

        # 2. Define the continuous integration time span
        # t=0.0 is the dark image, t=1.0 is the fully enhanced image
        if t is None:
            t_span = torch.tensor([0.0, 1.0], device=image.device)
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

    def enhance_iter(self, image: Tensor, A: Tensor, num_iters: int = 8) -> Tensor:
        """Apply the original iterative enhancement scheme."""
        y = image
        for _ in range(num_iters):
            y = y + A * (torch.pow(y, 2) - y)
        return y

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
    method = kwargs.pop("method", "dopri5")
    tol = kwargs.pop("tol", 1e-5)
    ode_options = kwargs.pop("ode_options", None)
    use_depth = kwargs.pop("use_depth", False)
    use_anscombe = kwargs.pop("use_anscombe", False)
    return IZ_DCE(
        name="iz_dce",
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        imgsz=imgsz,
        chunk_size=chunk_size,
        method=method,
        tol=tol,
        ode_options=ode_options,
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
