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

from typing import override

import torch
from tensordict import TensorDict
from torch import Tensor
from torch.nn import functional as F
from torchdiffeq import odeint

from mon.core import (
    is_weights_type,
    log,
    MODELS,
    Path,
    Size,
    Strategy,
    Task,
    Weights,
)
from mon.metrics import benchmark, create_dummy_image
from mon.nn import Model, ModelRegisterMixin
from .module import DecoderSIREN, Denoiser, Encoder, EnhancementCurveODE
from .utils import get_coords

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class SLICE(ModelRegisterMixin, Model):
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
    tasks: list[Task] = [Task.LLE]
    strategies: list[Strategy] = [Strategy.NATIVE]
    model_dir: Path = current_dir

    in_keys: set = {"image"}
    out_keys: set = {"enhanced"}
    debug_keys: set = {"curve_map", "noise_map", "denoised"}

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
        imgsz: Size = 256,
        method: str = "dopri5",
        tol: float = 1e-5,
        ode_options: dict | None = None,
        noise_level: float | None = None,
        use_depth: bool = False,
        use_anscombe: bool = False,
        weights: Weights | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model to use.
            in_channels (int, optional): Number of input channels. Defaults to 3.
            hidden_dim (int, optional): Hidden dimension. Defaults to 32.
            imgsz (Size, optional): Downsample the input image to this size
                for encoding. Defaults to 256.
            method (str, optional): ODE solver method. Defaults to "dopri5".
            tol (float, optional): Tolerance for solver. Defaults to 1e-5.
            ode_options (dict | None, optional): Additional options to pass to
                the ODE solver. Defaults to None.
            noise_level (float | None, optional): The noise level to add to the
                input. If None, no noise is added. Defaults to None.
            use_depth (bool, optional): Whether to use depth as an additional
                input channel. Defaults to False.
            use_anscombe (bool, optional): Whether to apply the Anscombe
                transform to the input before denoising. Defaults to False.
            weights (Weights | None, optional): Pre-trained weights to load.
                Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Validate inputs
        if method not in self.methods:
            raise ValueError(
                f"unsupported ODE solver {method}, must be one of {self.methods}."
            )

        # Assign attributes
        self.verbose = verbose
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.imgsz = Size.from_value(imgsz)
        self.method = method
        self.tol = tol
        self.ode_options = ode_options
        self.use_depth = use_depth

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
            mapping_size=imgsz,
        )

        # Load weights
        if weights is not None and is_weights_type(weights):
            self.load_state_dict(weights.state_dict())
            if self.verbose:
                log(f"initialized {name} from weights {weights.path.as_posix()}.")
        else:
            if self.verbose:
                log(f"Initialized {name} from scratch.")

    # --- Callable & Context Manager ---
    @override
    def forward(
        self,
        image: Tensor,
        depth: Tensor | None = None,
        T: Tensor | None = None,
        chunk_size: int = 65536,
        *args, **kwargs
    ) -> tuple[Tensor, ...]:
        """Route the inputs through the model's different forward methods based
        on the context.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            depth (Tensor | None, optional): Input depth map tensor of shape
                (B, 1, H, W) and values ranging from 0.0 to 1.0. If None, depth
                is not used. Defaults to None.
            T (Tensor | None, optional): Time span for the ODE solver.
                Defaults to None.
            chunk_size (int, optional): Chunk size for inference on large images.
                Defaults to 65,536.

        Returns:
            tuple[Tensor, ...]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - curve_map (Tensor): The estimated curve parameters of shape
                  (B, C, H, W) and values ranging from -1.0 to 1.0.
                - noise_map (Tensor): The estimated noise map of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - denoised (Tensor): The denoised image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
        """
        return self.forward_step(image=image, depth=depth, T=T, chunk_size=chunk_size)

    def forward_train(
        self,
        image: Tensor,
        depth: Tensor | None = None,
        T: Tensor | None = None,
        *args, **kwargs
    ) -> TensorDict:
        """Perform a single forward step of the model during training.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            depth (Tensor | None, optional): Input depth map tensor of shape
                (B, 1, H, W) and values ranging from 0.0 to 1.0. If None, depth
                is not used. Defaults to None.
            T (Tensor | None, optional): Time span for the ODE solver.
                Defaults to None.

        Returns:
            tuple[Tensor, ...]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - r (Tensor): The estimated curve parameters of shape (B, C*8, H, W)
                  and values ranging from -1.0 to 1.0.
        """
        # 1. Prepare inputs
        x = image
        d = depth if self.use_depth else None
        size1 = self.imgsz

        # 2. Denoise
        l_denoise, noise, p_x = self.denoiser(x)

        # 3. Fusion
        if d is not None:
            x_in = torch.cat([x, p_x, d], dim=1)
        else:
            x_in = torch.cat([x, p_x], dim=1)

        # 4. Encode global features
        features = self.encoder(x_in)

        # 5. Predict curve parameters
        A = self.gen_curve_map(features, size1)

        # 6. Enhance
        if "iter" in self.method:
            num_iters = int(self.method.split("iter")[-1])
            y = self.enhance_iter(image=image, A=A, num_iters=num_iters)
        else:
            y = self.enhance_ode(image=image, A=A, T=T)

        # 7. Return final and intermediate results for debugging
        outputs = {
            "enhanced": y,
            "curve_map": A,
            "noise_map": noise,
            "denoised": p_x,
            "l_denoise": l_denoise,
        }
        return TensorDict(outputs, batch_size=[])

    @override
    def forward_step(
        self,
        image: Tensor,
        depth: Tensor | None = None,
        T: Tensor | None = None,
        chunk_size: int = 65536,
        *args, **kwargs
    ) -> tuple[Tensor, ...]:
        """Perform a single forward step of the model.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            depth (Tensor | None, optional): Input depth map tensor of shape
                (B, 1, H, W) and values ranging from 0.0 to 1.0. If None, depth
                is not used. Defaults to None.
            T (Tensor | None, optional): Time span for the ODE solver.
                Defaults to None.
            chunk_size (int, optional): Chunk size for inference on large images.
                Defaults to 65,536.

        Returns:
            tuple[Tensor, ...]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - curve_map (Tensor): The estimated curve parameters of shape
                  (B, C, H, W) and values ranging from -1.0 to 1.0.
                - noise_map (Tensor): The estimated noise map of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - denoised (Tensor): The denoised image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
        """
        # 1. Prepare inputs
        x = image
        d = depth if self.use_depth else None
        size0 = Size.from_value(x)
        size1 = self.imgsz

        # Downsample the inputs to self.imgsz so the CNN doesn't cause an OOM
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

        # 4. Encode global features
        features = self.encoder(x_in)

        # 5. Predict curve parameters
        if size0 == size1:
            A = self.gen_curve_map(features, size1)
        else:
            A = self.gen_curve_map_chunk(features, size0, chunk_size=chunk_size)

        # 6. Enhance
        if "iter" in self.method:
            num_iters = int(self.method.split("iter")[-1])
            y = self.enhance_iter(image=image, A=A, num_iters=num_iters)
        else:
            y = self.enhance_ode(image=image, A=A, T=T)

        # 7. Return final and intermediate results for debugging
        return y, A, noise, p_x

    # --- Curve Map ---
    def gen_curve_map(self, features: Tensor, size: Size) -> Tensor:
        """Predict the curve parameters of the full image (for training)."""
        b = features.shape[0]

        coords = get_coords(features, size)
        sampled_feat = F.grid_sample(features, coords.unsqueeze(1), mode="bilinear", align_corners=True)
        sampled_feat = sampled_feat.squeeze(2).permute(0, 2, 1)  # [B, N, 32]

        A = self.decoder(sampled_feat, coords)
        A = A.view(b, size.h, size.w, 3).permute(0, 3, 1, 2)

        return A

    @torch.inference_mode()
    def gen_curve_map_chunk(self, features: Tensor, size: Size, chunk_size: int) -> Tensor:
        """Predict the curve parameters of the full image in chunks
        (for inference on large images).
        """
        b = features.shape[0]
        device = features.device
        total_points = size.area

        # 1. Pre-allocate the output tensor and get the coordinates for the full image
        coords = get_coords(features, size)
        A_flat = torch.empty((b, total_points, 3), dtype=features.dtype, device=device)

        # 2. Chunked MLP Inference
        # Process the points in batches of `chunk_size` to cap VRAM usage.
        for i in range(0, total_points, chunk_size):
            coords_chunk = coords[:, i:i+chunk_size, :]  # [B, chunk, 2]

            # Sample from the 512x512 feature map at the exact target coordinates
            sampled_feat = F.grid_sample(features, coords_chunk.unsqueeze(1), mode="bilinear", align_corners=True)
            sampled_feat = sampled_feat.squeeze(2).permute(0, 2, 1)  # [B, chunk, 32]

            # Predict the curve parameters for this chunk
            A_chunk = self.decoder(sampled_feat, coords_chunk)

            # 3. Inject directly into the pre-allocated tensor (No lists!)
            A_flat[:, i:i+chunk_size, :] = A_chunk

        # 4. Reconstruct the spatial curve parameter map
        A = A_flat.view(b, size.h, size.w, 3).permute(0, 3, 1, 2)  # [B, 3, H, W]

        return A

    # --- Enhance ---
    def enhance_ode(self, image: Tensor, A: Tensor, T: Tensor | None = None) -> Tensor:
        """Apply the continuous enhancement via Neural ODE."""
        # 1. Initialize the derivative function with our predicted curve
        ode_func = EnhancementCurveODE(A)

        # 2. Define the continuous integration time span
        # t=0.0 is the dark image, t=1.0 is the fully enhanced image
        if T is None:
            T_span = torch.tensor([0.0, 1.0], device=image.device)
        else:
            T_span = T

        # 3. Solve the ODE
        trajectory = odeint(
            func=ode_func,
            y0=image,
            t=T_span,
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
        A_orig = self.gen_curve_map(feat_orig, self.imgsz)

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
        A_pred_flip = self.gen_curve_map(feat_flipped, self.imgsz)

        # 3. The Equivariance Penalty
        # Manually flip the original prediction to see if they match
        A_orig_manually_flipped = torch.flip(A_orig, dims=[3])

        # Calculate the L1 loss between the two
        loss_equi = F.l1_loss(A_pred_flip, A_orig_manually_flipped)

        return loss_equi

    # --- Benchmark ---
    @override
    def benchmark(self, imgsz: Size, *args, **kwargs) -> dict[str, float]:
        """Perform a single forward step of the model to benchmark its performance.

        Args:
            imgsz (Size): Input image size.
            **kwargs: Additional arguments for benchmarking, such as number
                of runs, device, etc.

        Returns:
            dict[str, float]: A dictionary containing the benchmark results,
                such as latency, FLOPs, and parameter count.
        """
        device = next(self.parameters()).device

        # Create dummy inputs
        dummy_input = create_dummy_image(imgsz=imgsz, device=device)
        data = TensorDict({"image": dummy_input}, batch_size=dummy_input.shape)
        inputs = {"data": data}

        # Benchmark the model
        return benchmark(model=self, inputs=inputs, copy=False, *args, **kwargs)

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
