#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FLOL Models.

This module provides the FLOL definition and pre-trained weights.

References:
    - Paper: "FLOL: Fast Baselines for Real-World Low-Light Enhancement," arXiv 2026.
    - Code: https://github.com/cidautai/FLOL
"""

from __future__ import annotations

__all__ = [
    "FLOL",
    "FLOL_Weights",
    "flol",
]

import functools

import kornia
import torch
from tensordict import TensorDict
from torch import nn, Tensor
from torch.nn import functional as F
from typing_extensions import override

from mon.core import (
    is_weights_type,
    K,
    log,
    MODELS,
    PATCHERS,
    Path,
    Size,
    Strategy,
    Task,
    WEIGHTS,
    Weights,
    WeightsEnum,
    WeightsLike,
)
from mon.metrics import benchmark, create_dummy_image
from mon.nn import Model, ModelRegisterMixin
from mon.ops import ImagePatcher
from .module import AmplitudeNet_skip, make_layer, ResidualBlock_noBN, SFNet

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class FLOL(ModelRegisterMixin, Model):
    """FLOL model for low-light image enhancement.

    References:
        - Paper: "FLOL: Fast Baselines for Real-World Low-Light Enhancement,"
          arXiv 2026.
        - Code: https://github.com/cidautai/FLOL
    """

    arch: str = "flol"
    name: str = "flol"
    tasks: list[Task] = [Task.LLE]
    strategies: list[Strategy] = [Strategy.RESIZE, Strategy.PATCH]
    model_dir: Path = current_dir

    in_keys: set = {"image"}
    out_keys: set = {"enhanced"}
    debug_keys: set = {"amplitude"}

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        channels: int = 16,
        weights: Weights | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            channels (int, optional): Number of channels. Defaults to 16.
            weights (Weights, optional): Pre-trained weights to load.
                Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose
        self.channels = channels

        # Define network
        self.AmpNet = nn.Sequential(
            AmplitudeNet_skip(8),
            nn.Sigmoid(),
        )

        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, channels=channels)

        self.conv_first_1 = nn.Conv2d(3 * 2, channels, 3, 1, 1, bias=True)
        self.conv_first_2 = nn.Conv2d(channels, channels, 3, 2, 1, bias=True)
        self.conv_first_3 = nn.Conv2d(channels, channels, 3, 2, 1, bias=True)

        self.feature_extraction = make_layer(ResidualBlock_noBN_f, 1)
        self.recon_trunk = make_layer(ResidualBlock_noBN_f, 1)

        self.upconv1 = nn.Conv2d(channels * 2, channels * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(channels * 2, channels * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(channels * 2, channels, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(channels, 3, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.transformer = SFNet(channels, n=4)
        self.recon_trunk_light = make_layer(ResidualBlock_noBN_f, 6)

        # Load weights
        if weights is not None and is_weights_type(weights):
            state_dict = weights.state_dict()
            if "params" in state_dict:
                state_dict = state_dict["params"]
                torch.save(state_dict, str(weights.path))
            self.load_state_dict(state_dict)
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

    # --- Callable & Context Manager ---
    @override
    def forward(
        self,
        image: Tensor,
        use_patch: bool = False,
        *args, **kwargs
    ) -> tuple[Tensor, ...]:
        """Route the inputs through the model's different forward methods based
        on the context.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            use_patch (bool, optional): Whether to use patch-based strategy.
                Defaults to False.

        Returns:
            tuple[Tensor, ...]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - amplitude (Tensor): Intermediate image tensor after amplitude
                  enhancement, of shape (B, C, H, W) and values ranging from
                  0.0 to 1.0.
        """
        if use_patch:
            return self.forward_patch(image=image, *args, **kwargs)
        else:
            return self.forward_step(image=image, *args, **kwargs)

    @override
    def forward_step(self, image: Tensor, *args, **kwargs ) -> tuple[Tensor, ...]:
        """Perform a single forward step of the model.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            tuple[Tensor, ...]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - amplitude (Tensor): Intermediate image tensor after amplitude
                  enhancement, of shape (B, C, H, W) and values ranging from
                  0.0 to 1.0.
        """
        x = image

        # 1. Amplitude Estimation
        # 1.1. Frequency Stage
        _, _, h, w = x.shape
        image_fft = torch.fft.fft2(x, norm="backward")
        mag_image = torch.abs(image_fft)
        pha_image = torch.angle(image_fft)

        curve_amps = self.AmpNet(x)

        mag_image = mag_image / (curve_amps + 0.00000001)
        real_image_enhanced = mag_image * torch.cos(pha_image)
        imag_image_enhanced = mag_image * torch.sin(pha_image)
        img_amp_enhanced = torch.fft.ifft2(
            torch.complex(real_image_enhanced, imag_image_enhanced),
            s=(h, w),
            norm="backward",
        ).real

        x_center = img_amp_enhanced

        rate = 2 ** 3
        pad_h = (rate - h % rate) % rate
        pad_w = (rate - w % rate) % rate
        if pad_h != 0 or pad_w != 0:
            x_center = F.pad(x_center, (0, pad_w, 0, pad_h), "reflect")
            x = F.pad(x, (0, pad_w, 0, pad_h), "reflect")

        # 1.2. Spatial Stage
        L1_fea_1 = self.lrelu(self.conv_first_1(torch.cat((x_center, x), dim=1)))
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))  # Encoder
        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))

        fea = self.feature_extraction(L1_fea_3)
        fea_light = self.recon_trunk_light(fea)

        h_feature = fea.shape[2]
        w_feature = fea.shape[3]
        mask_image = self.get_mask(x_center)  # SNR Map
        mask = F.interpolate(mask_image, size=[h_feature, w_feature], mode="nearest")  # Resize and Normalize SNR map

        fea_unfold = self.transformer(fea)

        channel = fea.shape[1]
        mask = mask.repeat(1, channel, 1, 1)
        fea = fea_unfold * (1 - mask) + fea_light * mask  # SNR-based Interaction

        out_noise = self.recon_trunk(fea)
        out_noise = torch.cat([out_noise, L1_fea_3], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_2], dim=1)  # Decoder
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_1], dim=1)
        out_noise = self.lrelu(self.HRconv(out_noise))
        out_noise = self.conv_last(out_noise)
        out_noise = out_noise + x
        out_noise = out_noise[:, :, :h, :w]

        # 2. Return final and intermediate results for debugging
        return out_noise, x_center  #, mag_image, x_center, mask_image

    def forward_patch(
        self,
        image: Tensor,
        patcher: dict | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, ...]:
        """Forward the input through the network using the patch-based strategy.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            patcher (dict, optional): A dictionary containing the patching
                configuration, such as patch size and stride. Defaults to None
                means using the default patcher.

        Returns:
            tuple[Tensor, ...]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - gray (Tensor): Grayscale image tensor of shape (B, 1, H, W)
                  and values ranging from 0.0 to 1.0.
                - color_hist (Tensor): Color histogram tensor of shape
                  (B, d_hist * 3) where d_hist is the number of histogram bins
                  per channel.
        """
        # 1. Initialize image patcher
        patcher: dict = patcher or {"name": "uniform"}
        patcher: ImagePatcher = PATCHERS.build(image=image, **patcher)

        # 2. Iterate and Process
        for patch, x, y in patcher:
            # 2.1. Process the patch
            outputs = self.forward_step(image=patch, *args, **kwargs)
            patch_outputs = {
                "enhanced": outputs[0],
                "amplitude": outputs[1],
            }
            # 2.2. Feed result back to Patcher
            patcher(patches=patch_outputs, x=x, y=y)

        # 3. Get the merged results
        return tuple(patcher.output.values())

    def get_mask(self, dark: Tensor) -> Tensor:   # SNR map
        light = kornia.filters.gaussian_blur2d(dark, (5, 5), (1.5, 1.5))
        dark = (
            dark[:, 0:1, :, :] * 0.299
            + dark[:, 1:2, :, :] * 0.587
            + dark[:, 2:3, :, :] * 0.114
        )
        light = (
            light[:, 0:1, :, :] * 0.299
            + light[:, 1:2, :, :] * 0.587
            + light[:, 2:3, :, :] * 0.114
        )
        noise = torch.abs(dark - light)

        mask = torch.div(light, noise + 0.0001)
        b, c, h, w = mask.shape
        mask_max = torch.max(mask.view(b, -1), dim=1)[0]
        mask_max = mask_max.view(b, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, h, w)
        mask = mask * 1.0 / (mask_max + 0.0001)
        mask = torch.clamp(mask, min=0, max=1.0)
        return mask.float()

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
        return benchmark(model=self, inputs=inputs, *args, **kwargs)

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="flol")
class FLOL_Weights(WeightsEnum):

    LOL_V2_REAL = Weights(
        path=K.ZOO_ROOT / "enhance/lle/flol/flol/pretrained/flol_lol_v2_real.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={}
    )
    LOL_V2_REAL_UHD_LL = Weights(
        path=K.ZOO_ROOT / "enhance/lle/flol/flol/pretrained/flol_lol_v2_real_uhd_ll.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={}
    )
    DEFAULT = LOL_V2_REAL


# --- Model Variants ---

@MODELS.register(name="flol", metaclass=FLOL)
def flol(weights: WeightsLike = "default", *args, **kwargs):
    """Create a DCC-Net model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "flol")
    channels = kwargs.pop("channels", 16)
    return FLOL(
        name="flol",
        channels=channels,
        weights=FLOL_Weights(weights),
        *args, **kwargs,
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
