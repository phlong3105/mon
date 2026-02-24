#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SALEO Models.

This module provides the SALEO definition and pre-trained weights.

References:
    - Paper: "Scale-Arbitrary Low-Light Enhancement via Depth-Aware Implicit
      Neural Optimization"
    - Code: https://github.com/phlong3105/saleo
"""

from __future__ import annotations

__all__ = [
    "Saleo",
    "saleo_ffsiren",
    "saleo_siren",
]

import sys

import kornia
import torch
from torch import nn, Tensor

from mon.core import log, MODELS, Path, Task
from mon.nn import loss as L, ModelRegisterMixin

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]
if str(current_dir) not in sys.path:
    # Add the project root to sys.path so 'import saleo' works
    # even if you run this script from inside the folder
    sys.path.append(str(current_dir))

try:
    # Works when running as a module: python -m zero_dce.predict
    from .loss import ConfidenceGatedDepthLoss, ExplicitSmoothnessLoss
    from .module import ResidualINR
    from .utils import (
        get_local_features,
        get_nearest_features,
        get_v_component,
        hsv_to_rgb,
        JitteredGridSampler,
        RandomPixelSampler,
        replace_v_component,
        rgb_to_hsv,
    )
except ImportError:
    # Works when running as a script: python predict.py
    from loss import ConfidenceGatedDepthLoss, ExplicitSmoothnessLoss
    from module import ResidualINR
    from utils import (
        get_local_features,
        get_nearest_features,
        get_v_component,
        hsv_to_rgb,
        JitteredGridSampler,
        RandomPixelSampler,
        replace_v_component,
        rgb_to_hsv,
    )

# ==============================================================================
# region BASE CLASSES
# ==============================================================================

# noinspection PyMethodMayBeStatic
class Saleo(ModelRegisterMixin, nn.Module):
    """SALEO model.

    References:
        - Paper: "Scale-Arbitrary Low-Light Enhancement via Depth-Aware Implicit
          Neural Optimization"
        - Code: https://github.com/phlong3105/saleo
    """

    arch: str = "saleo"
    name: str = "saleo"
    tasks: list[Task] = [Task.ENHANCE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        window_size: int,
        hidden_dim: int,
        num_layers: int,
        add_layers: int,
        device: torch.device = torch.device("cpu"),
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model to use.
            window_size (int): Size of the patch window.
            hidden_dim (int): Hidden dimension of the networks.
            num_layers (int): Number of layers in the networks.
            add_layers (int): Number of layers to add between the two branches.
            device (torch.device, optional): Device to use for computation.
                Defaults to "cpu"."
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        # Satisfy PyTorch's empty signature first.
        super().__init__(name=name)
        # Initialize RegistrableMixin
        # ModelRegisterMixin.__init__(self, name=name)

        # Assign attributes
        self.verbose = verbose
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.add_layers = add_layers
        self.inr_args = args
        self.inr_kwargs = kwargs
        self.device = device

    # --- Callable & Context Manager ---
    def forward(
        self,
        image: Tensor,
        depth: Tensor = None,
        epochs: int = 100,
        batch_size: int = 8,
        E: float = 0.1,
        save_debug: bool = False,
    ) -> dict:
        """Forward the input through the network.

        For each input sample, a corresponding INR network is created and
        optimized to fit the illumination map. The final enhanced image is then
        reconstructed using the learned illumination map.

        Args:
            image (Tensor): Image tensor of shape (B, 3, H, W) and values
                ranging from 0.0 to 1.0.
            depth (Tensor, optional): Depth map tensor of shape (B, 1, H, W)
                and values ranging from 0.0 to 1.0. Defaults to None.
            epochs (int, optional): Number of optimization steps. Defaults to 100.
            batch_size (int, optional): Batch size. Defaults to 8.
            E (float, optional): Well-exposedness level E. Defaults to 0.1.
            save_debug (bool, optional): If True, return intermediate results
                for debugging. Defaults to False.
        """
        # 1. Move inputs
        image = image.to(self.device)

        if depth is not None:
            depth = depth.to(self.device)

        # 2. Convert color space
        image_hsv = rgb_to_hsv(image).to(self.device)
        image_i = get_v_component(image_hsv).to(self.device)

        # 3. Optimize and infer illumination residual
        image_i_res = self._estimate_residual(
            image_i=image_i,
            depth=depth,
            epochs=epochs,
            batch_size=batch_size,
            E=E,
        )

        # 4. Retinex reconstruction
        image_i_fixed = image_i + image_i_res       # Illumination
        image_r = image_i / (image_i_fixed + 1e-4)  # Reflectance
        image_r = self._refine_reflectance(
            image_r,
            image_i_fixed,
            sigma_color=0.1,
            sigma_space=3.0
        )

        # 5. Recombine
        image_hsv_fixed = replace_v_component(image_hsv, image_r)
        image_rgb_fixed = hsv_to_rgb(image_hsv_fixed)
        # image_rgb_fixed = image_rgb_fixed / torch.max(image_rgb_fixed)

        # 6. Return final and intermediate results for debugging
        outputs = { "enhanced": image_rgb_fixed }
        if save_debug:
            outputs |= {
                "image_i": image_i,
                "image_i_res": image_i_res,
                "image_i_fixed": image_i_fixed,
                "image_r": image_r,
            }
        return outputs

    def _estimate_residual(
        self,
        image_i: Tensor,
        depth: Tensor = None,
        epochs: int = 100,
        batch_size: int = 8,
        E: float = 0.1,
        tile_size: int = 256
    ) -> Tensor:
        """Estimate the full-resolution illumination residual map using an INR
        network.

        Args:
            image_i (Tensor): Illumination map tensor of shape (B, 1, H, W) and
                values ranging from 0.0 to 1.0.
            depth (Tensor, optional): Depth map tensor of shape (B, 1, H, W)
                and values ranging from 0.0 to 1.0. Defaults to None.
            epochs (int, optional): Number of optimization steps. Defaults to 100.
            batch_size (int, optional): Batch size. Defaults to 8.
            E (float, optional): Well-exposedness level E. Defaults to 0.1.
            tile_size (int, optional): Tile size for inference. Defaults to 256.

        Returns:
            Tensor: The full resolution illumination residual map tensor of
                shape (B, 1, H, W) and values ranging from 0.0 to 1.0.
        """
        # 1. Create the INR network
        if depth is not None:
            patch_dim = self.window_size ** 2 * 2
        else:
            patch_dim = self.window_size ** 2

        model = ResidualINR(
            patch_dim = patch_dim,
            hidden_dim = self.hidden_dim,
            num_layers = self.num_layers,
            add_layers = self.add_layers,
            *self.inr_args, **self.inr_kwargs
        ).to(self.device)

        # 2. Optimize the INR network
        self._optimize_residual(
            model=model,
            image_i=image_i,
            depth=depth,
            epochs=epochs,
            batch_size=batch_size,
            E=E,
        )
        """
        self._optimize_residual_stochastic(
            model      = model,
            image_i    = image_i,
            depth      = depth,
            epochs     = epochs,
            batch_size = 500000,
            E          = E,
        )
        """

        # 3. Infer the full-resolution illumination residual map
        image_i_res = self._infer_residual(
            model=model,
            image_i=image_i,
            depth=depth,
            tile_size=tile_size,
        )

        return image_i_res

    def _optimize_residual(
        self,
        model: nn.Module,
        image_i: Tensor,
        depth: Tensor = None,
        epochs: int = 100,
        batch_size: int = 8,
        E: float = 0.1,
    ):
        """Optimize the INR network on one input sample.

        Args:
            model (nn.Module): INR network to optimize.
            image_i (Tensor): Illumination map tensor of shape (B, 1, H, W) and
                values ranging from 0.0 to 1.0.
            depth (Tensor, optional): Depth map tensor of shape (B, 1, H, W)
                and values ranging from 0.0 to 1.0. Defaults to None.
            epochs (int, optional): Number of optimization steps. Defaults to 100.
            batch_size (int, optional): Batch size. Defaults to 8.
            E (float, optional): Well-exposedness level E. Defaults to 0.1.
        """
        # 1. Define optimizer & losses
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4, weight_decay=3e-4)
        L_exp = L.ExposureValueControlLoss(patch_size=16, mean_val=E).to(self.device)
        L_tv = L.TotalVariationLoss().to(self.device)
        L_de = ConfidenceGatedDepthLoss().to(self.device)

        # 2. Define sampler
        sampler = JitteredGridSampler(
            image=image_i,
            depth=depth,
            patch_size=self.hidden_dim,
            device=self.device,
        )

        # 3. Patch-based training
        for i in range(epochs):
            model.train()
            loss_epoch = 0.0
            num_items = 0

            # Use the iterator to cover the whole image
            iterator = sampler.get_epoch_iterator(batch_size=batch_size)

            for batch_i, batch_d, batch_coords in iterator:
                # 3.1. Preprocess
                batch_i = batch_i.to(self.device)
                batch_d = batch_d.to(self.device) if batch_d is not None else None
                batch_coords = batch_coords.to(self.device)

                # 3.2. Extract features from patches
                # Input: (B, 1, 64, 64) -> Output: (B, 64, 64, patch_dim)
                features_i = get_local_features(batch_i, kernel_size=self.window_size)
                features_d = get_local_features(batch_d, kernel_size=self.window_size) if batch_d is not None else None

                # 3.3. Concatenate features into a single tensor (The Fusion)
                # We stack them along the last dimension
                if features_d is not None:
                    features = torch.cat([features_i, features_d], dim=-1)  # (B, 64, 64, 18)
                else:
                    features = features_i

                # 3.4. Flatten for INR network
                # (B, H, W, patch_dim) -> (B*H*W, patch_dim)
                flat_coords = batch_coords.reshape(-1, 2)
                flat_patches = features.reshape(-1, features.shape[-1])

                # 3.5. Forward pass
                optimizer.zero_grad()

                # Returns: residual + noise (optional)
                flat_i_res = model(coords=flat_coords, patches=flat_patches)

                # Reshape back to images for Loss calculation
                # (N, 1) -> (B, 1, 64, 64)
                batch_i_res = flat_i_res.view(batch_i.shape)

                # 3.6. Retinex reconstruction
                batch_i_fixed = batch_i + batch_i_res       # Illumination
                batch_r = batch_i / (batch_i_fixed + 1e-4)  # Reflectance

                # 3.7. Loss (Computed on patches)
                l_spa = torch.mean(torch.abs(torch.pow(batch_i_fixed - batch_i, 2)))  # Spatial loss
                l_tv = L_tv(batch_i_fixed)                # TV loss
                l_exp = torch.mean(L_exp(batch_i_fixed))  # Exposure loss
                l_spar = torch.mean(batch_r)              # Sparsity loss
                l_de = L_de(batch_i, batch_d, batch_i_fixed) if batch_d is not None else 0.0
                loss = l_spa + (20 * l_tv) + (8 * l_exp) + (5 * l_spar) + (10 * l_de)

                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                num_items += 1

            # 3.8. Log debugging information
            if self.verbose:
                log(f"Epoch {i+1:4d}/{epochs:4d}: Loss = {(loss_epoch / num_items):6.2f}")

    def _optimize_residual_stochastic(
        self,
        model: nn.Module,
        image_i: Tensor,
        depth: Tensor = None,
        epochs: int = 100,
        batch_size: int = 500000,
        E: float = 0.1,
    ):
        """Optimize the INR network on one input sample.

        Args:
            model (nn.Module): INR network to optimize.
            image_i (Tensor): Illumination map tensor of shape (B, 1, H, W) and
                values ranging from 0.0 to 1.0.
            depth (Tensor, optional): Depth map tensor of shape (B, 1, H, W)
                and values ranging from 0.0 to 1.0. Defaults to None.
            epochs (int, optional): Number of optimization steps. Defaults to 100.
            batch_size (int, optional): Batch size. Defaults to 500,000.
            E (float, optional): Well-exposedness level E. Defaults to 0.1.
        """
        # 1. Define optimizer & losses
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4, weight_decay=3e-4)
        L_tv = ExplicitSmoothnessLoss().to(self.device)

        # 2. Define sampler
        sampler = RandomPixelSampler(
            image=image_i,
            depth=depth,
            window_size=self.window_size,
            device=self.device,
        )

        # 3. Patch-based training
        for i in range(epochs):
            model.train()
            optimizer.zero_grad()

            # 3.1. Sample a batch of pixels and their features
            coords, features, features_i = sampler.sample_batch(batch_size=batch_size)

            # 3.2. Flatten for INR network
            # (B, H, W, patch_dim) -> (B*H*W, patch_dim)
            flat_coords = coords.reshape(-1, 2)
            flat_patches = features.reshape(-1, features.shape[-1])
            flat_feature_i = features_i.reshape(-1, 1)

            # 3.3. Forward pass
            flat_i_res = model(coords=flat_coords, patches=flat_patches)

            # 3.4. Retinex reconstruction
            flat_i_fixed = flat_feature_i + flat_i_res
            flat_r = flat_feature_i / (flat_i_fixed + 1e-4)

            # 3.5. Calculate loss
            # Spatial Loss: Illumination should stay close to Input
            l_spa = torch.mean(torch.abs((flat_i_fixed - flat_feature_i) ** 2))
            # TV Loss: Replaced by Gradient Penalty (explicit_smoothness_loss)
            l_tv = L_tv(flat_i_fixed, flat_coords)
            # Exposure Loss: Mean of batch should match E
            l_exp = torch.mean((torch.mean(flat_i_fixed) - E) ** 2)
            # Sparsity Loss: Reflectance should be sparse (dark)
            l_spar = torch.mean(flat_r)
            # Weighted Sum (Using your original weights)
            loss = l_spa + (20 * l_tv) + (8 * l_exp) + (5 * l_spar)

            loss.backward()
            optimizer.step()

            # 3.8. Log debugging information
            if self.verbose:
                log(f"Epoch {i+1:4d}/{epochs:4d}: Loss = {loss:6.2f}")

    @torch.no_grad()
    def _infer_residual(
        self,
        model: nn.Module,
        image_i: Tensor,
        depth: Tensor = None,
        tile_size: int = 256
    ) -> Tensor:
        """Query the trained INR model for the full-resolution illumination
        residual map.

        Args:
            model (nn.Module): Trained INR network.
            image_i: Illumination map tensor of shape (B, 1, H, W) and values
                ranging from 0.0 to 1.0.
            depth (Tensor, optional): Depth map tensor of shape (B, 1, H, W)
                and values ranging from 0.0 to 1.0. Defaults to None.
            tile_size (int, optional): Tile size for inference. Defaults to 256.

        Returns:
            Tensor: The full resolution illumination residual map tensor of
                shape (B, 1, H, W) and values ranging from 0.0 to 1.0.
        """
        model.eval()

        H, W = image_i.shape[-2:]
        image_i_res = torch.zeros_like(image_i).to(self.device)
        tile_size = tile_size or self.hidden_dim  # 256

        for h in range(0, H, tile_size):
            for w in range(0, W, tile_size):
                # 1. Get tile coords
                h_end = min(h + tile_size, H)
                w_end = min(w + tile_size, W)
                real_h = h_end - h
                real_w = w_end - w

                # 2. Generate coords for tile
                y_range = torch.linspace(-1 + 2*h/(H-1), -1 + 2*(h_end-1)/(H-1), real_h)
                x_range = torch.linspace(-1 + 2*w/(W-1), -1 + 2*(w_end-1)/(W-1), real_w)
                grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
                coords = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, h, w, 2)
                coords = coords.to(self.device)

                # 3. Get features
                # (1, 1, H, W) -> (1, h, w, FeatDim)
                # We crop the input V to the relevant area (plus padding) for speed,
                # or just pass the whole image to get_nearest_features if memory allows.
                # For safety, let's use the helper on the whole image:
                features_i = get_nearest_features(image_i, coords, kernel_size=self.window_size)
                features_d = get_nearest_features(depth,   coords, kernel_size=self.window_size) if depth is not None else None

                # 4. Concatenate features
                if features_d is not None:
                    features = torch.cat([features_i, features_d], dim=-1)  # (1, h, w, 18)
                else:
                    features = features_i

                # 5. Forward pass
                flat_coords = coords.reshape(-1, 2)
                flat_patches = features.reshape(-1, features.shape[-1])
                flat_illu_res = model(coords=flat_coords, patches=flat_patches)

                # 6. Place in canvas
                image_i_res[:, :, h:h_end, w:w_end] = flat_illu_res.view(1, 1, real_h, real_w)

        return image_i_res

    def _refine_reflectance(
        self,
        image_r: Tensor,
        image_i: Tensor,
        sigma_color: float = 0.1,
        sigma_space: float = 1.5,
    ) -> Tensor:
        """Refine the reflectance map using a simple post-processing step."""
        # 1. Joint Bilateral filter
        # Radius is typically 3 * sigma
        radius = int(3 * sigma_space)
        if radius % 2 == 0:
            radius += 1 # Ensure odd
        kernel_size = (radius, radius)

        image_r = kornia.filters.joint_bilateral_blur(
            input=image_r,
            guidance=image_i,
            kernel_size=kernel_size,
            sigma_color=sigma_color,
            sigma_space=(sigma_space, sigma_space)
        )

        return image_r

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---


# --- Model Variants ---

cfgs: dict[str, dict] = {
    "s": {
        "window_size": 7,
        "hidden_dim": 256,
        "num_layers": 4,
        "add_layers": 2,
        "mapping_size": 256,
        "B": 30.0,
    },
}


def _saleo(cfg: str, *args, **kwargs):
    kwargs = cfgs[cfg] | kwargs
    return Saleo(*args, **kwargs)


@MODELS.register(name="saleo_siren", metaclass=Saleo)
def saleo_siren(*args, **kwargs):
    """Create a SALEO model with SIREN."""
    return _saleo(
        cfg="s",
        name="saleo_siren",
        inr="siren",
        pos_encode=False,
        *args, **kwargs
    )


@MODELS.register(name="saleo_ffsiren", metaclass=Saleo)
def saleo_ffsiren(*args, **kwargs):
    """Create a SALEO model with FF+SIREN."""
    return _saleo(
        cfg="s",
        name="saleo_ffsiren",
        inr="siren",
        pos_encode=True,
        *args, **kwargs
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
