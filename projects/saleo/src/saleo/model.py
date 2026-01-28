#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SALEO models and pre-trained weights.

This module provides the SALEO definition and pre-trained weights.

References:
    - Paper: "Scale-Arbitrary Low-Light Enhancement via Depth-Aware Implicit
      Neural Optimization"
    - Code: https://github.com/phlong3105/saleo
"""

from __future__ import annotations

__all__ = [
    "SALEO",
    "saleo_b_siren",
    "saleo_b_siren_ff",
]

import kornia
import torch

from mon import nn
from mon.core import create_device, log, MLType, MODELS, Path, Task
from mon.training import loss as L
from .loss import ConfidenceGatedDepthLoss
from .module import SIREN
from .utils import (
    get_local_features,
    get_nearest_features,
    JitteredGridSampler,
)

current_file = Path(__file__).normalize()
current_dir  = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES & MIXINS
# ==============================================================================

# --- Base Classes ---

class SALEO(nn.Module, nn.RegistrableMixin):
    """SALEO model.

    References:
        - Paper: "Scale-Arbitrary Low-Light Enhancement via Depth-Aware Implicit
          Neural Optimization"
        - Code: https://github.com/phlong3105/saleo
    """

    _arch     : str          = "saleo"
    _name     : str          = "saleo"
    _tasks    : list[Task]   = [Task.LLE]
    _mltypes  : list[MLType] = [MLType.TEST_TIME]
    _model_dir: Path         = current_dir
    _inr_funcs: dict         = {
        "siren": SIREN,
    }

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name       : str,
        inr        : str,
        window_size: int,
        hidden_dim : int,
        num_layers : int,
        add_layers : int,
        pos_encode : bool,
        E          : float,
        device     : torch.device = torch.device("cpu"),
        verbose    : bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name: Name of the model variant.
            inr: Name of the INR model to use.
            window_size: Size of the patch window.
            hidden_dim: Hidden dimension of the networks.
            num_layers: Number of layers in each branch.
            add_layers: Number of layers to add between the two branches.
            pos_encode: Whether to use positional encoding.
            E: Well-exposedness level E.
            device: Device to use for computation. Defaults to "cpu".
            verbose: Verbosity mode. Defaults to True.
        """
        # Satisfy PyTorch's empty signature first.
        super().__init__()
        # Initialize RegistrableMixin
        nn.RegistrableMixin.__init__(self, name=name)

        # Validate inputs
        if inr not in self._inr_funcs:
            raise KeyError(f"Expected 'inr' in {list(self._inr_funcs.keys())}, but got '{inr}'.")

        # Assign attributes
        self.verbose     = verbose
        self.inr_func    = self._inr_funcs[inr]
        self.window_size = window_size
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.add_layers  = add_layers
        self.pos_encode  = pos_encode
        self.E           = E
        self.device      = create_device(device)

    # --- Callable & Context Manager ---
    def forward(
        self,
        image       : torch.Tensor,
        depth       : torch.Tensor = None,
        epochs      : int          = 100,
        batch_size  : int          = 1,
        save_weights: bool         = False,
        save_debug  : bool         = False,
    ) -> dict:
        """Forward the input through the network.

        For each input sample, a corresponding INR network is created and optimized
        to fit the illumination map. The final enhanced image is then reconstructed
        using the learned illumination map.

        Args:
            image: Image, formatted as a torch.Tensor of shape (B, 3, H, W)
                and values ranging from 0.0 to 1.0.
            depth: Depth map, formatted as a torch.Tensor of shape (B, 1, H, W)
                and values ranging from 0.0 to 1.0. Defaults to None.
            epochs: Number of optimization steps. Defaults to 100.
            batch_size: Batch size. Defaults to 1.
            save_weights: Whether to save the trained weights. Defaults to False.
            save_debug: Whether to save intermediate results for debugging.
                Defaults to False.
        """
        # 1. Create INR network
        if depth is not None:
            patch_dim = self.window_size ** 2 * 2
        else:
            patch_dim = self.window_size ** 2

        model = self.inr_func(
            patch_dim  = patch_dim,
            hidden_dim = self.hidden_dim,
            num_layers = self.num_layers,
            add_layers = self.add_layers,
            pos_encode = self.pos_encode,
        ).to(self.device)

        # 2. Convert image to HSV
        image     = image.to(self.device)
        image_hsv = kornia.color.rgb_to_hsv(image)
        image_hv  = image_hsv[:, 0:2, :, :]
        image_v   = image_hsv[:, 2:3, :, :]

        # 3. Optimize INR network
        self._optimize(
            model      = model,
            illu       = image_v,
            depth      = depth,
            epochs     = epochs,
            batch_size = batch_size
        )

        # 4. Infer full-resolution illumination map
        illu_res = self._infer(model=model, illu=image_v, depth=depth)

        # 5. Retinex reconstruction
        illu            = illu_res + image_v
        image_v_fixed   = image_v / (illu + 1e-4)

        # 6. Convert back to RGB
        image_hsv_fixed = torch.cat((image_hv, image_v_fixed), dim=1)
        image_rgb_fixed = kornia.color.hsv_to_rgb(image_hsv_fixed)

        # 7. Normalize
        image_rgb_fixed = image_rgb_fixed / torch.max(image_rgb_fixed)
        image_rgb_fixed = torch.clamp(image_rgb_fixed, 0, 1)

        # 8. Return final and intermediate results for debugging
        outputs = { "enhanced": image_rgb_fixed }
        if save_debug:
            outputs |= {
                "image_v"      : image_v,
                "image_v_fixed": image_v_fixed,
                "residual"     : illu_res,
            }
        if save_weights:
            outputs |= { "model": model.state_dict() }
        return outputs

    def _optimize(
        self,
        model     : nn.Module,
        illu      : torch.Tensor,
        depth     : torch.Tensor = None,
        epochs    : int          = 100,
        batch_size: int          = 1
    ):
        """Optimize the INR network on one input sample.

        Args:
            model: INR network to optimize.
            illu: Illumination map (i.e., Value channel), formatted as a
                torch.Tensor of shape (B, 1, H, W) and values ranging from
                0.0 to 1.0.
            depth: Depth map, formatted as a torch.Tensor of shape (B, 1, H, W)
                and values ranging from 0.0 to 1.0. Defaults to None.
            epochs: Number of optimization steps. Defaults to 100.
            batch_size: Batch size. Defaults to 1.
        """
        # 1. Define optimizer & losses
        optimizer = torch.optim.Adam(
            params       = model.parameters(),
            lr           = 1e-4,
            betas        = (0.9, 0.999),
            weight_decay = 3e-4,
        )
        L_exp = L.ExposureValueControlLoss(
            patch_size   = 16,
            mean_val     = self.E,
            channel_mean = True,
        ).to(self.device)
        L_tv  = L.TotalVariationLoss().to(self.device)
        L_str = ConfidenceGatedDepthLoss().to(self.device)

        # 2. Define patch sampler
        sampler = JitteredGridSampler(image=illu, depth=depth, patch_size=self.hidden_dim)

        # 3. Training loop
        for i in range(epochs):
            model.train()
            loss_epoch = 0.0

            # Use the iterator to cover the whole image
            iterator = sampler.get_epoch_iterator(batch_size=batch_size)

            for batch_i, batch_d, batch_coords in iterator:
                # 3.1. Preprocess
                batch_i      = batch_i.to(self.device)
                batch_d      = batch_d.to(self.device) if batch_d is not None else None
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
                flat_coords  = batch_coords.reshape(-1, 2)
                flat_patches = features.reshape(-1, features.shape[-1])

                # 3.5. Forward pass
                optimizer.zero_grad()

                # Returns (N, 1)
                flat_illu_res, _ = model(coords=flat_coords, patches=flat_patches)

                # Reshape back to images for Loss calculation
                # (N, 1) -> (B, 1, 64, 64)
                batch_illu_res = flat_illu_res.view(batch_i.shape)

                # 3.6. Retinex reconstruction
                batch_illu     = batch_illu_res + batch_i
                batch_i_fixed  = batch_i / (batch_illu + 1e-4)

                # 3.7. Loss (Computed on patches)
                l_spa  = torch.mean(torch.abs(torch.pow(batch_illu - batch_i, 2)))  # Spatial loss
                l_tv   = L_tv(batch_illu)               # TV loss
                l_exp  = torch.mean(L_exp(batch_illu))  # Exposure loss
                l_spar = torch.mean(batch_i_fixed)      # Sparsity loss
                l_str  = L_str(batch_i, batch_d, batch_illu) if batch_d is not None else 0.0
                loss   = l_spa + (20 * l_tv) + (8 * l_exp) + (5 * l_spar) + l_str

                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()

            # 4. Log debugging information
            if self.verbose:
                log(f"Epoch {i+1}/{epochs}: Loss = {loss_epoch:.4f}")

    @torch.no_grad()
    def _infer(
        self,
        model: nn.Module,
        illu : torch.Tensor,
        depth: torch.Tensor = None
    ) -> torch.Tensor:
        """Query the trained INR model for the full-resolution illumination map.

        Args:
            model: INR model to optimize.
            illu: Illumination map (i.e., Value channel), formatted as a
                torch.Tensor of shape (B, 1, H, W) and values ranging from
                0.0 to 1.0.
            depth: Depth map, formatted as a torch.Tensor of shape (B, 1, H, W)
                and values ranging from 0.0 to 1.0. Defaults to None.

        Returns:
            The full resolution illumination map, formatted as a torch.Tensor of
            shape (B, 1, H, W) and values ranging from 0.0 to 1.0.
        """
        model.eval()

        illu_res  = torch.zeros_like(illu)
        H, W      = illu.shape[-2:]
        tile_size = self.hidden_dim  # 256

        for h in range(0, H, tile_size):
            for w in range(0, W, tile_size):
                # 1. Get tile coords
                h_end  = min(h + tile_size, H)
                w_end  = min(w + tile_size, W)
                real_h = h_end - h
                real_w = w_end - w

                # 2. Generate coords for tile
                y_range = torch.linspace(-1 + 2*h/(H-1), -1 + 2*(h_end-1)/(H-1), real_h)
                x_range = torch.linspace(-1 + 2*w/(W-1), -1 + 2*(w_end-1)/(W-1), real_w)
                grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
                coords  = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0) # (1, h, w, 2)
                coords  = coords.to(self.device)

                # 3. Get features
                # (1, 1, H, W) -> (1, h, w, FeatDim)
                # We crop the input V to the relevant area (plus padding) for speed,
                # or just pass the whole image to get_nearest_features if memory allows.
                # For safety, let's use the helper on the whole image:
                features_i = get_nearest_features(illu,  coords, kernel_size=self.window_size)
                features_d = get_nearest_features(depth, coords, kernel_size=self.window_size) if depth is not None else None

                # 4. Concatenate features
                if features_d is not None:
                    features = torch.cat([features_i, features_d], dim=-1)  # (1, h, w, 18)
                else:
                    features = features_i

                # 5. Forward pass
                flat_coords      = coords.reshape(-1, 2)
                flat_patches     = features.reshape(-1, features.shape[-1])
                flat_illu_res, _ = model(coords=flat_coords, patches=flat_patches)

                # 6. Place in canvas
                illu_res[:, :, h:h_end, w:w_end] = flat_illu_res.view(1, 1, real_h, real_w)

        return illu_res


# --- Mixins ---


# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---


# --- Model Variants ---

cfgs: dict[str, dict] = {
    "b": {
        "window_size": 7,
        "hidden_dim" : 256,
        "num_layers" : 4,
        "add_layers" : 2,
    },
}


def _saleo(cfg: str, *args, **kwargs):
    kwargs = cfgs[cfg] | kwargs
    return SALEO(*args, **kwargs)


@MODELS.register(name="saleo_b_siren", metaclass=SALEO)
def saleo_b_siren(E: float = 0.5, verbose: bool = False, *args, **kwargs):
    """Create a SALEO-B-SIREN model.

    Args:
        E: Well-exposedness level E. Defaults to 0.5.
        verbose: Verbosity mode. Defaults to False.
        args: Additional positional arguments for the SALEO model.
        kwargs: Additional keyword arguments for the SALEO model.

    Returns:
        A SALEO-B-SIREN model instance.
    """
    return _saleo(
        cfg        = "b",
        name       = "saleo_b_siren",
        inr        = "siren",
        pos_encode = False,
        E          = E,
        verbose    = verbose,
        *args, **kwargs
    )


@MODELS.register(name="saleo_b_siren_ff", metaclass=SALEO)
def saleo_b_siren_ff(E: float = 0.5, verbose: bool = False, *args, **kwargs):
    """Create a SALEO-B-SIREN-FF model.

    Args:
        E: Well-exposedness level E. Defaults to 0.5.
        verbose: Verbosity mode. Defaults to False.
        args: Additional positional arguments for the SALEO model.
        kwargs: Additional keyword arguments for the SALEO model.

    Returns:
        A SALEO-B-SIREN-FF model instance.
    """
    return _saleo(
        cfg        = "b",
        name       = "saleo_b_siren_ff",
        inr        = "siren",
        pos_encode = True,
        E          = E,
        verbose    = verbose,
        *args, **kwargs
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
