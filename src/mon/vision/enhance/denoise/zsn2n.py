#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements ZS-N2N (Zero-Shot Noise2Noise) models."""

from __future__ import annotations

__all__ = [
    "ZSN2N",
]

import cv2
import torch

from mon import core, nn
from mon.globals import MODELS, Scheme
from mon.vision.enhance.denoise import base

console = core.console


# region Model

@MODELS.register(name="zsn2n", arch="zsn2n")
class ZSN2N(base.DenoisingModel):
    """ZS-N2N (Zero-Shot Noise2Noise).
    
    Args:
        in_channels: The first layer's input channel. Default: ``3`` for RGB image.
        num_channels: Output channels for subsequent layers. Default: ``48``.
    
    References:
        `<https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing#scrollTo=Srf0GQTYrkxA>`_
    
    See Also: :class:`base.DenoisingModel`.
    """
    
    arch   : str  = "zsn2n"
    schemes: list[Scheme] = [Scheme.UNSUPERVISED, Scheme.ZERO_SHOT, Scheme.INSTANCE]
    zoo    : dict = {}
    
    def __init__(
        self,
        in_channels : int = 3,
        num_channels: int = 48,
        *args, **kwargs
    ):
        super().__init__(
            name        = "zsn2n",
            in_channels = in_channels,
            *args, **kwargs
        )
        
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            in_channels  = self.weights.get("in_channels" , in_channels)
            num_channels = self.weights.get("num_channels", num_channels)
        self.in_channels  = in_channels
        self.num_channels = num_channels
        
        # Construct network
        self.conv1 = nn.Conv2d(self.in_channels,  self.num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(self.num_channels, self.out_channels, kernel_size=1)
        self.act   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # Loss
        self._loss = None
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
    
    def init_weights(self, m: nn.Module):
        pass
    
    # region Forward Pass
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        noisy                = datapoint.get("image")
        noisy1, noisy2       = self.pair_downsampler(noisy)
        datapoint1           = datapoint | {"image": noisy1}
        datapoint2           = datapoint | {"image": noisy2}
        outputs1             = self.forward(datapoint=datapoint1, *args, **kwargs)
        outputs2             = self.forward(datapoint=datapoint2, *args, **kwargs)
        outputs              = self.forward(datapoint=datapoint,  *args, **kwargs)
        self.assert_datapoint(datapoint)
        self.assert_outputs(outputs)
        # Symmetric Loss
        pred1                = noisy1 - outputs1["enhanced"]
        pred2                = noisy2 - outputs2["enhanced"]
        noisy_denoised       =  noisy - outputs["enhanced"]
        denoised1, denoised2 = self.pair_downsampler(noisy_denoised)
        mse_loss  = nn.MSELoss()
        loss_res  = 0.5 * (mse_loss(noisy1, pred2)    + mse_loss(noisy2, pred1))
        loss_cons = 0.5 * (mse_loss(pred1, denoised1) + mse_loss(pred2, denoised2))
        loss      = loss_res + loss_cons
        # loss      = nn.reduce_loss(loss=loss, reduction="mean")
        outputs["loss"] = loss
        # Return
        return outputs
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x = datapoint.get("image")
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        y = self.conv3(x)
        if self.predicting:
            y = torch.clamp(y, 0, 1)
        return {"enhanced": y}
    
    # endregion
    
    # region Training
    
    def fit_one(
        self,
        datapoint    : dict,
        max_epochs   : int   = 3000,
        lr           : float = 0.001,
        step_size    : int   = 1000,
        gamma        : float = 0.5,
        reset_weights: bool  = True,
    ) -> torch.Tensor:
        """Train the model with a single datapoint. This method is used for any
        learning scheme performed on one single instance such as online learning,
        zero-shot learning, one-shot learning, etc.
        
        Note:
            In order to use this method, the model must implement the optimizer
            and/or scheduler.
        
        Args:
            datapoint: A :class:`dict` containing the attributes of a datapoint.
            max_epochs: Maximum number of epochs. Default: ``3000``.
            lr: Learning rate. Default: ``0.001``.
            step_size: Period of learning rate decay. Default: ``1000``.
            gamma: A multiplicative factor of learning rate decay. Default: ``0.5``.
            reset_weights: Whether to reset the weights before training. Default: ``True``.
        """
        # Initialize training components
        self.train()
        if reset_weights:
            self.load_state_dict(self.initial_state_dict)
        if isinstance(self.optims, dict):
            optimizer = self.optims.get("optimizer", None)
            scheduler = self.optims.get("scheduler", None)
        else:
            optimizer = nn.Adam(self.parameters(), lr=lr)
            scheduler = nn.StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        '''
        # Prepare input
        if isinstance(input, np.ndarray):
            input = core.to_image_tensor(input, False, True)
        input = input.to(self.device)
        assert input.shape[0] == 1
        datapoint = {"input": input, "target": None}
        '''
        
        # Training loop
        if self.verbose:
            with core.get_progress_bar() as pbar:
                for _ in pbar.track(
                    sequence    = range(max_epochs),
                    total       = max_epochs,
                    description = f"[bright_yellow] Training"
                ):
                    outputs = self.forward_loss(datapoint=datapoint)
                    optimizer.zero_grad()
                    loss = outputs["loss"]
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    scheduler.step()
        else:
            for _ in range(max_epochs):
                outputs = self.forward_loss(datapoint=datapoint)
                optimizer.zero_grad()
                loss = outputs["loss"]
                loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()
        
        # Post-processing
        self.eval()
        outputs = self.forward(datapoint=datapoint)
        # with torch.no_grad():
        #    pred = torch.clamp(self.forward(input=input), 0, 1)
        return outputs
        
    # endregion
    
# endregion


# region Main

def run_zsn2n():
    path      = core.Path("./data/00691.png")
    image     = cv2.imread(str(path))
    datapoint = {"image": core.to_image_tensor(image, False, True)}
    device    = torch.device("cuda:0")
    net       = ZSN2N(channels=3, num_channels=64).to(device)
    outputs   = net.fit_one(datapoint)
    denoise   = outputs.get("enhanced")
    denoise   = core.to_image_nparray(denoise, False, True)
    cv2.imshow("Image",    image)
    cv2.imshow("Denoised", denoise)
    cv2.waitKey(0)


if __name__ == "__main__":
    run_zsn2n()

# endregion
