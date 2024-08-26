#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements ZS-N2N (Zero-Shot Noise2Noise) models."""

from __future__ import annotations

__all__ = [
    "ZSN2N",
]

import cv2
import torch
from torch.nn.common_types import _size_2_t

from mon import core, nn
from mon.globals import MODELS, Scheme, Task
from mon.vision.enhance import base

console = core.console


# region Model

@MODELS.register(name="zsn2n", arch="zsn2n")
class ZSN2N(base.ImageEnhancementModel):
    """Zero-Shot Noise2Noise.
    
    Args:
        in_channels: The first layer's input channel. Default: ``3`` for RGB image.
        num_channels: Output channels for subsequent layers. Default: ``48``.
    
    References:
        https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing#scrollTo=Srf0GQTYrkxA
    """
    
    arch   : str          = "zsn2n"
    tasks  : list[Task]   = [Task.DENOISE]
    schemes: list[Scheme] = [Scheme.ZERO_REFERENCE, Scheme.INSTANCE]
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
        self.assert_datapoint(datapoint)
        noisy                = datapoint.get("image")
        noisy1, noisy2       = self.pair_downsampler(noisy)
        datapoint1           = datapoint | {"image": noisy1}
        datapoint2           = datapoint | {"image": noisy2}
        outputs1             = self.forward(datapoint=datapoint1, *args, **kwargs)
        outputs2             = self.forward(datapoint=datapoint2, *args, **kwargs)
        outputs              = self.forward(datapoint=datapoint,  *args, **kwargs)
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
    
    def infer(
        self,
        datapoint    : dict,
        image_size   : _size_2_t = 512,
        resize       : bool      = False,
        epochs       : int       = 3000,
        lr           : float     = 0.001,
        step_size    : int       = 1000,
        gamma        : float     = 0.5,
        reset_weights: bool      = True,
    ) -> dict:
        """Infer the model on a single datapoint. This method is different from
        :obj:`forward()` in term that you may want to perform additional
        pre-processing or post-processing steps.
        
        Notes:
            If you want to perform specific pre-processing or post-processing
            steps, you should override this method.
        
        Args:
            datapoint: A :obj:`dict` containing the attributes of a datapoint.
            image_size: The input size. Default: ``512``.
            resize: Resize the input image to the model's input size. Default: ``False``.
            epochs: Maximum number of epochs. Default: ``3000``.
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
        
        # Pre-processing
        self.assert_datapoint(datapoint)
        image  = datapoint.get("image")
        h0, w0 = core.get_image_size(image)
        for k, v in datapoint.items():
            if core.is_image(v):
                if resize:
                    datapoint[k] = core.resize(v, image_size)
                else:
                    datapoint[k] = core.resize_divisible(v, 32)
        for k, v in datapoint.items():
            if isinstance(v, torch.Tensor):
                datapoint[k] = v.to(self.device)
        
        # Training
        for _ in range(epochs):
            outputs = self.forward_loss(datapoint=datapoint)
            optimizer.zero_grad()
            loss = outputs["loss"]
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
        
        # Forward
        self.eval()
        timer = core.Timer()
        timer.tick()
        outputs = self.forward(datapoint=datapoint)
        # with torch.no_grad():
        #    pred = torch.clamp(self.forward(input=input), 0, 1)
        timer.tock()
        self.assert_outputs(outputs)
        
        # Post-processing
        for k, v in outputs.items():
            if core.is_image(v):
                h1, w1 = core.get_image_size(v)
                if h1 != h0 or w1 != w0:
                    outputs[k] = core.resize(v, (h0, w0))
        
        # Return
        outputs["time"] = timer.avg_time
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
    outputs   = net.infer(datapoint)
    denoise   = outputs.get("enhanced")
    denoise   = core.to_image_nparray(denoise, False, True)
    cv2.imshow("Image",    image)
    cv2.imshow("Denoised", denoise)
    cv2.waitKey(0)


if __name__ == "__main__":
    run_zsn2n()

# endregion
