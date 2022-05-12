#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RetinexNet.
"""

from __future__ import annotations

import os
from copy import deepcopy
from enum import Enum
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from one.core import EpochOutput
from one.core import ForwardOutput
from one.core import Indexes
from one.core import Int2T
from one.core import MODELS
from one.core import Pretrained
from one.core import StepOutput
from one.core import Tensors
from one.core import to_2tuple
from one.imgproc import imshow_plt
from one.nn import BaseModel
from one.vision.enhancement.retinexnet.loss import DecomLoss
from one.vision.enhancement.retinexnet.loss import EnhanceLoss
from one.vision.enhancement.retinexnet.loss import RetinexLoss

__all__ = [
    "DecomNet",
    "EnhanceNet",
    "EnhanceUNet",
    "RetinexNet",
    "ModelState"
]


# MARK: - Modules

class DecomNet(nn.Module):
    """DecomNet is one of the two sub-networks used in RetinexNet model.
    DecomNet breaks the RGB image into a 1-channel intensity map and a
    3-channels reflectance map.

    Attributes:
        num_activation_layers (int):
            Number of activation layers. Default: `5`.
        channels (int):
            Number of output channels (or filtering) for the `Conv2D` layer
            in the decomnet. Default: `64`.
        kernel_size (Int2T):
            Kernel size for the `Conv2D` layer in the decomnet.
            Default: `3`.
        use_batchnorm (bool):
            If `True`, use Batch Normalization layer between `Conv2D` and
            `Activation` layers. Default: `True`.
        name (str):
            Name of the backbone. Default: `decomnet`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        num_activation_layers: int    = 5,
        channels             : int    = 64,
        kernel_size          : Int2T = 3,
        use_batchnorm        : bool   = True,
        *args, **kwargs
    ):
        super().__init__()
        self.name 	  	   = "decomnet"
        self.use_batchnorm = use_batchnorm
        
        channels      = channels
        kernel_size   = to_2tuple(kernel_size)
        kernel_size_3 = tuple([i * 3 for i in kernel_size])
        
        convs = []
        # Shallow feature_extractor extraction
        convs.append(nn.Conv2d(4, channels, kernel_size_3, padding=4,
                               padding_mode="replicate"))
        # Activation layers
        for i in range(num_activation_layers):
            convs.append(nn.Conv2d(channels, channels, kernel_size, padding=1,
                                   padding_mode="replicate"))
            convs.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*convs)
        
        # Reconstruction layer
        self.recon = nn.Conv2d(channels, 4, kernel_size, padding=1,
                               padding_mode="replicate")
        self.bn = nn.BatchNorm2d(4)
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensors:
        """Forward pass.

        Args:
            x (Tensor):
                Input of shape [B, C, H, W].

        Returns:
            r (Tensor):
                Reflectance maps [:, 0:3, :, :]
            i (Tensor):
                Illumination maps [:, 3:4, :, :].
        """
        x_max = torch.max(input=x, dim=1, keepdim=True)[0]
        x_cat = torch.cat(tensors=(x_max, x), dim=1)
        x     = self.features(x_cat)
        pred  = self.recon(x)
        if self.use_batchnorm:
            pred = self.bn(pred)
        r = torch.sigmoid(pred[:, 0:3, :, :])
        i = torch.sigmoid(pred[:, 3:4, :, :])
        return r, i
    

# noinspection PyMethodOverriding
class EnhanceNet(nn.Module):
    """EnhanceNet is one of the two sub-networks used in RetinexNet model.
    EnhanceNet increases the light distribution in the 1-channel intensity map.

    Attributes:
        channels (int):
            Number of output channels (or filtering) for the `Conv2D` layer
            in the enhancenet. Default: `64`.
        kernel_size (Int2T):
            Kernel size for the `Conv2D` layer in the enhancenet.
            Default: `3`.
        use_batchnorm (bool):
            If `True`, use Batch Normalization layer between `Conv2D` and
            `Activation` layers. Default: `True`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        channels     : int    = 64,
        kernel_size  : Int2T = 3,
        use_batchnorm: bool   = True,
        *args, **kwargs
    ):
        super().__init__()
        self.name 		   = "enhancenet"
        self.use_batchnorm = use_batchnorm
        
        channels 	= channels
        kernel_size = to_2tuple(kernel_size)
        
        self.relu    = nn.ReLU(inplace=True)
        self.conv0_1 = nn.Conv2d(
            4, channels, kernel_size, padding=1, padding_mode="replicate"
        )
        self.conv1_1 = nn.Conv2d(
            channels, channels, kernel_size, (2, 2), padding=1,
            padding_mode="replicate"
        )
        self.conv1_2 = nn.Conv2d(
            channels, channels, kernel_size, (2, 2), padding=1,
            padding_mode="replicate"
        )
        self.conv1_3 = nn.Conv2d(
            channels, channels, kernel_size, (2, 2), padding=1,
            padding_mode="replicate"
        )
        self.deconv1_1 = nn.Conv2d(
            channels * 2, channels, kernel_size, padding=1,
            padding_mode="replicate"
        )
        self.deconv1_2 = nn.Conv2d(
            channels * 2, channels, kernel_size, padding=1,
            padding_mode="replicate"
        )
        self.deconv1_3 = nn.Conv2d(
            channels * 2, channels, kernel_size, padding=1,
            padding_mode="replicate"
        )
        self.fusion = nn.Conv2d(
            channels * 3, channels, (1, 1), padding=1,
            padding_mode="replicate"
        )
        self.output = nn.Conv2d(channels, 1, (3, 3), padding=0)
        self.bn     = nn.BatchNorm2d(channels)
    
    # MARK: Forward Pass
    
    def forward(self, r_low: Tensor, i_low: Tensor) -> Tensor:
        """Forward pass.

        Args:
            r_low (Tensor):
                Reflectance maps extracted from the low-light images.
            i_low (Tensor):
                Illumination maps extracted from the low-light images.

        Returns:
            pred (Tensor):
                Enhanced illumination map (a.k.a, i_delta).
        """
        x     = torch.cat(tensors=(r_low, i_low), dim=1)
        conv0 = self.conv0_1(x)
        conv1 = self.relu(self.conv1_1(conv0))
        conv2 = self.relu(self.conv1_1(conv1))
        conv3 = self.relu(self.conv1_1(conv2))
        
        conv3_up = F.interpolate(
            input=conv3, size=(conv2.size()[2], conv2.size()[3])
        )
        deconv1 = self.relu(
            self.deconv1_1(torch.cat(tensors=(conv3_up, conv2), dim=1))
        )
        deconv1_up = F.interpolate(
            input=deconv1, size=(conv1.size()[2], conv1.size()[3])
        )
        deconv2 = self.relu(
            self.deconv1_2(torch.cat(tensors=(deconv1_up, conv1), dim=1))
        )
        deconv2_up = F.interpolate(
            input=deconv2, size=(conv0.size()[2], conv0.size()[3])
        )
        deconv3 = self.relu(
            self.deconv1_3(torch.cat(tensors=(deconv2_up, conv0), dim=1))
        )
        
        deconv1_rs = F.interpolate(
            input=deconv1, size=(r_low.size()[2], r_low.size()[3])
        )
        deconv2_rs = F.interpolate(
            input=deconv2, size=(r_low.size()[2], r_low.size()[3])
        )
        feats_all = torch.cat(tensors=(deconv1_rs, deconv2_rs, deconv3), dim=1)
        feats_fus = self.fusion(feats_all)
        
        if self.use_batchnorm:
            feats_fus = self.bn(feats_fus)
        pred = self.output(feats_fus)
        return pred


# noinspection PyMethodOverriding
class EnhanceUNet(nn.Module):
    """EnhanceUNet is a variation of the EnhanceNet that adopts the UNet
    architecture. EnhanceUNet breaks the RGB image into a 1-channel intensity
    map and a 3-channels reflectance map.

    Attributes:
        channels (int):
            Number of output channels (or filtering) for the `Conv2D` layer
            in the enhancenet. Default: `64`.
        kernel_size (Int2T):
            Kernel size for the `Conv2D` layer in the enhancenet.
            Default: `3`.
        fuse_features (bool):
            If `True`, fuse features from all layers at the end.
            Default: `False`.
        use_batchnorm (bool):
            If `True`, use Batch Normalization layer between `Conv2D` and
            `Activation` layers. Default: `True`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        channels     : int 	  = 64,
        kernel_size  : Int2T = 3,
        fuse_features: bool   = False,
        use_batchnorm: bool   = True,
        *args, **kwargs
    ):
        super().__init__()
        self.name 		   = "enhance_unet"
        self.use_batchnorm = use_batchnorm
        self.relu          = nn.ReLU()
        
        channels 	  = channels
        kernel_size   = to_2tuple(kernel_size)
        fuse_features = fuse_features
        
        # Downscale
        conv1_channels = self.channels
        self.conv1_1   = nn.Conv2d(
            4, conv1_channels, kernel_size, padding=1, padding_mode="replicate")
        self.conv1_2 = nn.Conv2d(
            conv1_channels, conv1_channels, kernel_size, (2, 2), padding=1,
            padding_mode="replicate"
        )
        
        conv2_channels = conv1_channels * 2
        self.conv2_1 = nn.Conv2d(
            conv1_channels, conv2_channels, kernel_size, (2, 2), padding=1,
            padding_mode="replicate"
        )
        self.conv2_2 = nn.Conv2d(
            conv2_channels, conv2_channels, kernel_size, (2, 2), padding=1,
            padding_mode="replicate"
        )
        
        conv3_channels = conv2_channels * 2
        self.conv3_1 = nn.Conv2d(
            conv2_channels, conv3_channels, kernel_size, (2, 2), padding=1,
            padding_mode="replicate"
        )
        self.conv3_2 = nn.Conv2d(
            conv3_channels, conv3_channels, kernel_size, (2, 2), padding=1,
            padding_mode="replicate"
        )
        
        conv4_channels = conv3_channels * 2
        self.conv4_1 = nn.Conv2d(
            conv3_channels, conv4_channels, kernel_size, (2, 2), padding=1,
            padding_mode="replicate"
        )
        self.conv4_2 = nn.Conv2d(
            conv4_channels, conv4_channels, kernel_size, (2, 2), padding=1,
            padding_mode="replicate"
        )
        
        conv5_channels = conv4_channels * 2
        self.conv5_1 = nn.Conv2d(
            conv4_channels, conv5_channels, kernel_size, (2, 2), padding=1,
            padding_mode="replicate"
        )
        
        # Upscale
        self.deconv4_1 = nn.Conv2d(
            conv5_channels, conv4_channels, kernel_size, padding=1,
            padding_mode="replicate"
        )
        self.deconv4_2 = nn.Conv2d(
            conv4_channels * 2, conv4_channels, kernel_size, padding=1,
            padding_mode="replicate"
        )
        self.deconv4_3 = nn.Conv2d(
            conv4_channels, conv1_channels, kernel_size, padding=1,
            padding_mode="replicate"
        )
        
        self.deconv3_1 = nn.Conv2d(
            conv4_channels, conv3_channels, kernel_size, padding=1,
            padding_mode="replicate"
        )
        self.deconv3_2 = nn.Conv2d(
            conv3_channels * 2, conv3_channels, kernel_size, padding=1,
            padding_mode="replicate"
        )
        self.deconv3_3 = nn.Conv2d(
            conv3_channels, conv1_channels, kernel_size, padding=1,
            padding_mode="replicate"
        )
        
        self.deconv2_1 = nn.Conv2d(
            conv3_channels, conv2_channels, kernel_size, padding=1,
            padding_mode="replicate"
        )
        self.deconv2_2 = nn.Conv2d(
            conv2_channels * 2, conv2_channels, kernel_size, padding=1,
            padding_mode="replicate"
        )
        self.deconv2_3 = nn.Conv2d(
            conv2_channels, conv1_channels, kernel_size, padding=1,
            padding_mode="replicate"
        )
        
        self.deconv1_1 = nn.Conv2d(
            conv2_channels, conv1_channels, kernel_size, padding=1,
            padding_mode="replicate"
        )
        self.deconv1_2 = nn.Conv2d(
            conv1_channels * 2, conv1_channels, kernel_size, padding=1,
            padding_mode="replicate"
        )
        
        # Fusion and output
        self.fusion = nn.Conv2d(
            conv1_channels * 4, conv1_channels, (1, 1), padding=1,
            padding_mode="replicate"
        )
        self.output = nn.Conv2d(conv1_channels, 1, (3, 3), padding=0)
        self.bn 	= nn.BatchNorm2d(channels)
    
    # MARK: Forward Pass
    
    def forward(self, r: Tensor, i: Tensor) -> Tensor:
        """Forward pass.

        Args:
            r (Tensor):
                Reflectance map.
            i (Tensor):
                Illumination map.

        Returns:
            pred (Tensor):
                Enhanced illumination map (a.k.a, i_delta).
        """
        x = torch.cat(tensors=(r, i), dim=1)
        
        # Downsample path
        conv1_1 = self.conv1_1(x)
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        
        conv2_1 = self.relu(self.conv2_1(conv1_2))
        conv2_2 = self.relu(self.conv2_2(conv2_1))
        
        conv3_1 = self.relu(self.conv3_1(conv2_2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        
        conv4_1 = self.relu(self.conv4_1(conv3_2))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        
        conv5_1 = self.relu(self.conv5_1(conv4_2))
        
        # Upsample path
        deconv4_1    = self.relu(self.deconv4_1(conv5_1))
        deconv4_1_up = F.interpolate(
            input=deconv4_1, size=(conv4_2.size()[2], conv4_2.size()[3])
        )
        deconv4_2 	 = self.relu(
            self.deconv4_2(torch.cat(tensors=(deconv4_1_up, conv4_2), dim=1))
        )
        
        deconv3_1    = self.relu(self.deconv3_1(deconv4_2))
        deconv3_1_up = F.interpolate(
            input=deconv3_1, size=(conv3_2.size()[2], conv3_2.size()[3])
        )
        deconv3_2 	 = self.relu(
            self.deconv3_2(torch.cat(tensors=(deconv3_1_up, conv3_2), dim=1))
        )
        
        deconv2_1    = self.relu(self.deconv2_1(deconv3_2))
        deconv2_1_up = F.interpolate(
            input=deconv2_1, size=(conv2_2.size()[2], conv2_2.size()[3])
        )
        deconv2_2 	 = self.relu(
            self.deconv2_2(torch.cat(tensors=(deconv2_1_up, conv2_2), dim=1))
        )
        
        deconv1_1    = self.relu(self.deconv1_1(deconv2_2))
        deconv1_1_up = F.interpolate(
            input=deconv1_1, size=(conv1_2.size()[2], conv1_2.size()[3])
        )
        deconv1_2 	 = self.relu(
            self.deconv1_2(torch.cat(tensors=(deconv1_1_up, conv1_2), dim=1))
        )
        deconv1_2_rs = F.interpolate(
            input=deconv1_2, size=(r.size()[2], r.size()[3])
        )
        
        final_layer = deconv1_2_rs
        if self.fuse_features:
            deconv4_3    = self.relu(self.deconv4_3(deconv4_2))
            deconv4_3_rs = F.interpolate(
                input=deconv4_3, size=(r.size()[2], r.size()[3])
            )
            deconv3_3    = self.relu(self.deconv3_3(deconv3_2))
            deconv3_3_rs = F.interpolate(
                input=deconv3_3, size=(r.size()[2], r.size()[3])
            )
            deconv2_3    = self.relu(self.deconv2_3(deconv2_2))
            deconv2_3_rs = F.interpolate(
                input=deconv2_3, size=(r.size()[2], r.size()[3])
            )
            feats_all = torch.cat(
                tensors=(deconv4_3_rs, deconv3_3_rs, deconv2_3_rs,
                         deconv1_2_rs),
                dim=1
            )
            final_layer = self.fusion(feats_all)
        
        if self.use_batchnorm:
            final_layer = self.bn(final_layer)
        output = self.output(final_layer)
        pred   = F.interpolate(input=output, size=(r.size()[2], r.size()[3]))
        return pred
    
    
class ModelState(Enum):
    """Phases of the Retinex model."""

    # Train the DecomNet ONLY. Produce predictions, calculate losses and
    # metrics, update weights at the end of each epoch/step.
    DECOMNET   = "decomnet"
    # Train the EnhanceNet ONLY. Produce predictions, calculate losses and
    # metrics, update weights at the end of each epoch/step.
    ENHANCENET = "enhancenet"
    # Train the whole network. Produce predictions, calculate losses and
    # metrics, update weights at the end of each epoch/step.
    RETINEXNET = "retinexnet"

    TRAINING   = "training"
    # Produce predictions, calculate losses and metrics, DO NOT update weights
    # at the end of each epoch/step.
    TESTING    = "testing"
    # Produce predictions ONLY.
    INFERENCE  = "inference"
    
    @staticmethod
    def values() -> list[str]:
        """Return the list of all values.

        Returns:
            (list):
                List of string.
        """
        return [e.value for e in ModelState]

    @staticmethod
    def keys():
        """Return the list of all enum keys.

        Returns:
            (list):
                List of enum keys.
        """
        return [e for e in ModelState]
    

# MARK: - RetinexNet

@MODELS.register(name="retinexnet")
class RetinexNet(BaseModel):
    """Retinex-based models combine two submodels: DecomNet and EnhanceNet.
    RetinexNet is a multi-stage-training enhancer. We have to train the
    DecomNet first and then train the EnhanceNet.
    
    Notes:
        - When training the DecomNet: epoch=75, using Adam(lr=0.00001) gives
          best results.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # Hyperparameters
        decomnet   = DecomNet(
            num_activation_layers=5, channels=64, kernel_size=3,
            use_batchnorm=True
        ),
        enhancenet = EnhanceNet(
            channels=64, kernel_size=3, use_batchnorm=True
        ),
        # BaseModel's args
        basename   : Optional[str] = "retinexnet",
        name       : Optional[str] = "retinexnet",
        num_classes: Optional[int] = None,
        model_state: ModelState    = ModelState.TRAINING,
        out_indexes: Indexes       = -1,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        super().__init__(
            basename    = basename,
            name        = name,
            num_classes = num_classes,
            pretrained  = pretrained,
            out_indexes = out_indexes,
            *args, **kwargs
        )
        # NOTE: Get Hyperparameters
        self.decomnet     = decomnet
        self.enhancenet   = enhancenet
        self.phase 		  = model_state
        self.decom_loss	  = DecomLoss()
        self.enhance_loss = EnhanceLoss()
        self.retinex_loss = RetinexLoss()
        
        # NOTE: Load Pretrained
        if self.pretrained:
            self.load_pretrained()
    
    # noinspection PyAttributeOutsideInit
    @BaseModel.model_state.setter
    def model_state(self, phase: ModelState):
        """Configure the model's running model_state.

        Args:
            phase (ModelState):
                Fphase of the model.
        """
        # assert model_state in ModelState, f"Model's `model_state` must be one of the
        # following values: {ModelState.keys()}"
        self._phase = phase
        
        if self._phase is ModelState.DECOMNET:
            self.decomnet.train()
            self.enhancenet.eval()
        elif self._phase is ModelState.ENHANCENET:
            self.decomnet.eval()
            self.enhancenet.train()
        elif self._phase is ModelState.RETINEXNET:
            self.decomnet.train()
            self.enhancenet.train()
        elif self._phase in [ModelState.TESTING, ModelState.INFERENCE]:
            self.decomnet.eval()
            self.enhancenet.eval()
    
    @BaseModel.debug_image_filepath.getter
    def debug_image_filepath(self) -> str:
        """Return the debug image filepath."""
        save_dir = self.debug_dir
        if self.debugger:
            save_dir = (self.debug_image_dir if self.debugger.save_to_subdir
                        else self.debug_dir)

        return os.path.join(
            save_dir,
            f"{self.phase.value}_"
            f"{(self.current_epoch + 1):03d}_"
            f"{(self.epoch_step + 1):06}.jpg"
        )
    
    # MARK: Forward Pass

    def forward_loss(self, x: Tensor, y: Tensor, *args, **kwargs) -> ForwardOutput:
        """Forward pass with loss value. Loss function may require more
        arguments beside the ground-truth and prediction values.
        For calculating the metrics, we only need the final predictions and
        ground-truth.

        Args:
            x (Tensors):
                Input of shape [B, C, H, W].
            y (Tensors):
                Ground-truth of shape [B, C, H, W].
                
        Returns:
            yhat (Tensors):
                Predictions.
            loss (Tensor, optional):
                Loss.
        """
        if self.phase is ModelState.DECOMNET:
            r_low,  i_low  = self.decomnet(x)
            r_high, i_high = self.decomnet(y)
            loss = self.decom_loss(x, y, r_low, r_high, i_low, i_high)
            yhat = (r_low, r_high, i_low, i_high)
            return yhat, loss
        
        elif self.phase is ModelState.ENHANCENET:
            r_low, i_low = self.decomnet(x)
            i_delta      = self.enhancenet(r_low, i_low)
            i_delta_3 	 = torch.cat((i_delta, i_delta, i_delta), dim=1)
            yhat 		 = r_low * i_delta_3
            loss = self.enhance_loss(y, r_low, i_delta, i_delta_3, yhat)
            yhat = (r_low, i_low, i_delta, yhat)
            return yhat, loss
        
        else:
            r_low,  i_low   = self.decomnet(x)
            r_high, i_high  = self.decomnet(y)
            i_delta         = self.enhancenet(r_low, i_low)
            i_delta_3 	    = torch.cat((i_delta, i_delta, i_delta), dim=1)
            yhat 		    = r_low * i_delta_3
            loss = self.retinex_loss(x, y, r_low, r_high, i_low, i_high, i_delta)
            yhat = (r_low, i_low, i_delta, yhat)
            return yhat, loss
    
    def forward(
        self, x: Tensor, augment: bool = False, *args, **kwargs
    ) -> Tensor:
        """Forward pass. This is the primary `forward` function of the model.
        It supports augmented inference.
        
        In this function, we perform test-time augmentation and pass the
        transformed input to `forward_once()`.

        Args:
            x (Tensor):
                Input of shape [B, C, H, W].
            augment (bool):
                Augmented inference. Default: `False`.
                
        Returns:
            pred (Tensor):
                Predictions.
        """
        if augment:
            # NOTE: For now just forward the input. Later, we will implement
            # the test-time augmentation for image classification
            return self.forward_once(x=x, *args, **kwargs)
        else:
            return self.forward_once(x=x, *args, **kwargs)
    
    def forward_once(self, x: Tensor, *args, **kwargs) -> Tensors:
        """Forward pass once. Implement the logic for a single forward pass.

        Args:
            x (Tensor):
                Input of shape [B, C, H, W].
            
        Returns:
            yhat (Tensors):
                Predictions.
        """
        r_low, i_low = self.decomnet(x)
        i_delta      = self.enhancenet(r_low, i_low)
        i_delta_3 	 = torch.cat((i_delta, i_delta, i_delta), dim=1)
        yhat 		 = r_low * i_delta_3
        yhat         = (r_low, i_low, i_delta, yhat)
        return yhat
    
    # MARK: Training
    
    def training_step(
        self, batch: Any, batch_idx: int, *args, **kwargs
    ) -> Optional[StepOutput]:
        """Training step.

        Args:
            batch (Any):
                Batch of inputs. It can be a tuple of (`x`, `y`, extra).
            batch_idx (int):
                Batch index.

        Returns:
            outputs (StepOutput, optional):
                - A single loss tensor.
                - A dictionary with the first key must be the `loss`.
                - `None`, training will skip to the next batch.
        """
        # NOTE: Forward pass
        x, y, extra = batch[0], batch[1], batch[2:]
        yhat, loss  = self.forward_loss(x=x, y=y, *args, **kwargs )
        
        if self.phase is ModelState.DECOMNET:
            r_low, r_high, i_low, i_high = yhat
            return {
                "loss": loss, "input": x, "y": y, "r_low": r_low,
                "r_high": r_high, "i_low": i_low, "i_high": i_high
            }
        elif self.phase is ModelState.ENHANCENET:
            r_low, i_low, i_delta, yhat = yhat
            return {
                "loss": loss, "input": x, "y": y, "yhat": yhat,
                "r_low": r_low, "i_low": i_low, "i_delta": i_delta
            }
        else:
            r_low, i_low, i_delta, yhat = yhat
            return {
                "loss": loss, "input": x, "y": y, "yhat": yhat,
                "r_low": r_low, "i_low": i_low, "i_delta": i_delta
            }
            
    def training_step_end(
        self, outputs: Optional[StepOutput], *args, **kwargs
    ) -> Optional[StepOutput]:
        """Use this when training with dp or ddp2 because training_step() will
        operate on only part of the batch. However, this is still optional and
        only needed for things like softmax or NCE loss.
        
        Note:
            If you later switch to ddp or some other mode, this will still be
            called so that you don't have to change your code.
        """
        if not isinstance(outputs, dict):
            return None
        
        # NOTE: Gather results
        # For DDP strategy
        if self.trainer.num_processes > 1:
            outputs = self.all_gather(outputs)
        
        loss    = outputs["loss"]               # losses from each GPU
        x       = outputs["x"]                  # images from each GPU
        y       = outputs["y"]                  # ground-truths from each GPU
        yhat    = outputs.get("yhat",    None)  # predictions from each GPU
        r_low   = outputs.get("r_low",   None)
        r_high  = outputs.get("r_high",  None)
        i_low   = outputs.get("i_low",   None)
        i_high  = outputs.get("i_high",  None)
        i_delta = outputs.get("i_delta", None)
        
        # NOTE: Tensors
        if self.trainer.num_processes > 1:
            x       = x.flatten(start_dim=0, end_dim=1)
            y       = y.flatten(start_dim=0, end_dim=1)
            yhat    = yhat.flatten(start_dim=0, end_dim=1)    if yhat is not None else None
            r_low   = r_low.flatten(start_dim=0, end_dim=1)   if r_low is not None else None
            r_high  = r_high.flatten(start_dim=0, end_dim=1)  if r_high is not None else None
            i_low   = i_low.flatten(start_dim=0, end_dim=1)   if i_low is not None else None
            i_high  = i_high.flatten(start_dim=0, end_dim=1)  if i_high is not None else None
            i_delta = i_delta.flatten(start_dim=0, end_dim=1) if i_delta is not None else None
            
        # NOTE: Loss
        loss = loss.mean() if loss is not None else None
        self.ckpt_log_scalar(f"checkpoint/loss/train_step", loss)
        # self.tb_log(f"{loss_tag}", loss, "step")
        
        # NOTE: Metrics
        if self.with_train_metrics:
            for i, metric in enumerate(self.train_metrics):
                if self.phase is ModelState.DECOMNET:
                    value = metric(r_low, r_high)
                else:
                    value = metric(yhat, y)
                self.ckpt_log_scalar(f"checkpoint/{metric.name}/train_step", value, True)
                # self.tb_log(f"{metric.name}/train_step", value, "step")
        
        self.epoch_step += 1
        return {"loss": loss}
    
    def training_epoch_end(self, outputs: EpochOutput):
        # NOTE: Loss
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.ckpt_log_scalar(f"checkpoint/loss/train_epoch", loss)
        self.tb_log_scalar(f"loss/train_epoch", loss, "epoch")
        
        # NOTE: Metrics
        if self.with_train_metrics:
            for i, metric in enumerate(self.train_metrics):
                value = metric.compute()
                self.ckpt_log_scalar(f"checkpoint/{metric.name}/train_epoch", value)
                self.tb_log_scalar(f"{metric.name}/train_epoch", value, "epoch")
                metric.reset()
    
    def validation_step(
        self, batch: Any, batch_idx: int, *args, **kwargs
    ) -> Optional[StepOutput]:
        """Validation step.

        Args:
            batch (Any):
                Batch of inputs. It can be a tuple of (`x`, `y`, extra).
            batch_idx (int):
                Batch index.

        Returns:
            outputs (StepOutput, optional):
                - A single loss image.
                - A dictionary with the first key must be the `loss`.
                - `None`, training will skip to the next batch.
        """
        x, y, extra = batch[0], batch[1], batch[2:]
        yhat, loss = self.forward_loss(x=x, y=y, *args, **kwargs)
        
        if self.phase is ModelState.DECOMNET:
            r_low, r_high, i_low, i_high = yhat
            return {
                "loss": loss, "x": x, "y": y, "r_low": r_low,
                "r_high": r_high, "i_low": i_low, "i_high": i_high
            }
        elif self.phase is ModelState.ENHANCENET:
            r_low, i_low, i_delta, yhat = yhat
            return {
                "loss": loss, "x": x, "y": y, "yhat": yhat,
                "r_low": r_low, "i_low": i_low, "i_delta": i_delta
            }
        else:
            r_low, i_low, i_delta, yhat = yhat
            return {
                "loss": loss, "x": x, "y": y, "yhat": yhat,
                "r_low": r_low, "i_low": i_low, "i_delta": i_delta
            }
        
    def validation_step_end(
        self, outputs: Optional[StepOutput], *args, **kwargs
    ) -> Optional[StepOutput]:
        """Use this when validating with dp or ddp2 because `validation_step`
        will operate on only part of the batch. However, this is still optional
        and only needed for things like softmax or NCE loss.

        Note:
            If you later switch to ddp or some other mode, this will still be
            called so that you don't have to change your code.
        """
        if not isinstance(outputs, dict):
            return None
        
        # NOTE: Gather results
        # For DDP strategy
        if self.trainer.num_processes > 1:
            outputs = self.all_gather(outputs)
        
        loss    = outputs["loss"]               # losses from each GPU
        x       = outputs["x"]                  # images from each GPU
        y       = outputs["y"]                  # ground-truths from each GPU
        yhat    = outputs.get("yhat",    None)  # predictions from each GPU
        r_low   = outputs.get("r_low",   None)
        r_high  = outputs.get("r_high",  None)
        i_low   = outputs.get("i_low",   None)
        i_high  = outputs.get("i_high",  None)
        i_delta = outputs.get("i_delta", None)
        
        # NOTE: Tensors
        if self.trainer.num_processes > 1:
            x       = x.flatten(start_dim=0, end_dim=1)
            y       = y.flatten(start_dim=0, end_dim=1)
            yhat    = yhat.flatten(start_dim=0, end_dim=1)    if yhat is not None else None
            r_low   = r_low.flatten(start_dim=0, end_dim=1)   if r_low is not None else None
            r_high  = r_high.flatten(start_dim=0, end_dim=1)  if r_high is not None else None
            i_low   = i_low.flatten(start_dim=0, end_dim=1)   if i_low is not None else None
            i_high  = i_high.flatten(start_dim=0, end_dim=1)  if i_high is not None else None
            i_delta = i_delta.flatten(start_dim=0, end_dim=1) if i_delta is not None else None
            
        # NOTE: Debugging
        epoch = self.current_epoch + 1
        if (self.debugger and epoch % self.debugger.every_n_epochs == 0
            and self.epoch_step < self.debugger.save_max_n):
            if self.trainer.is_global_zero:
                if self.phase is ModelState.DECOMNET:
                    _pred = (r_low, r_high, i_low, i_high)
                else:
                    _pred = (r_low, i_low, i_delta, yhat)
                self.debugger.run(
                    deepcopy(x), deepcopy(y), deepcopy(_pred),
                    self.debug_image_filepath
                )

        # NOTE: Loss
        loss = loss.mean() if loss is not None else None
        self.ckpt_log_scalar(f"checkpoint/loss/val_step", loss)
        # self.tb_log(f"{loss_tag}", loss, "step")
        
        # NOTE: Metrics
        if self.with_val_metrics:
            for i, metric in enumerate(self.val_metrics):
                if self.phase is ModelState.DECOMNET:
                    value = metric(r_low, r_high)
                else:
                    value = metric(yhat, y)
                self.ckpt_log_scalar(f"checkpoint/{metric.name}/val_step", value)
                # self.tb_log(f"{metric.name}/val_step", value, "step")
            
        self.epoch_step += 1
        return {"loss": loss}

    def validation_epoch_end(self, outputs: EpochOutput):
        # NOTE: Loss
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.ckpt_log_scalar(f"checkpoint/loss/val_epoch", loss)
        self.tb_log_scalar(f"loss/val_epoch", loss, "epoch")
        
        # NOTE: Metrics
        if self.with_val_metrics:
            for i, metric in enumerate(self.val_metrics):
                value = metric.compute()
                self.ckpt_log_scalar(f"checkpoint/{metric.name}/val_epoch", value)
                self.tb_log_scalar(f"{metric.name}/val_epoch", value, "epoch")
                metric.reset()
    
    def test_step(
        self, batch: Any, batch_idx: int, *args, **kwargs
    ) -> Optional[StepOutput]:
        """Test step.

        Args:
            batch (Any):
                Batch of inputs. It can be a tuple of (`x`, `y`, extra).
            batch_idx (int):
                Batch index.

        Returns:
            outputs (StepOutput, optional):
                - A single loss image.
                - A dictionary with the first key must be the `loss`.
                - `None`, training will skip to the next batch.
        """
        x, y, extra = batch[0], batch[1], batch[2:]
        yhat, loss  = self.forward_loss(x=x, y=y, *args, **kwargs)
        if self.phase is ModelState.DECOMNET:
            r_low, r_high, i_low, i_high = yhat
            return {
                "loss": loss, "x": x, "y": y, "r_low": r_low,
                "r_high": r_high, "i_low": i_low, "i_high": i_high
            }
        elif self.phase is ModelState.ENHANCENET:
            r_low, i_low, i_delta, yhat = yhat
            return {
                "loss": loss, "x": x, "y": y, "yhat": yhat,
                "r_low": r_low, "i_low": i_low, "i_delta": i_delta
            }
        else:
            r_low, i_low, i_delta, yhat = yhat
            return {
                "loss": loss, "x": x, "y": y, "yhat": yhat,
                "r_low": r_low, "i_low": i_low, "i_delta": i_delta
            }
    
    def test_step_end(
        self, outputs: Optional[StepOutput], *args, **kwargs
    ) -> Optional[StepOutput]:
        """Use this when testing with dp or ddp2 because `test_step` will
        operate on only part of the batch. However, this is still optional and
        only needed for things like softmax or NCE loss.

        Note:
            If you later switch to ddp or some other mode, this will still be
            called so that you don't have to change your code.
        """
        if not isinstance(outputs, dict):
            return None
        
        # NOTE: Gather results
        # For DDP strategy
        if self.trainer.num_processes > 1:
            outputs = self.all_gather(outputs)
        
        loss    = outputs["loss"]               # losses from each GPU
        x       = outputs["x"]                  # images from each GPU
        y       = outputs["y"]                  # ground-truths from each GPU
        yhat    = outputs.get("yhat",    None)  # predictions from each GPU
        r_low   = outputs.get("r_low",   None)
        r_high  = outputs.get("r_high",  None)
        i_low   = outputs.get("i_low",   None)
        i_high  = outputs.get("i_high",  None)
        i_delta = outputs.get("i_delta", None)
        
        # NOTE: Tensors
        if self.trainer.num_processes > 1:
            x       = x.flatten(start_dim=0, end_dim=1)
            y       = y.flatten(start_dim=0, end_dim=1)
            yhat    = yhat.flatten(start_dim=0, end_dim=1)    if yhat is not None else None
            r_low   = r_low.flatten(start_dim=0, end_dim=1)   if r_low is not None else None
            r_high  = r_high.flatten(start_dim=0, end_dim=1)  if r_high is not None else None
            i_low   = i_low.flatten(start_dim=0, end_dim=1)   if i_low is not None else None
            i_high  = i_high.flatten(start_dim=0, end_dim=1)  if i_high is not None else None
            i_delta = i_delta.flatten(start_dim=0, end_dim=1) if i_delta is not None else None
        
        # NOTE: Loss
        loss = loss.mean() if loss is not None else None
        self.ckpt_log_scalar(f"checkpoint/loss/test_step", loss)
        # self.tb_log(f"loss/test_step", loss, "step")
        
        # NOTE: Metrics
        if self.with_test_metrics:
            for i, metric in enumerate(self.test_metrics):
                if self.phase is ModelState.DECOMNET:
                    value = metric(r_low, r_high)
                else:
                    value = metric(yhat, y)
                self.ckpt_log_scalar(f"checkpoint/{metric.name}/test_step", value)
                # self.tb_log(f"{metric.name}/test_step", value, "step")
        
        self.epoch_step += 1
        return {"loss": loss}

    def test_epoch_end(self, outputs: EpochOutput):
        # NOTE: Loss
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.ckpt_log_scalar(f"checkpoint/loss/test_epoch", loss)
        self.tb_log_scalar(f"loss/test_epoch", loss, "epoch")

        # NOTE: Metrics
        if self.with_test_metrics:
            for i, metric in enumerate(self.test_metrics):
                value = metric.compute()
                self.ckpt_log_scalar(f"checkpoint/{metric.name}/test_epoch", value)
                self.tb_log_scalar(f"{metric.name}/test_epoch", value, "epoch")
                metric.reset()
    
    # MARK: Visualization
    
    def show_results(
        self,
        x            : Optional[Tensor] = None,
        y            : Optional[Tensor] = None,
        yhat         : Optional[Tensor] = None,
        filepath     : Optional[str]    = None,
        image_quality: int              = 95,
        verbose      : bool             = False,
        show_max_n   : int              = 8,
        wait_time    : float            = 0.01,
        *args, **kwargs
    ):
        """Draw `result` over input image.

        Args:
            x (Tensor, optional):
                Low-light images.
            y (Tensor, optional):
                Normal-light images.
            yhat (Tensor, optional):
                Predictions. When `model_state=DECOMNET`, it is (r_low, r_high,
                i_low, i_high). Otherwise, it is (r_low, i_low, i_delta,
                enhanced image).
            filepath (str, optional):
                File path to save the debug result.
            image_quality (int):
                Image quality to be saved. Default: `95`.
            verbose (bool):
                If `True` shows the results on the screen. Default: `False`.
            show_max_n (int):
                Maximum debugging items to be shown. Default: `8`.
            wait_time (float):
                Pause some times before showing the next image.
        """
        # NOTE: Prepare images
        yhat = self.prepare_results(yhat) if yhat is not None else None
        
        if self.phase is ModelState.DECOMNET:
            (r_low, r_high, i_low, i_high) = yhat
            results = {
                "low": x, "high": y, "r_low": r_low,
                "r_high": r_high, "i_low": i_low, "i_high": i_high
            }
        else:
            (r_low, i_low, i_delta, enhance) = yhat
            results = {
                "low": x, "r_low": r_low, "i_low": i_low,
                "i_delta": i_delta, "enhance": enhance, "high": y
            }
        
        filepath = self.debug_image_filepath if filepath is None else filepath
        save_cfg = {
            "filepath"  : filepath,
            "pil_kwargs": dict(quality=image_quality)
        }
        imshow_plt(
            images     = results,
            scale      = 2,
            save_cfg   = save_cfg,
            verbose    = verbose,
            show_max_n = show_max_n,
            wait_time  = wait_time
        )
    
    def prepare_results(self, yhat: Tensors, *args, **kwargs) -> Tensors:
        """Prepare results for visualization.

        Args:
            yhat (Images, Arrays):
                Predictions. When `model_state=DECOMNET`, it is (r_low, r_high,
                i_low, i_high). Otherwise, it is (r_low, i_low, i_delta,
                enhanced image).

        Returns:
            results (Tensors):
                Results for visualization.
        """
        if self.phase is ModelState.DECOMNET:
            (r_low, r_high, i_low, i_high) = yhat
            i_low_3  = torch.cat(tensors=(i_low,  i_low,  i_low),  dim=1)
            i_high_3 = torch.cat(tensors=(i_high, i_high, i_high), dim=1)
            # r_low    = to_image(r_low)
            # r_high   = to_image(r_high)
            # i_low_3  = to_image(i_low_3)
            # i_high_3 = to_image(i_high_3)
            return r_low, r_high, i_low_3, i_high_3

        elif self.phase in [
            ModelState.ENHANCENET, ModelState.RETINEXNET, ModelState.TESTING, ModelState.INFERENCE
        ]:
            (r_low, i_low, i_delta, enhance) = yhat
            n, b, _, _ = i_low.shape
            if b == 1:
                i_low_3 = torch.cat(tensors=(i_low, i_low, i_low), dim=1)
            else:
                i_low_3 = i_low
            i_delta_3 = torch.cat(tensors=(i_delta, i_delta, i_delta), dim=1)
            # r_low     = to_image(r_low)
            # i_low_3   = to_image(i_low_3)
            # i_delta_3 = to_image(i_delta_3)
            # enhance   = to_image(enhance)
            return r_low, i_low_3, i_delta_3, enhance
        
        else:
            return yhat
    

# MARK: - RetinexUNet

@MODELS.register(name="retinex_unet")
class RetinexUNet(RetinexNet):
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # Hyperparameters
        decomnet   = DecomNet(
            num_activation_layers=5, channels=64, kernel_size=3,
            use_batchnorm=True
        ),
        enhancenet = EnhanceNet(
            channels=64, kernel_size=3, fuse_features=False, use_batchnorm=True
        ),
        # BaseModel's args
        name       : Optional[str] 		  = "retinex_unet",
        num_classes: Optional[int]        = None,
        out_indexes: Indexes              = -1,
        model_state      : ModelState                = ModelState.TRAINING,
        metrics	   : Optional[list[dict]] = None,
        pretrained : Pretrained 		  = False,
        *args, **kwargs
    ):
        super().__init__(
            decomnet    = decomnet,
            enhancenet  = enhancenet,
            name        = name,
            num_classes = num_classes,
            out_indexes = out_indexes,
            model_state= model_state,
            metrics     = metrics,
            pretrained  = pretrained,
            *args, **kwargs
        )
