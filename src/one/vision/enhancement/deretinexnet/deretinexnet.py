#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DeRetinexNet.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from typing import Optional

import torch
from torch import Tensor

from one.core import ForwardOutput
from one.core import MODELS
from one.core import StepOutput
from one.core import Tensors
from one.imgproc import imshow_plt
from one.vision.enhancement.retinexnet import ModelState
from one.vision.enhancement.retinexnet import RetinexNet

__all__ = [
    "DeRetinexNet",
	"ModelState"
]


# MARK: - DeRetinexNet

@MODELS.register(name="deretinexnet")
class DeRetinexNet(RetinexNet):
    """Retinex-based models combine two submodels: DecomNet and EnhanceNet.
    RetinexNet is a multi-stage-training enhancer. We have to train the
    DecomNet first and then train the EnhanceNet.
  
    Notes:
        - When training the DecomNet: epoch=75, using Adam(lr=0.00001) gives
          best results.
    """
    
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
            r_high, i_high = self.decomnet(x)
            r_low,  i_low  = self.decomnet(y)
            loss = self.decom_loss(x, y, r_high, r_low, i_high, i_low)
            yhat = (r_high, r_low, i_high, i_low)
            return yhat, loss
        
        elif self.phase is ModelState.ENHANCENET:
            r_high, i_high = self.decomnet(x)
            i_delta        = self.enhancenet( r_high, i_high)
            i_delta_3 	   = torch.cat((i_delta, i_delta, i_delta), dim=1)
            yhat 		   = r_high * i_delta_3
            loss = self.enhance_loss(y, r_high, i_delta, i_delta_3, yhat)
            yhat = (r_high, i_high, i_delta, yhat)
            return yhat, loss
        
        else:
            r_high, i_high = self.decomnet(x)
            r_low,  i_low  = self.decomnet(y)
            i_delta        = self.enhancenet(r_high, i_high)
            i_delta_3 	   = torch.cat((i_delta, i_delta, i_delta), dim=1)
            yhat 		   = r_high * i_delta_3
            loss = self.retinex_loss(x, y, r_high, r_low, i_high, i_low, i_delta )
            yhat = (r_high, i_high, i_delta, yhat)
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
            yhat (Tensor):
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
        r_high, i_high = self.decomnet(x)
        i_delta        = self.enhancenet(r_high, i_high)
        i_delta_3 	   = torch.cat((i_delta, i_delta, i_delta), dim=1)
        yhat 		   = r_high * i_delta_3
        yhat           = (r_high, r_high, i_delta, yhat)
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
        y, x, extra = batch[0], batch[1], batch[2:]
        pred, loss  = self.forward_loss(x=x, y=y, *args, **kwargs)
        
        if self.phase is ModelState.DECOMNET:
            r_high, r_low, i_high, i_low = pred
            return {
                "loss": loss, "x": x, "y": y, "r_high": r_high,
                "r_low": r_low, "i_high": i_high, "i_low": i_low,
            }
        elif self.phase is ModelState.ENHANCENET:
            r_high, i_high, i_delta, pred = pred
            return {
                "loss": loss, "x": x, "y": y, "pred": pred,
                "r_high": r_high, "i_high": i_high, "i_delta": i_delta
            }
        else:
            r_high, i_high, i_delta, pred = pred
            return {
                "loss": loss, "x": x, "y": y, "pred": pred,
                "r_high": r_high, "i_high": i_high, "i_delta": i_delta
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
        y, x, extra = batch[0], batch[1], batch[2:]
        pred, loss = self.forward_loss(x=x, y=y, *args, **kwargs)

        if self.phase is ModelState.DECOMNET:
            r_high, r_low, i_high, i_low = pred
            return {
                "loss": loss,  "x": x, "y": y, "r_high": r_high, "r_low": r_low,
                "i_high": i_high, "i_low": i_low,
            }
        elif self.phase is ModelState.ENHANCENET:
            r_high, i_high, i_delta, pred = pred
            return {
                "loss": loss, "x": x, "y": y, "pred": pred,
                "r_high": r_high, "i_high": i_high, "i_delta": i_delta
            }
        else:
            r_high, i_high, i_delta, pred = pred
            return {
                "loss": loss, "x": x, "y": y, "pred": pred,
                "r_high": r_high, "i_high": i_high, "i_delta": i_delta
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
                    _pred = (r_high, i_high, i_delta, yhat)
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
        y, x, extra = batch[0], batch[1], batch[2:]
        yhat, loss  = self.forward_loss(x=x, y=y, *args, **kwargs)
        
        if self.phase is ModelState.DECOMNET:
            r_high, r_low, i_high, i_low = yhat
            return {
                "loss": loss, "x": x, "y": y, "r_high": r_high, "r_low": r_low,
                "i_high": i_high, "i_low": i_low,
            }
        elif self.phase is ModelState.ENHANCENET:
            r_high, i_high, i_delta, yhat = yhat
            return {
                "loss": loss, "x": x, "y": y, "yhat": yhat,
                "r_high": r_high, "i_high": i_high, "i_delta": i_delta
            }
        else:
            r_high, i_high, i_delta, yhat = yhat
            return {
                "loss": loss, "x": x, "y": y, "yhat": yhat,
                "r_high": r_high, "i_high": i_high, "i_delta": i_delta
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
            (r_high, r_low, i_high, i_low) = yhat
            results = {
                "high": x, "low": y, "r_high": r_high, "r_low": r_low,
                "i_high": i_high, "i_low": i_low,
            }
        else:
            (r_high, i_high, i_delta, enhance) = yhat
            results = {
                "high": x, "r_high": r_high, "i_high": i_high,
                "i_delta": i_delta, "enhance": enhance, "low": y
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
                Predictions. When `model_state=DECOMNET`, it is (r_high, r_low,
                i_high, i_low). Otherwise, it is (r_high, i_high, i_delta,
                enhanced image).

        Returns:
            results (Tensors):
                Results for visualization.
        """
        if self.phase is ModelState.DECOMNET:
            (r_high, r_low, i_high, i_low) = yhat
            i_high_3 = torch.cat(tensors=(i_high, i_high, i_high), dim=1)
            i_low_3  = torch.cat(tensors=(i_low, i_low, i_low), dim=1)
            # r_low    = to_image(r_low)
            # r_high   = to_image(r_high)
            # i_low_3  = to_image(i_low_3)
            # i_high_3 = to_image(i_high_3)
            return r_high, r_low, i_high_3, i_low_3
    
        elif self.phase in [
            ModelState.ENHANCENET, ModelState.RETINEXNET, ModelState.TESTING, ModelState.INFERENCE
        ]:
            (r_high, i_high, i_delta, enhance) = yhat
            n, b, _, _ = i_high.shape
            if b == 1:
                i_high_3 = torch.cat(tensors=(i_high, i_high, i_high), dim=1)
            else:
                i_high_3 = i_high
            i_delta_3 = torch.cat(tensors=(i_delta, i_delta, i_delta), dim=1)
            # r_low     = to_image(r_low)
            # i_low_3   = to_image(i_low_3)
            # i_delta_3 = to_image(i_delta_3)
            # enhance   = to_image(enhance)
            return r_high, i_high_3, i_delta_3, enhance
    
        else:
            return yhat
