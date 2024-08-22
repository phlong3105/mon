#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all vision models."""

from __future__ import annotations

__all__ = [
    "VisionModel",
]

from abc import ABC
from copy import deepcopy

import torch
from fvcore.nn import parameter_count

from mon import core, nn
from mon.core import _size_2_t
from mon.globals import ZOO_DIR

console = core.console


# region Model

class VisionModel(nn.Model, ABC):
    """The base class for all vision models, i.e., image or video as the primary
    input.
    
    """
    
    zoo_dir: core.Path  = ZOO_DIR / "vision"
    
    # region Initialize Model
    
    def compute_efficiency_score(
        self,
        image_size: _size_2_t = 512,
        channels  : int       = 3,
        runs      : int       = 100,
        verbose   : bool      = False,
    ) -> tuple[float, float, float]:
        """Compute the efficiency score of the model, including FLOPs, number
        of parameters, and runtime.
        """
        # Define input tensor
        h, w      = core.parse_hw(image_size)
        datapoint = {"image": torch.rand(1, channels, h, w).to(self.device)}
        
        # Get FLOPs and Params
        flops, params = core.custom_profile(deepcopy(self), inputs=datapoint, verbose=verbose)
        # flops         = FlopCountAnalysis(self, datapoint).total() if flops == 0 else flops
        params        = self.params                if hasattr(self, "params") and params == 0 else params
        params        = parameter_count(self)      if hasattr(self, "params")  else params
        params        = sum(list(params.values())) if isinstance(params, dict) else params
        
        # Get time
        timer = core.Timer()
        for i in range(runs):
            timer.tick()
            _ = self(datapoint)
            timer.tock()
        avg_time = timer.avg_time
        
        # Print
        if verbose:
            console.log(f"FLOPs (G) : {flops:.4f}")
            console.log(f"Params (M): {params:.4f}")
            console.log(f"Time (s)  : {avg_time:.4f}")
        
        return flops, params, avg_time
        
    # endregion
    
    # region Predicting
    
    @torch.no_grad()
    def infer(
        self,
        datapoint: dict,
        imgsz    : _size_2_t = 512,
        resize   : bool      = False,
        *args, **kwargs
    ) -> dict:
        """Infer the model on a single datapoint. This method is different from
        :obj:`forward()` in term that you may want to perform additional
        pre-processing or post-processing steps.
        
        Notes:
            If you want to perform specific pre-processing or post-processing
            steps, you should override this method.
        
        Args:
            datapoint: A :obj:`dict` containing the attributes of a datapoint.
            imgsz: The input size. Default: ``512``.
            resize: Resize the input image to the model's input size. Default: ``False``.
        """
        # Pre-processing
        self.assert_datapoint(datapoint)
        image  = datapoint.get("image")
        h0, w0 = core.get_image_size(image)
        for k, v in datapoint.items():
            if core.is_image(v):
                if resize:
                    datapoint[k] = core.resize(v, imgsz)
                else:
                    datapoint[k] = core.resize_divisible(v, 32)
        for k, v in datapoint.items():
            if isinstance(v, torch.Tensor):
                datapoint[k] = v.to(self.device)
                
        # Forward
        timer   = core.Timer()
        timer.tick()
        outputs = self.forward(datapoint, *args, **kwargs)
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
