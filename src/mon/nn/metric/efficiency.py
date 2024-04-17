#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements efficiency score metrics."""

from __future__ import annotations

__all__ = [
	"calculate_efficiency_score",
]

import time
from copy import deepcopy

import thop
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count
from torch import nn

from mon import core
from mon.core import _size_2_t

console = core.console


# region Efficiency Metric

def calculate_efficiency_score(
	model     : nn.Module,
	image_size: _size_2_t = 512,
	channels  : int       = 3,
	runs      : int       = 100,
	use_cuda  : bool      = True,
	verbose   : bool      = False,
):
	# Define input tensor
	h, w  = core.parse_hw(image_size)
	input = torch.rand(1, channels, h, w)
	
	# Deploy to cuda
	if use_cuda:
		input = input.cuda()
		model = model.cuda()
		# device = torch.device("cuda:0")
		# input  = input.to(device)
		# model  = model.to(device)
	
	# Get FLOPs and Params
	flops, params = thop.profile(deepcopy(model), inputs=(input, ), verbose=verbose)
	flops         = FlopCountAnalysis(model, input).total() if flops == 0 else flops
	params        = model.params if hasattr(model, "params") and params == 0 else params
	params        = parameter_count(model) if hasattr(model, "params") else params
	params        = sum(list(params.values())) if isinstance(params, dict) else params
	g_flops       = flops * 1e-9
	m_params      = int(params) * 1e-6
	
	# Get time
	start_time = time.time()
	for i in range(runs):
		_ = model(input)
	runtime    = time.time() - start_time
	avg_time   = runtime / runs
	
	# Print
	if verbose:
		console.log(f"FLOPs (G) : {flops:.4f}")
		console.log(f"Params (M): {params:.4f}")
		console.log(f"Time (s)  : {avg_time:.4f}")
	
	return flops, params, avg_time

# endregion
