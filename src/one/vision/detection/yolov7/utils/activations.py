#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Activation functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# SiLU https://arxiv.org/pdf/1606.08415.pdf ----------------------------------------------------------------------------
class SiLU(nn.Module):  # export-friendly version of nn.SiLU()
	@staticmethod
	def forward(x: Tensor) -> Tensor:
		return x * torch.sigmoid(x)


class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
	@staticmethod
	def forward(x: Tensor) -> Tensor:
		# return x * F.hardsigmoid(x)  # for torchscript and CoreML
		return x * F.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX


class MemoryEfficientSwish(nn.Module):
	class F(torch.autograd.Function):
		@staticmethod
		def forward(ctx, x: Tensor) -> Tensor:
			ctx.save_for_backward(x)
			return x * torch.sigmoid(x)
		
		@staticmethod
		def backward(ctx, grad_output: Tensor) -> Tensor:
			x  = ctx.saved_tensors[0]
			sx = torch.sigmoid(x)
			return grad_output * (sx * (1 + x * (1 - sx)))
	
	def forward(self, x: Tensor) -> Tensor:
		return self.F.apply(x)


# Mish https://github.com/digantamisra98/Mish --------------------------------------------------------------------------
class Mish(nn.Module):
	@staticmethod
	def forward(x: Tensor) -> Tensor:
		return x * F.softplus(x).tanh()


class MemoryEfficientMish(nn.Module):
	class F(torch.autograd.Function):
		@staticmethod
		def forward(ctx, x: Tensor) -> Tensor:
			ctx.save_for_backward(x)
			return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))
		
		@staticmethod
		def backward(ctx, grad_output: Tensor) -> Tensor:
			x  = ctx.saved_tensors[0]
			sx = torch.sigmoid(x)
			fx = F.softplus(x).tanh()
			return grad_output * (fx + x * sx * (1 - fx * fx))
	
	def forward(self, x: Tensor) -> Tensor:
		return self.F.apply(x)


# FReLU https://arxiv.org/abs/2007.11824 -------------------------------------------------------------------------------
class FReLU(nn.Module):
	def __init__(self, c1: int, k: int = 3):  # ch_in, kernel
		super().__init__()
		self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
		self.bn   = nn.BatchNorm2d(c1)
	
	def forward(self, x: Tensor) -> Tensor:
		return torch.max(x, self.bn(self.conv(x)))
