#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements transformer-based attention modules."""

from __future__ import annotations

__all__ = [
	"ScaledDotProductAttention",
]

import torch
from torch import nn
from torch.nn import functional as F


class ScaledDotProductAttention(nn.Module):
	"""Scaled Dot-Product Attention
	
	Reference:
		`<https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance/blob/main/models/archs/transformer/Modules.py>`__
	"""
	
	def __init__(self, temperature, dropout: float = 0.0):
		super().__init__()
		self.temperature = temperature
		self.dropout     = nn.Dropout(dropout)
	
	def forward(
		self,
		q   : torch.Tensor,
		k   : torch.Tensor,
		v   : torch.Tensor,
		mask: torch.Tensor | None = None
	):
		attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
		if mask is not None:
			attn = attn.masked_fill(mask == 0, -1e9)
		attn   = self.dropout(F.softmax(attn, dim=-1))
		output = torch.matmul(attn, v)
		return output, attn
