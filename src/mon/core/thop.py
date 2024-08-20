#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module extend :mod:`thop.profile` to work with :mod:`mon.nn`."""

from __future__ import annotations

# noinspection PyUnresolvedReferences
from thop import *
from thop.profile import register_hooks
from thop.utils import prRed
from thop.vision.basic_hooks import *

default_dtype  = torch.float64


def profile_mon_model(
	model: nn.Module,
	inputs,
	custom_ops     = None,
	verbose        = True,
	ret_layer_info = False,
	report_missing = False,
):
	"""Extend :func:`thop.profile` to work with :mod:`mon.nn.model.Model`."""
	handler_collection = {}
	types_collection   = set()
	if custom_ops is None:
		custom_ops = {}
	if report_missing:
		# overwrite `verbose` option when enable report_missing
		verbose = True
	
	def add_hooks(m: nn.Module):
		"""Registers hooks to a neural network module to track total operations and parameters."""
		m.register_buffer("total_ops",    torch.zeros(1, dtype=torch.float64))
		m.register_buffer("total_params", torch.zeros(1, dtype=torch.float64))
		
		# for p in m.parameters():
		#     m.total_params += torch.DoubleTensor([p.numel()])
		
		m_type = type(m)
		
		fn = None
		if m_type in custom_ops:
			# if defined both op maps, use custom_ops to overwrite.
			fn = custom_ops[m_type]
			if m_type not in types_collection and verbose:
				print(f"[INFO] Customize rule {fn.__qualname__}() {m_type}.")
		elif m_type in register_hooks:
			fn = register_hooks[m_type]
			if m_type not in types_collection and verbose:
				print(f"[INFO] Register {fn.__qualname__}() for {m_type}.")
		else:
			if m_type not in types_collection and report_missing:
				prRed(f"[WARN] Cannot find rule for {m_type}. Treat it as zero Macs and zero Params.")
		
		if fn is not None:
			handler_collection[m] = (
				m.register_forward_hook(fn),
				m.register_forward_hook(count_parameters),
			)
		types_collection.add(m_type)
	
	prev_training_status = model.training
	
	model.eval()
	model.apply(add_hooks)
	
	with torch.no_grad():
		model(datapoint=inputs)
	
	def dfs_count(module: nn.Module, prefix="\t") -> (int, int):
		"""Recursively counts the total operations and parameters of the given PyTorch module and its submodules."""
		total_ops    = 0
		total_params = 0
		ret_dict     = {}
		for n, m in module.named_children():
			# if not hasattr(m, "total_ops") and not hasattr(m, "total_params"):  # and len(list(m.children())) > 0:
			#     m_ops, m_params = dfs_count(m, prefix=prefix + "\t")
			# else:
			#     m_ops, m_params = m.total_ops, m.total_params
			next_dict = {}
			if m in handler_collection and not isinstance(m, (nn.Sequential, nn.ModuleList)):
				m_ops, m_params = m.total_ops.item(), m.total_params.item()
			else:
				m_ops, m_params, next_dict = dfs_count(m, prefix=prefix + "\t")
			ret_dict[n]   = (m_ops, m_params, next_dict)
			total_ops    += m_ops
			total_params += m_params
		# print(prefix, module._get_name(), (total_ops, total_params))
		return total_ops, total_params, ret_dict
	
	total_ops, total_params, ret_dict = dfs_count(model)
	
	# reset model to original status
	model.train(prev_training_status)
	for m, (op_handler, params_handler) in handler_collection.items():
		op_handler.remove()
		params_handler.remove()
		m._buffers.pop("total_ops")
		m._buffers.pop("total_params")
	
	if ret_layer_info:
		return total_ops, total_params, ret_dict
	return total_ops, total_params
