#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module provides utility functions for working with :mod:`mon.nn.modules.cam`
module.
"""

from __future__ import annotations

__all__ = [
	"fasterrcnn_reshape_transform",
	"find_layer_predicate_recursive",
	"find_layer_types_recursive",
	"get_2d_projection",
	"replace_all_layer_type_recursive",
	"replace_layer_recursive",
	"swint_reshape_transform",
	"vit_reshape_transform",
]

from typing import Sequence

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import zoom
from torch import nn
from torchvision.transforms import Compose, Normalize, ToTensor


# region Find Layers

def replace_layer_recursive(model: nn.Module, old_layer, new_layer):
	for name, layer in model._modules.items():
		if layer == old_layer:
			model._modules[name] = new_layer
			return True
		elif replace_layer_recursive(layer, old_layer, new_layer):
			return True
	return False


def replace_all_layer_type_recursive(model: nn.Module, old_layer_type, new_layer):
	for name, layer in model._modules.items():
		if isinstance(layer, old_layer_type):
			model._modules[name] = new_layer
		replace_all_layer_type_recursive(layer, old_layer_type, new_layer)


def find_layer_types_recursive(model: nn.Module, layer_types):
	def predicate(layer):
		return type(layer) in layer_types
	return find_layer_predicate_recursive(model, predicate)


def find_layer_predicate_recursive(model: nn.Module, predicate):
	result = []
	for name, layer in model._modules.items():
		if predicate(layer):
			result.append(layer)
		result.extend(find_layer_predicate_recursive(layer, predicate))
	return result

# endregion


# region Image

def preprocess_image(
	image: np.ndarray,
	mean : Sequence = [0.5, 0.5, 0.5],
	std  : Sequence = [0.5, 0.5, 0.5]
) -> torch.Tensor:
	preprocessing = Compose([
		ToTensor(),
		Normalize(mean=mean, std=std),
	])
	return preprocessing(image.copy()).unsqueeze(0)


def deprocess_image(image: np.ndarray) -> np.ndarray:
	"""See https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
	image = image - np.mean(image)
	image = image / (np.std(image) + 1e-5)
	image = image * 0.1
	image = image + 0.5
	image = np.clip(image, 0, 1)
	return np.uint8(image * 255)


def show_cam_on_image(
	image       : np.ndarray,
	mask        : np.ndarray,
	use_rgb     : bool  = False,
	colormap    : int   = cv2.COLORMAP_JET,
	image_weight: float = 0.5
) -> np.ndarray:
	"""This function overlays the cam mask on the image as a heatmap. By default
	the heatmap is in BGR format.
	
	Args:
		image: The base image in RGB or BGR format.
		mask: The cam mask.
		use_rgb: Whether to use an RGB or BGR heatmap, this should be set to
			``True`` if :param:`image` is in RGB format.
		colormap: The OpenCV colormap to be used.
		image_weight: The final result is: :math:`image_weight * img + (1-image_weight) * mask`.
	
	Returns:
		The default image with the cam overlay.
	"""
	heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
	if use_rgb:
		heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
	heatmap = np.float32(heatmap) / 255
	
	if np.max(image) > 1:
		raise Exception("The input image should :class:`np.float32` in the range :math:`[0, 1]`")
	
	if image_weight < 0 or image_weight > 1:
		raise Exception(f":param:`image_weight` should be in the range :math:`[0, 1]`, but got: {image_weight}")
	
	cam = (1 - image_weight) * heatmap + image_weight * image
	cam = cam / np.max(cam)
	return np.uint8(255 * cam)


def create_labels_legend(
	concept_scores: np.ndarray,
	labels        : dict[int, str],
	top_k         : int = 2
) -> list[str]:
	concept_categories  = np.argsort(concept_scores, axis=1)[:, ::-1][:, :top_k]
	concept_labels_topk = []
	for concept_index in range(concept_categories.shape[0]):
		categories     = concept_categories[concept_index, :]
		concept_labels = []
		for category in categories:
			score = concept_scores[concept_index, category]
			label = f"{','.join(labels[category].split(',')[:3])}:{score:.2f}"
			concept_labels.append(label)
		concept_labels_topk.append("\n".join(concept_labels))
	return concept_labels_topk


def show_factorization_on_image(
	image         : np.ndarray,
	explanations  : np.ndarray,
	colors        : list[np.ndarray] = None,
	image_weight  : float = 0.5,
	concept_labels: list = None
) -> np.ndarray:
	"""Color code the different component heatmaps on top of the image. Every
	component color code will be magnified according to the heatmap intensity
	(by modifying the V channel in the HSV color space), and optionally create
	a legend that shows the labels.
	
	Since different factorization component heatmaps can overlap in principle,
	we need a strategy to decide how to deal with the overlaps. This keeps the
	component that has a higher value in it's heatmap.
	
	Args:
		image: The base image RGB format.
		explanations: A tensor of shape :math:`[num_componetns, height, width]`,
			with the component visualizations.
		colors: List of R, G, B colors to be used for the components. If ``None``,
			will use the ``gist_rainbow`` cmap as a default.
		image_weight: The final result is :math:`image_weight * img + (1-image_weight) * visualization`.
		concept_labels: A list of strings for every component. If this is passed,
			a legend that shows the labels and their colors will be added to the
			image.
	
	Returns:
		The visualized image.
	"""
	n_components = explanations.shape[0]
	if colors is None:
		# taken from https://github.com/edocollins/DFF/blob/master/utils.py
		_cmap  = plt.cm.get_cmap("gist_rainbow")
		colors = [np.array(_cmap(i)) for i in np.arange(0, 1, 1.0 / n_components)]
	concept_per_pixel = explanations.argmax(axis=0)
	masks = []
	for i in range(n_components):
		mask = np.zeros(shape=(image.shape[0], image.shape[1], 3))
		mask[:, :, :] = colors[i][:3]
		explanation   = explanations[i]
		explanation[concept_per_pixel != i] = 0
		mask = np.uint8(mask * 255)
		mask = cv2.cvtColor(mask, cv2.COLOR_RGB2HSV)
		mask[:, :, 2] = np.uint8(255 * explanation)
		mask = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB)
		mask = np.float32(mask) / 255
		masks.append(mask)
	
	mask   = np.sum(np.float32(masks), axis=0)
	result = image * image_weight + mask * (1 - image_weight)
	result = np.uint8(result * 255)
	
	if concept_labels is not None:
		px    = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
		fig   = plt.figure(figsize=(result.shape[1] * px, result.shape[0] * px))
		plt.rcParams["legend.fontsize"] = int(14 * result.shape[0] / 256 / max(1, n_components / 6))
		lw    = 5 * result.shape[0] / 256
		lines = [Line2D([0], [0], color=colors[i], lw=lw) for i in range(n_components)]
		plt.legend(
			lines,
			concept_labels,
			mode     = "expand",
			fancybox = True,
			shadow   = True,
		)
		plt.tight_layout(pad=0, w_pad=0, h_pad=0)
		plt.axis("off")
		fig.canvas.draw()
		data   = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
		plt.close(fig=fig)
		data   = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
		data   = cv2.resize(data, (result.shape[1], result.shape[0]))
		result = np.hstack((result, data))
	return result


def scale_cam_image(cam, target_size=None):
	result = []
	for img in cam:
		img = img - np.min(img)
		img = img / (1e-7 + np.max(img))
		if target_size is not None:
			if len(img.shape) > 2:
				img = zoom(np.float32(img), [
					(t_s / i_s) for i_s, t_s in zip(img.shape, target_size[::-1])])
			else:
				img = cv2.resize(np.float32(img), target_size)
		
		result.append(img)
	result = np.float32(result)
	
	return result


def scale_across_batch_and_channels(tensor, target_size):
	batch_size, channel_size = tensor.shape[:2]
	reshaped_tensor = tensor.reshape(
		batch_size * channel_size, *tensor.shape[2:])
	result = scale_cam_image(reshaped_tensor, target_size)
	result = result.reshape(
		batch_size,
		channel_size,
		target_size[1],
		target_size[0])
	return result

# endregion


# region Reshape Transforms

def fasterrcnn_reshape_transform(x: dict) -> torch.Tensor:
	target_size = x["pool"].size()[-2:]
	activations = []
	for k, v in x.items():
		activations.append(torch.nn.functional.interpolate(torch.abs(v), target_size, mode="bilinear"))
	activations = torch.cat(activations, axis=1)
	return activations


def swint_reshape_transform(x: torch.Tensor, height: int = 7, width: int = 7) -> torch.Tensor:
	result = x.reshape(x.size(0), height, width, x.size(2))
	# Bring the channels to the first dimension, like in CNNs.
	result = result.transpose(2, 3).transpose(1, 2)
	return result


def vit_reshape_transform(x: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
	result = x[:, 1:, :].reshape(x.size(0), height, width, x.size(2))
	# Bring the channels to the first dimension, like in CNNs.
	result = result.transpose(2, 3).transpose(1, 2)
	return result

# endregion


# region SVD on Activations

def get_2d_projection(activation_batch: list) -> np.ndarray:
	# TBD: use pytorch batch svd implementation
	activation_batch[np.isnan(activation_batch)] = 0
	projections = []
	for activations in activation_batch:
		reshaped_activations = (activations).reshape(activations.shape[0], -1).transpose()
		# Centering before the SVD seems to be important here, otherwise the image returned is negative.
		reshaped_activations = reshaped_activations - reshaped_activations.mean(axis=0)
		U, S, VT   = np.linalg.svd(reshaped_activations, full_matrices=True)
		projection = reshaped_activations @ VT[0, :]
		projection = projection.reshape(activations.shape[1:])
		projections.append(projection)
	return np.float32(projections)

# endregion
