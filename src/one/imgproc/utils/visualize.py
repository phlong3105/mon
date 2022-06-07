#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Visualize the image onto the screen.
"""

from __future__ import annotations

import math
from typing import Any
from typing import Optional
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from one.core import Arrays
from one.core import make_image_grid
from one.core import to_4d_array_list
from one.core import to_channel_last
from one.core import to_image
from one.core import to_pil_image

plt.ion()
plt.switch_backend("qt5agg")

__all__ = [
	"imshow_cls_plt",
	"imshow_plt",
	"move_figure",
	"show_images"
]


# MARK: - Visualize Images

def show_images(
	images     : Any,
	nrow       : int   = 8,
	denormalize: bool  = False,
	figure_num : int   = 0,
	wait_time  : float = 0.001
):
	"""Visualize images for debugging purpose.

	Args:
		images (Any):
			Images to be shown.
		nrow (int):
			Number of images displayed in each row of the grid. Ffinal grid
			size is `[B / nrow, nrow]`.
		denormalize (bool):
			Should unnormalize the images? Default: `False`.
		figure_num (int):
			matplotlib figure id.
		wait_time (float):
			Wait some time (in seconds) to display the figure then reset.
	"""
	# NOTE: Make an image grid
	cat_image = make_image_grid(images, nrow)
	cat_image = to_image(cat_image, denormalize=True)
	"""
	cat_image = cat_image.numpy() if torch.is_tensor(cat_image) else cat_image
	cat_image = to_channel_last(cat_image)
	if denormalize:
		from onecore.vision import denormalize_naive
		cat_image = denormalize_naive(cat_image)
	"""
	# NOTE: Convert to PIL Image
	cat_image = to_pil_image(image=cat_image)
	
	# NOTE: Clear old figure and visualize current one
	fig = plt.figure(num=figure_num)
	fig.clf()
	plt.imshow(cat_image, interpolation="bicubic")
	plt.pause(wait_time)


def imshow_plt(
	images    : Any,
	labels    : Optional[Sequence[str]] = None,
	scale     : int                     = 1,
	save_cfg  : Optional[dict]          = None,
	verbose	  : bool					= True,
	show_max_n: int					    = 8,
	wait_time : float                   = 0.01,
	figure_num: int                     = 0,
):
	"""Visualize images as a grid using matplotlib.
	
	Args:
		images (Any):
			A sequence of images. Each element is of shape [B, C, H, W].
		labels (Sequence[str], optional):
			Sequence of images' labels string.
		scale (int):
			Scale the size of matplotlib figure. `1` means (default size x 1).
		save_cfg (dict, optional):
			Save figure config.
		verbose (bool):
			If `True`, verbose the debug image, else skip. Default: `True`.
		show_max_n (int):
			Show at max n images. Default: `8`.
		wait_time (float):
			Wait some time (in seconds) to display the figure then reset.
		figure_num (int):
			Fmatplotlib figure id. Default: `0`.
			
	Examples:
		>>> import cv2
		>>> import numpy as np
		>>> bgr     = cv2.imread("MLKit/tests/lenna.png")
		>>> rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
		>>>
		>>> rgbs    = np.array([rgb, rgb, rgb])
		>>> bgrs    = np.array([bgr, bgr, bgr])
		>>>
		>>> outputs = [rgbs, bgrs]
		>>> labels  = ["RGB", "BGR"]
		>>> imshow_plt(images=outputs, labels=labels, figure_num=0, scale=1)
		>>>
		>>> outputs = { "RGB": rgbs, "BGR": bgrs }
		>>> imshow_plt(images=outputs, figure_num=1, scale=2)
	"""
	if not verbose and save_cfg is None:
		return
	
	# NOTE: Prepare images
	images_ = to_4d_array_list(images)  # List of 4D-array
	images_ = [to_channel_last(i)	for i in images_]
	from one.core import denormalize_naive
	images_ = [denormalize_naive(i) for i in images_]
	images_ = [i[: show_max_n]    	for i in images_]
	
	# NOTE: Prepare labels
	if isinstance(images, dict):
		labels = images.keys() if labels is None else labels
	if labels is None:
		labels = ["" for _ in range(len(images))]
	if len(labels) != len(images):
		raise ValueError(f"`labels` and `images` must have the same length."
		                 f" But got: {len(labels)} != {len(images)}.")
	
	# NOTE: Create an image grid
	ncols           = len(images_)
	nrows, h, w, c  = images_[0].shape
	nrows          += 0 if nrows > 1 else 1
	
	fig, axes = plt.subplots(
		nrows   = nrows,
		ncols   = ncols,
		figsize = [ncols * scale, nrows * scale],
		num     = figure_num,
		clear   = True,
	)
	move_figure(x=0, y=0)
	[ax.set_title(l) for ax, l in zip(axes[0], labels)]
	for i, imgs in enumerate(images_):
		# axes[0, i].set_title(labels[i])
		for j, img in enumerate(imgs):
			axes[j, i].imshow(img, aspect="auto")
			axes[j, i].set_yticklabels([])
			axes[j, i].set_xticklabels([])
	plt.tight_layout()
	plt.subplots_adjust(wspace=0.0, hspace=0.0)
	plt.show()
	
	# NOTE: Save figure
	if save_cfg:
		filepath = save_cfg.pop("filepath")
		plt.savefig(filepath, **save_cfg)
		
	# NOTE: Show
	if verbose:
		plt.show()
		plt.pause(wait_time)
		

# MARK: - Visualize Image Classification
def imshow_cls_plt(
	images 	    : Any,
	preds	    : Optional[Arrays]		  = None,
	targets	    : Optional[Arrays]		  = None,
	labels	    : Optional[Sequence[str]] = None,
	class_labels: Optional["ClassLabels"] = None,
	top_k	    : int 				      = 5,
	scale       : int                     = 1,
	save_cfg    : Optional[dict]          = None,
	verbose	    : bool					  = True,
	show_max_n  : int					  = 8,
	wait_time   : float                   = 0.01,
	figure_num  : int                     = 0,
):
	"""Visualize classification results on images using matplotlib.
	
	Args:
		images (Any):
			A sequence of images. Each element is of shape [B, C, H, W].
		preds (Arrays, optional):
			A sequence of predicted classes probabilities. Default: `None`.
		targets (Arrays, optional):
			A sequence of ground-truths. Default: `None`.
		labels (Sequence[str], optional):
			Sequence of images' labels string. Default: `None`.
		class_labels (ClassLabels, optional):
			`ClassLabels` objects that contains all class labels in the
			datasets. Default: `None`.
		top_k (int):
			Show only the top k classes' probabilities. Default: `5`.
		scale (int):
			Scale the size of matplotlib figure. `1` means (default size x 1).
		save_cfg (dict, optional):
			Save figure config.
		verbose (bool):
			If `True`, verbose the debug image, else skip. Default: `True`.
		show_max_n (int):
			Show at max n images. Default: `8`.
		wait_time (float):
			Wait some time (in seconds) to display the figure then reset.
		figure_num (int):
			Matplotlib figure id. Default: `0`.
	"""
	if not verbose and save_cfg is None:
		return
	
	# NOTE: Prepare images
	images_ = to_image(images, denormalize=True)  # 4D-array
	images_ = to_channel_last(images_)
	images_ = images_[:show_max_n]
	
	# NOTE: Prepare preds and targets
	pred_scores   = np.sort(preds, axis=1)[:show_max_n, -top_k:][:, ::]
	pred_labels   = preds.argsort(axis=1)[:show_max_n, -top_k:][:, ::]
	pred_1_labels = preds.argsort(axis=1)[:show_max_n, -top_k:][:, -1]
	if class_labels:
		pred_labels   = [[class_labels.get_name(value=l) for l in pred]
						 for pred in pred_labels]
		pred_1_labels = [class_labels.get_name(value=l) for l in pred_1_labels]
		targets 	  = [class_labels.get_name(value=l) for l in targets]
	
	# NOTE: Prepare labels
	if labels is None:
		labels = [f"" for i in range(images_.shape[0])]
	labels = [f"{l} \n gt={gt}"
	          for (l, pred, gt) in zip(labels, pred_1_labels, targets)]
	colors = ["darkgreen" if pred == gt else "red"
	          for (pred, gt) in zip(pred_1_labels, targets)]
	if len(labels) != images_.shape[0]:
		raise ValueError(f"Length of `labels` and `images` batch size must be the same."
		                 f" But got: {len(labels)} != {images_.shape[0]}.")
	
	# NOTE: Create an image grid
	n, h, w, c = images_.shape
	ncols 	   = 8
	nrows	   = int(math.ceil(n / ncols))
	y_pos 	   = np.arange(top_k)
	fig = plt.figure(
		constrained_layout=True, num=figure_num,
		figsize=[(ncols * 1.0) * scale, (nrows * 1.5) * scale]
	)
	subfigs 	 = fig.subfigures(nrows=nrows, ncols=ncols)
	subfigs_flat = subfigs.flat
	move_figure(x=0, y=0)
	
	for idx, (img, label, color, scores, labels) in \
		enumerate(zip(images_, labels, colors, pred_scores, pred_labels)):
		subfig = subfigs_flat[idx]
		subfig.suptitle(label, x=0.5, y=1.0, color=color)
		axes = subfig.subplots(2, 1)
		for ax in axes.flat:
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.tick_params(axis="x", direction="in", pad=-10)
			ax.tick_params(axis="y", direction="in", pad=-10)
		# Image
		axes[0].imshow(img, aspect="auto")
		# Classlabels
		pps = axes[1].barh(y_pos, scores, align="center", color="deepskyblue")
		axes[1].set_xlim(left=0.0)
		axes[1].set_yticks(y_pos)
		axes[1].set_yticklabels(labels, horizontalalignment="left")
		# Scores
		max_width = max([rect.get_width() for rect in pps])
		for i, rect in enumerate(pps):
			if scores[i] > 0:
				axes[1].text(
					rect.get_x() + max_width, rect.get_y(),
					"{:.2f}".format(scores[i]), ha="right", va="bottom",
					rotation=0
				)
	# plt.tight_layout()
	plt.subplots_adjust(wspace=0.0, hspace=0.0)
	
	# NOTE: Save figure
	if save_cfg:
		filepath = save_cfg.pop("filepath")
		plt.savefig(filepath, **save_cfg)

	# NOTE: Show
	if verbose:
		plt.show()
		plt.pause(wait_time)


# MARK: - Figure Manager

def move_figure(x: int, y: int):
	"""Move figure's upper left corner to pixel (x, y)."""
	mngr    = plt.get_current_fig_manager()
	fig     = plt.gcf()
	backend = matplotlib.get_backend()
	
	if backend == "TkAgg":
		mngr.window.wm_geometry("+%d+%d" % (x, y))
	elif backend == "WXAgg":
		mngr.window.SetPosition((x, y))
	else:  # This works for QT and GTK. You can also use window.setGeometry
		mngr.window.move(x, y)
