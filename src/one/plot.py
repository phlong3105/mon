# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plotting functions.
"""

from __future__ import annotations

import inspect
import math
import sys
from typing import Any

import matplotlib
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torch import Tensor

from one.core import Arrays
from one.core import assert_same_length
from one.core import ClassLabel
from one.core import Strs
from one.core import Tensors
from one.core import to_3d_tensor_list
from one.core import to_list
from one.vision.acquisition import to_image
from one.vision.transformation import denormalize_naive

# mpl.use("wxAgg")

plt.ion()
plt.switch_backend("qt5agg")
plt.rcParams["savefig.bbox"] = "tight"


# H1: - Drawing ----------------------------------------------------------------


# H1: - Positioning ------------------------------------------------------------

def get_grid_size(n: int, nrow: int | None = 8) -> tuple[int, int]:
    """
    It takes a number of items and a maximum number of items per row, and returns
    the number of rows and columns needed to display the items in a grid.
    
    Args:
        n (int): The number of items to be plotted.
        nrow (int | None): The maximum number of items to display in a row.
            The final grid size is (n / nrow, nrow). If None, then the number
            of items in a row will be the same as the number of items in the
            list. Defaults to 8.
    
    Returns:
        A tuple of the number of rows and columns.
    """
    if isinstance(nrow, int) and nrow > 0:
        ncols = nrow
    else:
        ncols = n
    nrows = math.ceil(n / ncols)
    return nrows, ncols
    

def make_image_grid(
    images     : Tensors,
    nrow       : int | None = 8,
    denormalize: bool       = False,
) -> Tensor:
    """
    Make a grid of images.

    Args:
        images (Tensor):
        
        nrow (int | None): The maximum number of items to display in a row.
            The final grid size is (n / nrow, nrow). If None, then the number
            of items in a row will be the same as the number of items in the
            list. Defaults to 8.
        denormalize (bool): If True, the image will be denormalized to
            [0, 255]. Defaults to True.
            
    Returns:
        Image grid tensor.
    """
    nrow   = nrow if isinstance(nrow, int) and nrow > 0 else 8
    images = to_3d_tensor_list(images)
    images = denormalize_naive(images) if denormalize else images
    return torchvision.utils.make_grid(tensor=images, nrow=nrow)


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


# H1: - Visualize --------------------------------------------------------------

def imshow(
    winname    : str,
    image      : Tensor,
    label      : Strs | None = None,
    denormalize: bool        = False,
    scale      : int         = 1,
    save_cfg   : dict | None = None,
    max_n      : int  | None = None,
    nrow       : int  | None = None,
    wait_time  : float       = 0.01,
    figure_num : int         = 0,
):
    """
    Show an image tensor in a window using matplotlib.
    
    Args:
        winname (str): The name of the window to display the image in.
        image (Tensor): The images to be displayed. Can be a 3D tensor
            [C, H, W] or 4D tensor of shape [B, C, H, W].
        label (Strs | None): Sequence of images' labels string. Defaults to
            None.
        denormalize (bool): If True, the image will be denormalized to
            [0, 255]. Defaults to True.
        scale (int): Scale the size of matplotlib figure. Defaults to 1 means
            size x 1.
        save_cfg (dict | None): Save figure config. Defaults to None.
        max_n (int | None): Show max n images if `image` has a batch size
            of more than `max_n` images. Defaults to None means show all.
        nrow (int | None): The maximum number of items to display in a row.
            The final grid size is (n / nrow, nrow). If None, then the number
            of items in a row will be the same as the number of items in the
            list. Defaults to 8.
        wait_time (float): Wait some time (in seconds) to display the figure
            then reset. Defaults to 0.
        figure_num (int): Matplotlib figure id. Defaults to 0.
    """
    # Prepare image and label
    image = to_3d_tensor_list(image)
    image = image[: max_n] if isinstance(max_n, int) else image
    image = [to_image(image=i, keep_dims=False, denormalize=denormalize)
             for i in image]
    
    if label is not None:
        label = to_list(label)
        label = label[: max_n] if isinstance(max_n, int) else label
        assert_same_length(image, label)
    
    # Draw figure
    nrows, ncols = get_grid_size(n=len(image), nrow=nrow)
    fig, axs     = plt.subplots(
        nrows   = nrows,
        ncols   = ncols,
        figsize = [ncols * scale, nrows  * scale],
        num     = figure_num,
        squeeze = False,
        clear   = True
    )
    move_figure(x=0, y=0)
    
    for idx, img in enumerate(image):
        i   = math.ceil(idx / nrow)
        j   = int(idx % nrow)
        img = to_image(image=img, keep_dims=False, denormalize=denormalize)
        axs[i, j].imshow(np.asarray(img))
        axs[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[i, j].set_title(label[i])
    plt.title(winname)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.show()
    
    # Save figure
    if save_cfg:
        filepath = save_cfg.pop("filepath")
        plt.savefig(filepath, **save_cfg)
    
    plt.show()
    plt.pause(wait_time)
    

def imshow_cls_plt(
    images 	    : Any,
    preds	    : Arrays | None     = None,
    targets	    : Arrays | None     = None,
    labels	    : Strs | None       = None,
    class_labels: ClassLabel | None = None,
    top_k	    : int 				= 5,
    scale       : int               = 1,
    save_cfg    : dict | None       = None,
    verbose	    : bool				= True,
    show_max_n  : int				= 8,
    wait_time   : float             = 0.01,
    figure_num  : int               = 0,
):
    """Visualize classification results on images using matplotlib.
    
    Args:
        images (Any):
            A sequence of images. Each element is of shape [B, C, H, W].
        preds (Arrays, None):
            A sequence of predicted classes probabilities. Default: `None`.
        targets (Arrays, None):
            A sequence of ground-truths. Default: `None`.
        labels (Sequence[str], None):
            Sequence of images' labels string. Default: `None`.
        class_labels (ClassLabels, None):
            `ClassLabels` objects that contains all class labels in the
            datasets. Default: `None`.
        top_k (int):
            Show only the top k classes' probabilities. Default: `5`.
        scale (int):
            Scale the size of matplotlib figure. `1` means (default size x 1).
        save_cfg (dict, None):
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
    from one.vision.acquisition import to_image
    from one.vision.acquisition import to_channel_last
    
    if not verbose and save_cfg is None:
        return
    
    # Prepare images
    images_ = to_image(images, denormalize=True)  # 4D-array
    images_ = to_channel_last(images_)
    images_ = images_[:show_max_n]
    
    # Prepare preds and targets
    pred_scores   = np.sort(preds, axis=1)[:show_max_n, -top_k:][:, ::]
    pred_labels   = preds.argsort(axis=1)[:show_max_n, -top_k:][:, ::]
    pred_1_labels = preds.argsort(axis=1)[:show_max_n, -top_k:][:, -1]
    if class_labels:
        pred_labels   = [[class_labels.get_name(value=l) for l in pred]
                         for pred in pred_labels]
        pred_1_labels = [class_labels.get_name(value=l) for l in pred_1_labels]
        targets 	  = [class_labels.get_name(value=l) for l in targets]
    
    # Prepare labels
    if labels is None:
        labels = [f"" for i in range(images_.shape[0])]
    labels = [f"{l} \n gt={gt}"
              for (l, pred, gt) in zip(labels, pred_1_labels, targets)]
    colors = ["darkgreen" if pred == gt else "red"
              for (pred, gt) in zip(pred_1_labels, targets)]
    if len(labels) != images_.shape[0]:
        raise ValueError(f"Length of `labels` and `images` batch size must be the same."
                         f" But got: {len(labels)} != {images_.shape[0]}.")
    
    # Create an image grid
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
    
    # Save figure
    if save_cfg:
        filepath = save_cfg.pop("filepath")
        plt.savefig(filepath, **save_cfg)

    # Show
    if verbose:
        plt.show()
        plt.pause(wait_time)


# H1: - All --------------------------------------------------------------------

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
