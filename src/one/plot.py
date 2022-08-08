# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plotting functions.
"""

from __future__ import annotations

import math
from typing import Any

import matplotlib
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import Tensor

import one.vision.transformation as t
from one.core import Arrays
from one.core import assert_same_length
from one.core import assert_tensor_of_ndim
from one.core import Ints
from one.core import Strs
from one.core import Tensors
from one.core import to_3d_tensor_list
from one.core import to_list
from one.data import ClassLabels
from one.vision.acquisition import to_image

# mpl.use("wxAgg")

plt.ion()
plt.switch_backend("qt5agg")
plt.rcParams["savefig.bbox"] = "tight"


# H1: - Drawing ----------------------------------------------------------------

def draw_pixel(image: Tensor, x: int, y: int, color: Tensor | Ints):
    """
    Draws a pixel into an image.
    
    Args:
        image (Tensor): The input image to where to draw the lines with shape 
            [C, H, W].
        x (int): The x coordinate of the pixel.
        y (int): The y coordinate of the pixel.
        color (Tensor |Ints): The color of the pixel with [C] where `C` is the
            number of channels of the image.
    """
    image[:, y, x] = color


def draw_rectangle(
    image    : Tensor,
    rectangle: Tensor,
    color    : Tensor | Ints | None = None,
    fill     : bool = False
) -> Tensor:
    """
    Draw N rectangles on a batch of image tensors.
    
    Args:
        image (Tensor): Tensor of [B, C, H, W].
        rectangle (Tensor): Represents number of rectangles to draw in [B, N, 4]
            N is the number of boxes to draw per batch index[x1, y1, x2, y2]
            4 is in (top_left.x, top_left.y, bot_right.x, bot_right.y).
        color (Tensor | Ints | None): A size 1, size 3, [B, N, 1], or [B, N, 3]
            tensor. If C is 3, and color is 1 channel it will be broadcasted.
        fill (bool): A flag used to fill the boxes with color if True.
    
    Returns:
        This operation modifies image inplace but also returns the drawn tensor
        for convenience with same shape the of the input [B, C, H, W].
    
    Example:
        >>> img  = torch.rand(2, 3, 10, 12)
        >>> rect = torch.tensor([[[0, 0, 4, 4]], [[4, 4, 10, 10]]])
        >>> out  = draw_rectangle(img, rect)
    """
    batch, c, h, w = image.shape
    batch_rect, num_rectangle, num_points = rectangle.shape
    if batch != batch_rect:
        raise ValueError("Image batch and rectangle batch must be equal.")
    if num_points != 4:
        raise ValueError("Number of points in rectangle must be 4.")
    
    # Clone rectangle, in case it's been expanded assignment from clipping
    # causes problems
    rectangle = rectangle.long().clone()
    
    # Clip rectangle to hxw bounds
    rectangle[:, :, 1::2] = torch.clamp(rectangle[:, :, 1::2], 0, h - 1)
    rectangle[:, :,  ::2] = torch.clamp(rectangle[:, :,  ::2], 0, w - 1)
    
    if color is None:
        color = torch.tensor([255.0] * c).expand(batch, num_rectangle, c)
    if isinstance(color, list):
        color = torch.tensor(color)
    if len(color.shape) == 1:
        color = color.expand(batch, num_rectangle, c)
    b, n, color_channels = color.shape
    if color_channels == 1 and c == 3:
        color = color.expand(batch, num_rectangle, c)

    for b in range(batch):
        for n in range(num_rectangle):
            if fill:
                image[
                    b, :,
                    int(rectangle[b, n, 1]) : int(rectangle[b, n, 3] + 1),
                    int(rectangle[b, n, 0]) : int(rectangle[b, n, 2] + 1),
                ] = color[b, n, :, None, None]
            else:
                image[
                    b, :,
                    int(rectangle[b, n, 1]) : int(rectangle[b, n, 3] + 1),
                    rectangle[b, n, 0]
                ] = color[b, n, :, None]
                image[
                    b, :,
                    int(rectangle[b, n, 1]) : int(rectangle[b, n, 3] + 1),
                    rectangle[b, n, 2]
                ] = color[b, n, :, None]
                image[
                    b, :,
                    rectangle[b, n, 1],
                    int(rectangle[b, n, 0]) : int(rectangle[b, n, 2] + 1)
                ] = color[b, n, :, None]
                image[
                    b, :,
                    rectangle[b, n, 3],
                    int(rectangle[b, n, 0]) : int(rectangle[b, n, 2] + 1)
                ] = color[b, n, :, None]

    return image


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
    images = t.denormalize(images) if denormalize else images
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
    denormalize: bool        = True,
    scale      : int         = 1,
    save_cfg   : dict | None = None,
    max_n      : int  | None = None,
    nrow       : int  | None = 8,
    wait_time  : float       = 0.01
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
    """
    # Prepare image and label
    image = to_3d_tensor_list(image)
    max_n = max_n if isinstance(max_n, int) else len(image)
    image = image[: max_n]
    
    if label is not None:
        label = to_list(label)
        label = label[:max_n]
        assert_same_length(image, label)
    
    # Draw figure
    n            = len(image)
    nrow         = n if n < nrow else nrow
    nrows, ncols = get_grid_size(n=n, nrow=nrow)
    fig, axs     = plt.subplots(
        nrows   = nrows,
        ncols   = ncols,
        figsize = [ncols * scale, nrows  * scale],
        num     = abs(hash(winname)) % (10 ** 8),
        squeeze = False,
        clear   = True
    )
    move_figure(x=0, y=0)
    
    for idx, img in enumerate(image):
        i   = int(idx / nrow)
        j   = int(idx % nrow)
        img = to_image(image=img, keepdim=False, denormalize=denormalize)
        axs[i, j].imshow(np.asarray(img))
        axs[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if label is not None:
            axs[i, j].set_title(label[i])
    plt.get_current_fig_manager().set_window_title(winname)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    
    # Save figure
    if save_cfg:
        filepath = save_cfg.pop("filepath")
        plt.savefig(filepath, **save_cfg)
    
    plt.show()
    plt.pause(wait_time)
    

def imshow_cls(
    winname    : str,
    image 	   : Tensor,
    pred	   : Tensor | None      = None,
    target	   : Tensor | None      = None,
    label	   : Strs   | None      = None,
    classlabels: ClassLabels | None = None,
    top_k	   : int  | None    	= 5,
    denormalize: bool               = True,
    scale      : int                = 1,
    save_cfg   : dict | None        = None,
    max_n      : int  | None 		= None,
    nrow       : int  | None        = 8,
    wait_time  : float              = 0.01,
):
    """
    Show classification results on images using matplotlib.
    
    Args:
        winname (str): The name of the window to display the image in.
        image (Tensor): The images to be displayed. Can be a 3D tensor
            [C, H, W] or 4D tensor of shape [B, C, H, W].
        pred (Tensor | None): Predicted classes probabilities. Can be a tensor
            of shape [B, N] where N is the total number of all classes in the
            dataset. Defaults to None.
        target (Tensor | None): A sequence of ground-truths id. Defaults to None.
        label (Strs | None): Sequence of images' labels string. Defaults to
            None.
        classlabels (ClassLabels | None): ClassLabels objects that contains all
            class labels in the datasets. Defaults to None.
        top_k (int | None): Show only the top k classes' probabilities. If None
            then show all. Defaults to 5.
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
    """
    # Prepare image and label
    image = to_3d_tensor_list(image)
    max_n = max_n if isinstance(max_n, int) else len(image)
    top_k = top_k if isinstance(top_k, int) else (
        len(classlabels.list) if isinstance(classlabels, ClassLabels) else 5
    )
    image = image[: max_n]

    scores_topk = None
    pred_topk   = None
    pred_top1   = None
    if isinstance(pred, Tensor):
        assert_tensor_of_ndim(pred, 2)
        pred          = pred.clone()
        probs, idxes  = torch.sort(pred, dim=1)
        scores_topk   = probs[:max_n, -top_k:][:, ::].tolist()
        pred_topk     = idxes[:max_n, -top_k:][:, ::].tolist()
        pred_top1     = idxes[:max_n, -top_k:][:, -1].tolist()
    else:
        scores_topk   = [[0.0] * top_k for _ in range(len(image))]
        
    if isinstance(classlabels, ClassLabels):
        if pred_topk:
            pred_topk = [[classlabels.get_name(key="id", value=v) for v in p]
                          for p in pred_topk]
        if pred_top1 is not None:
            pred_top1 = [classlabels.get_name(key="id", value=v) for v in pred_top1]
        if target is not None:
            target    = target.clone()
            target    = target.tolist()
            target    = [classlabels.get_name(key="id", value=v) for v in target]
            
    label = label[:max_n] if label is not None else [f"" for _ in range(len(image))]
    assert_same_length(image, label)
    
    if pred_top1 and target:
        label  = [f"{l} \n pred={p} gt={t}"        for (l, p, t) in zip(label, pred_top1, target)]
        colors = ["darkgreen" if p == t else "red" for (p, t) in zip(pred_top1, target)]
    elif target:
        label  = [f"{l} \n gt={t}" for (l, t) in zip(label, target)]
        colors = ["black" for _ in range(len(image))]
    else:
        colors = ["black" for _ in range(len(image))]
    
    # Draw figure
    n            = len(image)
    nrow         = n if n < nrow else nrow
    nrows, ncols = get_grid_size(n=n, nrow=nrow)
    y_pos        = np.arange(top_k)
    fig = plt.figure(
        constrained_layout = True,
        figsize            = [(ncols * 1.0) * scale, (nrows * 1.5) * scale],
        num                = abs(hash(winname)) % (10 ** 8),
    )
    subfigs      = fig.subfigures(nrows=nrows, ncols=ncols)
    subfigs_flat = subfigs.flat
    move_figure(x=0, y=0)
    
    for idx, (img, label, color, scores) in \
        enumerate(zip(image, label, colors, scores_topk)):
        subfig = subfigs_flat[idx]
        subfig.suptitle(label, x=0.5, y=1.0, color=color)
        axs = subfig.subplots(2, 1)
        for ax in axs.flat:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis="x", direction="in", pad=-10)
            ax.tick_params(axis="y", direction="in", pad=-10)
        # Image
        img = to_image(image=img, keepdim=False, denormalize=denormalize)
        axs[0].imshow(np.asarray(img), aspect="auto")
        # Classlabels
        pps = axs[1].barh(y_pos, scores, align="center", color="deepskyblue")
        axs[1].set_xlim(left=0.0)
        axs[1].set_yticks(y_pos)
        axs[1].set_yticklabels(scores, horizontalalignment="left")
        # Scores
        max_width = max([rect.get_width() for rect in pps])
        for i, rect in enumerate(pps):
            if scores[i] > 0:
                axs[1].text(
                    rect.get_x() + max_width, rect.get_y(),
                    "{:.2f}".format(scores[i]), ha="right", va="bottom",
                    rotation=0
                )
    plt.get_current_fig_manager().set_window_title(winname)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    
    # Save figure
    if save_cfg:
        filepath = save_cfg.pop("filepath")
        plt.savefig(filepath, **save_cfg)

    plt.show()
    plt.pause(wait_time)
