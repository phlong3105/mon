# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements visualization functions.

https://www.geeksforgeeks.org/concatenate-images-using-opencv-in-python/
"""

from __future__ import annotations

__all__ = [
    "get_grid_size", "imshow", "imshow_classification", "imshow_enhancement",
    "move_figure", "plt",
]

from typing import Any

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt

from mon.core import builtins, math
from mon.nn import data as md
from mon.vision import core

# mpl.use("wxAgg")

plt.ion()
plt.show()
# plt.switch_backend("qt6agg")
plt.rcParams["savefig.bbox"] = "tight"


# region Window Positioning

def get_grid_size(n: int, nrow: int | None = 4) -> list[int]:
    """Calculate the number of rows and columns needed to display the items
    in a grid.
    
    Args:
        n: The number of items.
        nrow: The number of items in a row. The final grid size is
            :math:`(n / nrow, nrow)`. If ``None``, put all items in a single
            row. Default: ``4``.
    
    Returns:
        A :class:`tuple` of :math:`(nrows, ncols)`, where nrows is the number of
        rows and ncols is the number of columns.
    """
    if isinstance(nrow, int) and nrow > 0:
        ncols = nrow
    else:
        ncols = n
    nrows = math.ceil(n / ncols)
    return [nrows, ncols]


def move_figure(x: int, y: int):
    """Move the matplotlib figure around the window. The upper-left corner to
    the location specified by :math:`(x, y)`.
    """
    mngr = plt.get_current_fig_manager()
    fig  = plt.gcf()
    backend = matplotlib.get_backend()
    if backend == "TkAgg":
        mngr.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == "WXAgg":
        mngr.window.SetPosition((x, y))
    else:  # This works for QT and GTK. You can use window.setGeometry
        mngr.window.move(x, y)


# endregion


# region Show

def imshow(
    winname    : str,
    image      : Any,
    label      : str | list[str] | None = None,
    denormalize: bool                   = True,
    scale      : int                    = 1,
    save_config: dict | None            = None,
    max_n      : int  | None            = None,
    nrow       : int  | None            = 8,
    wait_time  : float                  = 0.01
):
    """Show an image tensor in a window using matplotlib.
    
    Args:
        winname: The name of the window to display the image in.
        image: The images to be displayed. Can be a 3-D or 4-D image, or a
            :class:`list` of 3-D images.
        label: Sequence of images' labels string. Default: ``None``.
        denormalize: If ``True``, convert image to :math:`[0, 255]`.
            Default: ``True``.
        scale: Scale the size of matplotlib figure. Default: ``1`` means
            :math:`size x 1`.
        save_config: Save figure config. Default: ``None``.
        max_n: Show max n images if :param:`image` has a batch size of more than
            :param:`max_n` images. Default: ``None`` means show all.
        nrow: The maximum number of items to display in a row. The final grid
            size is :math:`(n / nrow, nrow)`. If ``None``, then the number of
            items in a row will be the same as the number of items in the
            :class:`list`. Default: ``8``.
        wait_time: Wait for some time (in seconds) to display the figure then
            reset. Default: ``0.01``.
    """
    # Prepare image and label
    image = core.to_list_of_3d_image(image)
    max_n = max_n if isinstance(max_n, int) else len(image)
    image = image[: max_n]
    
    if label is not None:
        label = builtins.to_list(x=label)
        label = label[:max_n]
        if not len(image) == len(label):
            raise ValueError(
                f"image and label must have the same length, but got "
                f"{len(image)} and {len(label)}."
            )
    
    # Draw figure
    n    = len(image)
    nrow = n if n < nrow else nrow
    nrows, ncols = get_grid_size(n=n, nrow=nrow)
    fig, axs = plt.subplots(
        nrows   = nrows,
        ncols   = ncols,
        figsize = [ncols * scale, nrows * scale],
        num     = abs(hash(winname)) % (10 ** 8),
        squeeze = False,
        clear   = True
    )
    # move_figure(x=0, y=0)
    
    for idx, img in enumerate(image):
        i   = int(idx / nrow)
        j   = int(idx % nrow)
        img = core.to_image_nparray(
            input=img, keepdim=False, denormalize=denormalize)
        axs[i, j].imshow(np.asarray(img), aspect="auto")
        axs[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if label is not None:
            axs[i, j].set_title(label[i])
    plt.get_current_fig_manager().set_window_title(winname)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    
    # Save figure
    if save_config:
        filepath = save_config.pop("filepath")
        plt.savefig(filepath, **save_config)
    
    plt.pause(wait_time)


def imshow_classification(
    winname    : str,
    image      : Any,
    pred       : Any                    = None,
    target     : Any                    = None,
    label      : str | list[str] | None = None,
    classlabels: md.ClassLabels | None  = None,
    top_k      : int  | None            = 5,
    denormalize: bool                   = True,
    scale      : int                    = 1,
    save_config: dict | None            = None,
    max_n      : int  | None            = None,
    nrow       : int  | None            = 8,
    wait_time  : float                  = 0.01,
):
    """Show classification results on images using matplotlib.
    
    Args:
        winname: The name of the window to display the image in.
        image: The images to be displayed. Can be a 3-D or 4-D image, or a
            :class:`list` of 3-D images.
        pred: Predicted classes probabilities. Can be a tensor of shape
            :math:`[B, N]` where ``N`` is the total number of all classes in the
            dataset. Defaults to ``None``.
        target: A sequence of ground-truths ID. Default: ``None``.
        label: A sequence of images' labels strings. Default: ``None``.
        classlabels: :class:`mon.nn.ClassLabels` objects that contain all class
            labels in the datasets. Default: ``None``.
        top_k: Show only the top k classes' probabilities. If ``None`` then
            shows all. Default: ``5``.
        denormalize: If ``True``, denormalize the image :math:`[0, 255]`.
            Default: ``True``.
        scale: Scale the size of matplotlib figure. Default: ``1`` means
            :math:`size x 1`.
        save_config: Save figure config. Default: ``None``.
        max_n: Show max n images if :param:`image` has a batch size of more than
            :param:`max_n` images. Default: ``None`` means show all.
        nrow: The maximum number of items to display in a row. The final grid
            size is :math:`(n / nrow, nrow)`. If ``None``, then the number of
            items in a row will be the same as the number of items in the
            :class:`list`. Default: ``8``.
        wait_time: Wait for some time (in seconds) to display the figure then
            reset. Default: ``0``.
    """
    # Prepare image and label
    image = core.to_list_of_3d_image(image)
    max_n = max_n if isinstance(max_n, int) else len(image)
    top_k = top_k if isinstance(top_k, int) else (
        len(classlabels) if isinstance(classlabels, md.ClassLabels) else 5
    )
    image = image[: max_n]
    
    pred_topk = None
    pred_top1 = None
    if isinstance(pred, torch.Tensor):
        assert isinstance(pred, torch.Tensor) and pred.ndim == 2
        pred         = pred.clone()
        probs, idxes = torch.sort(pred, dim=1)
        scores_topk  = probs[:max_n, -top_k:][:, ::].tolist()
        pred_topk    = idxes[:max_n, -top_k:][:, ::].tolist()
        pred_top1    = idxes[:max_n, -top_k:][:, -1].tolist()
    else:
        scores_topk = [[0.0] * top_k for _ in range(len(image))]
    
    if isinstance(classlabels, md.ClassLabels):
        if pred_topk:
            pred_topk = [
                [classlabels.get_name(key="id", value=v) for v in p]
                for p in pred_topk
            ]
        if pred_top1 is not None:
            pred_top1 = [classlabels.get_name(key="id", value=v) for v in pred_top1]
        if target is not None:
            target = target.clone()
            target = target.tolist()
            target = [classlabels.get_name(key="id", value=v) for v in target]
    
    label = label[:max_n] if label is not None else [f"" for _ in range(len(image))]
    if not len(image) == len(label):
        raise ValueError(
            f"image and label must have the same length, but got "
            f"{len(image)} and {len(label)}."
        )
    
    if pred_top1 is not None and target is not None:
        label  = [f"{l} \n pred={p} gt={t}" for (l, p, t) in zip(label, pred_top1, target)]
        colors = ["darkgreen" if p == t else "red" for (p, t) in zip(pred_top1, target)]
    elif target is not None:
        label  = [f"{l} \n gt={t}" for (l, t) in zip(label, target)]
        colors = ["black" for _ in range(len(image))]
    else:
        colors = ["black" for _ in range(len(image))]
    
    # Draw figure
    n = len(image)
    nrow = n if n < nrow else nrow
    nrows, ncols = get_grid_size(n=n, nrow=nrow)
    y_pos = np.arange(top_k)
    fig = plt.figure(
        constrained_layout=True,
        figsize=[(ncols * 1.0) * scale, (nrows * 1.5) * scale],
        num=abs(hash(winname)) % (10 ** 8),
    )
    subfigs      = fig.subfigures(nrows=nrows, ncols=ncols)
    subfigs_flat = subfigs.flat
    # move_figure(x=0, y=0)
    
    for idx, (img, label, color, scores) in enumerate(
        zip(image, label, colors, scores_topk)
    ):
        subfig = subfigs_flat[idx]
        subfig.suptitle(label, x=0.5, y=1.0, color=color)
        axs = subfig.subplots(2, 1)
        for ax in axs.flat:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis="x", direction="in", pad=-10)
            ax.tick_params(axis="y", direction="in", pad=-10)
        # Image
        img = core.to_image_nparray(
            input=img, keepdim=False, denormalize=denormalize)
        axs[0].imshow(np.asarray(img), aspect="auto")
        # Classlabels
        pps = axs[1].barh(y_pos, scores, align="center", color="deepskyblue")
        axs[1].set_xlim(left=0.0)
        axs[1].set_yticks(y_pos)
        if pred_topk is not None:
            axs[1].set_yticklabels(pred_topk[idx], horizontalalignment="left")
        # Scores
        max_width = max([rect.get_width() for rect in pps])
        for i, rect in enumerate(pps):
            if scores[i] > 0:
                axs[1].text(
                    rect.get_x() + max_width, rect.get_y(),
                    f"{scores[i]:.2f}", ha="right", va="bottom",
                    rotation=0
                )
    plt.get_current_fig_manager().set_window_title(winname)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    
    # Save figure
    if save_config:
        filepath = save_config.pop("filepath")
        plt.savefig(filepath, **save_config)
    
    plt.pause(wait_time)


def imshow_enhancement(
    winname    : str,
    image      : dict,
    label      : str | list[str] | None = None,
    denormalize: bool        = True,
    scale      : int         = 1,
    save_config: dict | None = None,
    max_n      : int | None  = None,
    nrow       : int | None  = 8,
    wait_time  : float       = 0.01,
):
    """Show image enhancement results using matplotlib.
    
    Args:
        winname: The name of the window to display the image in.
        image: A collection of images to be displayed. Each item is a 4D tensor
            of shape :math:`[B, C, H, W]` represented an image type (i.e.,
            input, pred, target, enhanced image, ...). If given a dictionary,
            the key will be used as the column label.
        label: A sequence of images' labels :class:`str`. Default: ``None``.
        denormalize: If ``True``, convert image to :math:`[0, 255]`.
            Default: ``True``.
        scale: Scale the size of matplotlib figure. Default: ``1`` means
            :math:`size x 1`.
        save_config: Save figure config. Default: ``None``.
        max_n: Show max n images if :param:`image` has a batch size of more than
            :param:`max_n` images. Default: ``None`` means show all.
        nrow: The maximum number of items to display in a row. The final grid
            size is :math:`(n / nrow, nrow)`. If ``None``, then the number of
            items in a row will be the same as the number of items in the
            :class:`list`. Default: ``8``.
        wait_time: Wait for some time (in seconds) to display the figure then
            reset. Default: ``0``.
    """
    # Prepare image and label
    header = list(image.keys())
    image  = list(image.values())
    image  = [core.to_list_of_3d_image(i) for i in image]
    max_n  = max_n if isinstance(max_n, int) else len(image[0])
    image  = [i[: max_n] for i in image]
    
    assert len(image) == len(header)
    if label is not None:
        label = builtins.to_list(label)
        label = label[:max_n]
        assert len(image[0]) == len(label)
    
    # Draw figure
    ncols = len(image)
    nrows = max_n
    fig, axs = plt.subplots(
        nrows   = nrows,
        ncols   = ncols,
        figsize = [ncols * scale, nrows * scale],
        num     = abs(hash(winname)) % (10 ** 8),
        squeeze = False,
        clear   = True
    )
    # move_figure(x=0, y=0)
    
    [ax.set_title(l) for ax, l in zip(axs[0], header)]
    for i, img in enumerate(image):
        for j, im in enumerate(img):
            im = core.to_image_nparray(
                input=im, keepdim=False, denormalize=denormalize)
            axs[j, i].imshow(np.asarray(im), aspect="auto")
            axs[j, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        # if label is not None:
        #     axs[i].set_title(label[i])
    plt.get_current_fig_manager().set_window_title(winname)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    
    # Save figure
    if save_config:
        filepath = save_config.pop("filepath")
        plt.savefig(filepath, **save_config)
    
    plt.pause(wait_time)

# endregion
