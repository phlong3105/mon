#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Measure metrics for DarkCityscapes dataset."""

from __future__ import annotations

import os

import click
import cv2
import numpy as np

import mon

console = mon.console


# region Function

class SegmentationMetric:
    def __init__(self, num_class):
        self.confusion_matrix = None
        self.num_class        = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def pixel_accuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def class_pixel_accuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        class_acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)  # IMPORTANT: the axis must be 0
        return class_acc

    def mean_pixel_accuracy(self):
        class_acc = self.class_pixel_accuracy()
        mean_acc  = np.nanmean(class_acc)
        return mean_acc

    def mean_iou(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix)  # axis = 1 -> row value； axis = 0 -> column value
        IoU   = intersection / union
        mIoU  = np.nanmean(IoU)
        return mIoU

    def gen_confusion_matrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask  = (imgLabel >= 0) & (imgLabel < self.num_class)
        label = self.num_class * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def frequency_weighted_iou(self):
        # FWIOU = [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq  = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu    = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def add_batch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusion_matrix += self.gen_confusion_matrix(imgPredict, imgLabel)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class, self.num_class))


def color_to_gray(path):
    # color to grayscale and then flatten
    src       = cv2.imread(path)
    gray      = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 125, 1, cv2.THRESH_BINARY)
    thresh    = thresh[:1020, :2040]
    pred      = thresh.flatten()
    return pred


@click.command(name="metric", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option(
    "--input-dir",
    type    = click.Path(exists=False),
    default = mon.DATA_DIR/"llie"/"predict"/"MODEL"/"darkcityscapes"/"hrnetv2"/"single-scale"/"pseudo_color",
    help    = "Image directory."
)
@click.option(
    "--target-dir",
    type    = click.Path(exists=False),
    default = mon.DATA_DIR/"llie"/"test"/"darkcityscapes"/"labels",
    help    = "Ground-truth directory."
)
@click.option("--result-file",    type=str, default=None,           help="Result file.")
@click.option("--name",           type=str, default="ground-truth", help="Model name.")
@click.option("--variant",        type=str, default=None,           help="Model variant.")
@click.option("--save-txt",       is_flag=True)
@click.option("--append-results", is_flag=True)
@click.option("--verbose",        is_flag=True)
def measure_metric(
    input_dir     : mon.Path,
    target_dir    : mon.Path | None,
    result_file   : mon.Path | str,
    name          : str,
    variant       : int | str | None,
    save_txt      : bool,
    append_results: bool,
    verbose       : bool,
):
    variant       = variant if variant not in [None, "", "none"] else None
    model_variant = f"{name}-{variant}" if variantImageDataset else f"{name}"
    console.rule(f"[bold red] {model_variant}")

    input_dir = input_dir.replace("MODEL", model_variant)

    assert input_dirImageDataset and mon.Path(input_dir).is_dir()
    assert target_dirImageDataset and mon.Path(target_dir).is_dir()

    if result_fileImageDataset:
        assert (mon.Path(result_file).is_dir()
                or mon.Path(result_file).is_file()
                or isinstance(result_file, str))
        result_file = mon.Path(result_file)

    input_dir  = mon.Path(input_dir)
    target_dir = mon.Path(target_dir) \
        if target_dirImageDataset \
        else input_dir.replace("low", "labels")

    result_file = mon.Path(result_file) if result_fileImageDataset else None
    if save_txt and result_fileImageDataset and result_file.is_dir():
        result_file /= "metric.txt"
        result_file.parent.mkdir(parents=True, exist_ok=True)

    image_files = list(input_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)

    # Measuring
    num_items = 0
    metric    = SegmentationMetric(num_class=19)
    with mon.get_progress_bar() as pbar:
        for image_file in pbar.track(
            sequence    = image_files,
            total       = len(image_files),
            description = f"[bright_yellow] Measuring"
        ):
            image = color_to_gray(path=str(image_file))

            target_file = None
            stem        = image_file.stem
            stem        = stem.replace("_leftImg8bit", "_gtFine_color")
            for ext in mon.IMAGE_FILE_FORMATS:
                temp = target_dir / f"{stem}{ext}"
                if temp.exists():
                    target_file = temp

            if target_fileImageDataset and target_file.exists():
                num_items += 1
                target     = color_to_gray(path=str(target_file))
                metric.add_batch(image, target)

    mIoU = metric.mean_iou()
    mPA  = metric.mean_pixel_accuracy()

    # Show results
    if append_results:
        console.log(f"{model_variant}")
        console.log(f"{input_dir.name}")
        message = ""
        message += f"{'mIoU':<10}\t"
        message += f"{'mPA':<10}\t"
        message += "\n"
        message += f"{mIoU:.10f}\t"
        message += f"{mPA:.10f}\t"
        console.log(f"{message}")
        print(f"COPY THIS:")
        print(message)
    else:
        console.log(f"{model_variant}")
        console.log(f"{input_dir.name}")
        console.log(f"{'mIoU':<10}: {mIoU:.10f}")
        console.log(f"{'mPA':<10}: {mPA:.10f}")
    # Save results
    if save_txt:
        if not append_results:
            mon.delete_files(regex=result_file.name, path=result_file.parent)
        with open(str(result_file), "a") as f:
            if os.stat(str(result_file)).st_size == 0:
                f.write(f"{'model':<10}\t{'data':<10}\t")
                f.write(f"{f'{mIoU}':<10}\t")
                f.write(f"{f'{mPA}':<10}\t")
            f.write(f"{f'{model_variant}':<10}\t{f'{input_dir.name}':<10}\t")
            f.write(f"{mIoU:.10f}\t")
            f.write(f"{mPA:.10f}\t")
            f.write(f"\n")
 

if __name__ == "__main__":
    measure_metric()

# endregion
