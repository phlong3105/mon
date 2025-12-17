"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""

import contextlib
import copy
import os

import faster_coco_eval.core.mask as mask_util
import numpy as np
import torch
from faster_coco_eval import COCO, COCOeval_faster

from ...core import register
from ...misc import dist_utils

__all__ = ["CocoEvaluator", "CocoEvaluatorFishEye8K"]


@register()
class CocoEvaluator(object):

    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt            = copy.deepcopy(coco_gt)
        self.coco_gt: COCO = coco_gt
        self.iou_types     = iou_types

        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(coco_gt, iouType=iou_type, print_function=print, separate_eval=True)

        self.img_ids   = []
        self.eval_imgs = {k: [] for k in iou_types}

    def cleanup(self):
        self.coco_eval = {}
        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(self.coco_gt, iouType=iou_type, print_function=print, separate_eval=True)
        self.img_ids   = []
        self.eval_imgs = {k: [] for k in self.iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results   = self.prepare(predictions, iou_type)
            coco_eval = self.coco_eval[iou_type]

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = self.coco_gt.loadRes(results) if results else COCO()
                    coco_eval.cocoDt = coco_dt
                    coco_eval.params.imgIds = list(img_ids)
                    coco_eval.evaluate()

            self.eval_imgs[iou_type].append(
                np.array(coco_eval._evalImgs_cpp).reshape(
                    len(coco_eval.params.catIds),
                    len(coco_eval.params.areaRng),
                    len(coco_eval.params.imgIds)
                )
            )

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            img_ids, eval_imgs = merge(self.img_ids, self.eval_imgs[iou_type])

            coco_eval = self.coco_eval[iou_type]
            coco_eval.params.imgIds = img_ids
            coco_eval._paramsEval   = copy.deepcopy(coco_eval.params)
            coco_eval._evalImgs_cpp = eval_imgs

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize_original(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    #######################
    # My Modifications    #
    # Calculate F1 scores #
    #######################
    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

            # Calculate F1 scores
            precisions     = coco_eval.eval['precision']  # Shape: (T, R, K, A, M)
            recalls        = coco_eval.params.recThrs     # Shape: (R,)
            iou_thrs       = coco_eval.params.iouThrs     # Shape: (T,)
            area_rngs      = coco_eval.params.areaRngLbl  # ['all', 'small', 'medium', 'large']
            max_dets       = coco_eval.params.maxDets     # e.g., [1, 10, 100]
            # Initialize arrays for appending to stats
            new_stats      = []
            new_all_stats  = []
            metric_configs = [
                {"name": "(F1) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]", "iou_idx": slice(None),                      "area_idx": 0, "max_det_idx": -1},
                {"name": "(F1) @[ IoU=0.50      | area=   all | maxDets=100 ]", "iou_idx": np.where(iou_thrs == 0.50)[0][0], "area_idx": 0, "max_det_idx": -1},
                {"name": "(F1) @[ IoU=0.75      | area=   all | maxDets=100 ]", "iou_idx": np.where(iou_thrs == 0.75)[0][0], "area_idx": 0, "max_det_idx": -1},
                {"name": "(F1) @[ IoU=0.5:0.95  | area= small | maxDets=100 ]", "iou_idx": slice(None),                      "area_idx": 1, "max_det_idx": -1},
                {"name": "(F1) @[ IoU=0.5:0.95  | area=medium | maxDets=100 ]", "iou_idx": slice(None),                      "area_idx": 2, "max_det_idx": -1},
                {"name": "(F1) @[ IoU=0.5:0.95  | area= large | maxDets=100 ]", "iou_idx": slice(None),                      "area_idx": 3, "max_det_idx": -1},
                {"name": "(F1) @[ IoU=0.5:0.95  | area=   all | maxDets=  1 ]", "iou_idx": slice(None),                      "area_idx": 0, "max_det_idx":  0},
                {"name": "(F1) @[ IoU=0.5:0.95  | area=   all | maxDets= 10 ]", "iou_idx": slice(None),                      "area_idx": 0, "max_det_idx":  1},
                {"name": "(F1) @[ IoU=0.5:0.95  | area=   all | maxDets=100 ]", "iou_idx": slice(None),                      "area_idx": 0, "max_det_idx": -1},
            ]

            for config in metric_configs:
                name        = config["name"]
                iou_idx     = config["iou_idx"]
                area_idx    = config["area_idx"]
                max_det_idx = config["max_det_idx"]

                # Extract precision
                precision = precisions[iou_idx, :, :, area_idx, max_det_idx]
                if isinstance(iou_idx, slice):
                    precision = np.mean(precision, axis=0)  # Average over IoU thresholds
                precision = np.mean(precision, axis=1)      # Average over categories
                recall    = recalls

                valid = precision > -1
                if not np.any(valid):
                    print(f"Warning: No valid detections for {name}")
                    best_precision = best_recall = best_f1 = 0.0
                else:
                    precision      = precision[valid]
                    recall         = recall[valid]
                    f1_scores      = np.zeros_like(precision)
                    non_zero       = (precision + recall) > 0
                    f1_scores[non_zero] = 2 * (precision[non_zero] * recall[non_zero]) / (precision[non_zero] + recall[non_zero])
                    best_idx       = np.argmax(f1_scores)
                    best_f1        = f1_scores[best_idx]
                    best_precision = precision[best_idx]
                    best_recall    = recall[best_idx]

                print(f" F1-Score           {name} = {best_f1:.3f} (Precision: {best_precision:.3f}, Recall: {best_recall:.3f})")

                # Append to stats
                # new_stats.extend(    [best_precision, best_recall, best_f1])
                # new_all_stats.extend([best_precision, best_recall, best_f1])
                new_stats.extend(    [best_f1])
                new_all_stats.extend([best_f1])

            # Update coco_eval.stats and all_stats
            coco_eval.all_stats = np.append(coco_eval.all_stats, new_stats)
            coco_eval.stats     = np.append(coco_eval.stats,     new_stats)

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes  = prediction["boxes"]
            boxes  = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id"   : original_id,
                        "category_id": labels[k],
                        "bbox"       : box,
                        "score"      : scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks  = prediction["masks"]

            masks  = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles   = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id"    : original_id,
                        "category_id" : labels[k],
                        "segmentation": rle,
                        "score"       : scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes     = prediction["boxes"]
            boxes     = convert_to_xywh(boxes).tolist()
            scores    = prediction["scores"].tolist()
            labels    = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id"   : original_id,
                        "category_id": labels[k],
                        'keypoints'  : keypoint,
                        "score"      : scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


@register()
class CocoEvaluatorFishEye8K(object):

    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt            = copy.deepcopy(coco_gt)
        self.coco_gt: COCO = coco_gt
        self.iou_types     = iou_types

        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(coco_gt, iouType=iou_type, print_function=print, separate_eval=True)
            self.coco_eval[iou_type].params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 64 ** 2], [64 ** 2, 192 ** 2], [192 ** 2, 1e5 ** 2]]
            print("Set areaRng to: {}".format(self.coco_eval[iou_type].params.areaRng))

        self.img_ids   = []
        self.eval_imgs = {k: [] for k in iou_types}

    def cleanup(self):
        self.coco_eval = {}
        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(self.coco_gt, iouType=iou_type, print_function=print, separate_eval=True)
        self.img_ids   = []
        self.eval_imgs = {k: [] for k in self.iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results   = self.prepare(predictions, iou_type)
            coco_eval = self.coco_eval[iou_type]

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = self.coco_gt.loadRes(results) if results else COCO()
                    coco_eval.cocoDt = coco_dt
                    coco_eval.params.imgIds = list(img_ids)
                    coco_eval.evaluate()

            self.eval_imgs[iou_type].append(
                np.array(coco_eval._evalImgs_cpp).reshape(
                    len(coco_eval.params.catIds),
                    len(coco_eval.params.areaRng),
                    len(coco_eval.params.imgIds)
                )
            )

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            img_ids, eval_imgs = merge(self.img_ids, self.eval_imgs[iou_type])

            coco_eval = self.coco_eval[iou_type]
            coco_eval.params.imgIds = img_ids
            coco_eval._paramsEval   = copy.deepcopy(coco_eval.params)
            coco_eval._evalImgs_cpp = eval_imgs

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize_original(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    #######################
    # My Modifications    #
    # Calculate F1 scores #
    #######################
    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

            # Calculate F1 scores
            precisions     = coco_eval.eval['precision']  # Shape: (T, R, K, A, M)
            recalls        = coco_eval.params.recThrs     # Shape: (R,)
            iou_thrs       = coco_eval.params.iouThrs     # Shape: (T,)
            area_rngs      = coco_eval.params.areaRngLbl  # ['all', 'small', 'medium', 'large']
            max_dets       = coco_eval.params.maxDets     # e.g., [1, 10, 100]
            # Initialize arrays for appending to stats
            new_stats      = []
            new_all_stats  = []
            metric_configs = [
                {"name": "(F1) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]", "iou_idx": slice(None),                      "area_idx": 0, "max_det_idx": -1},
                {"name": "(F1) @[ IoU=0.50      | area=   all | maxDets=100 ]", "iou_idx": np.where(iou_thrs == 0.50)[0][0], "area_idx": 0, "max_det_idx": -1},
                {"name": "(F1) @[ IoU=0.75      | area=   all | maxDets=100 ]", "iou_idx": np.where(iou_thrs == 0.75)[0][0], "area_idx": 0, "max_det_idx": -1},
                {"name": "(F1) @[ IoU=0.5:0.95  | area= small | maxDets=100 ]", "iou_idx": slice(None),                      "area_idx": 1, "max_det_idx": -1},
                {"name": "(F1) @[ IoU=0.5:0.95  | area=medium | maxDets=100 ]", "iou_idx": slice(None),                      "area_idx": 2, "max_det_idx": -1},
                {"name": "(F1) @[ IoU=0.5:0.95  | area= large | maxDets=100 ]", "iou_idx": slice(None),                      "area_idx": 3, "max_det_idx": -1},
                {"name": "(F1) @[ IoU=0.5:0.95  | area=   all | maxDets=  1 ]", "iou_idx": slice(None),                      "area_idx": 0, "max_det_idx":  0},
                {"name": "(F1) @[ IoU=0.5:0.95  | area=   all | maxDets= 10 ]", "iou_idx": slice(None),                      "area_idx": 0, "max_det_idx":  1},
                {"name": "(F1) @[ IoU=0.5:0.95  | area=   all | maxDets=100 ]", "iou_idx": slice(None),                      "area_idx": 0, "max_det_idx": -1},
            ]

            for config in metric_configs:
                name        = config["name"]
                iou_idx     = config["iou_idx"]
                area_idx    = config["area_idx"]
                max_det_idx = config["max_det_idx"]

                # Extract precision
                precision = precisions[iou_idx, :, :, area_idx, max_det_idx]
                if isinstance(iou_idx, slice):
                    precision = np.mean(precision, axis=0)  # Average over IoU thresholds
                precision = np.mean(precision, axis=1)      # Average over categories
                recall    = recalls

                valid = precision > -1
                if not np.any(valid):
                    print(f"Warning: No valid detections for {name}")
                    best_precision = best_recall = best_f1 = 0.0
                else:
                    precision      = precision[valid]
                    recall         = recall[valid]
                    f1_scores      = np.zeros_like(precision)
                    non_zero       = (precision + recall) > 0
                    f1_scores[non_zero] = 2 * (precision[non_zero] * recall[non_zero]) / (precision[non_zero] + recall[non_zero])
                    best_idx       = np.argmax(f1_scores)
                    best_f1        = f1_scores[best_idx]
                    best_precision = precision[best_idx]
                    best_recall    = recall[best_idx]

                print(f" F1-Score           {name} = {best_f1:.3f} (Precision: {best_precision:.3f}, Recall: {best_recall:.3f})")

                # Append to stats
                # new_stats.extend(    [best_precision, best_recall, best_f1])
                # new_all_stats.extend([best_precision, best_recall, best_f1])
                new_stats.extend(    [best_f1])
                new_all_stats.extend([best_f1])

            # Update coco_eval.stats and all_stats
            coco_eval.all_stats = np.append(coco_eval.all_stats, new_stats)
            coco_eval.stats     = np.append(coco_eval.stats,     new_stats)

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes  = prediction["boxes"]
            boxes  = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id"   : original_id,
                        "category_id": labels[k],
                        "bbox"       : box,
                        "score"      : scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks  = prediction["masks"]

            masks  = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles   = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id"    : original_id,
                        "category_id" : labels[k],
                        "segmentation": rle,
                        "score"       : scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes     = prediction["boxes"]
            boxes     = convert_to_xywh(boxes).tolist()
            scores    = prediction["scores"].tolist()
            labels    = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id"   : original_id,
                        "category_id": labels[k],
                        'keypoints'  : keypoint,
                        "score"      : scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids    = dist_utils.all_gather(img_ids)
    all_eval_imgs  = dist_utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.extend(p)

    merged_img_ids   = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, axis=2).ravel()
    # merged_eval_imgs = np.array(merged_eval_imgs).T.ravel()

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)

    return merged_img_ids.tolist(), merged_eval_imgs.tolist()
