#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import click

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import mon


# region Main

@click.command(name="metric", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--result-file", type=click.Path(exists=True),  default=None, help="Result .json file.")
@click.option("--gt-file",     type=click.Path(exists=False), default=None, help="Groundtruth .json file.")
@click.option("--verbose",     is_flag=True)
def measure_metric(
    result_file: str,
    gt_file    : str,
    verbose    : bool,
):
    assert result_file is not None and mon.Path(result_file).exists()
    assert gt_file     is not None and mon.Path(gt_file).exists()

    coco_gt = COCO(str(gt_file))
    coco_dt = coco_gt.loadRes(str(result_file))

    print(type(coco_dt))
    print(type(coco_gt))

    imgIds = sorted(coco_gt.getImgIds())

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = imgIds
    # coco_eval.params.catIds = [0, 1, 2, 3, 4]  # class specified
    # coco_eval.params.maxDets[2] = len(imgIds)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print("----------------------------------------")
    print("AP_0.5-0.95: ", coco_eval.stats[0])
    print("AP_0.5     : ", coco_eval.stats[1])
    print("AP_S       : ", coco_eval.stats[3])
    print("AP_M       : ", coco_eval.stats[4])
    print("AP_L       : ", coco_eval.stats[5])
    print("f1_score   : ", coco_eval.stats[20])
    print("----------------------------------------")


if __name__ == "__main__":
    measure_metric()

# endregion
