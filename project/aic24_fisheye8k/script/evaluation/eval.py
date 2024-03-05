#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import click

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# region Main

@click.command(name="eval", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--result-file", type=click.Path(exists=True),  default=None, help="Result .json file.")
@click.option("--gt-file",     type=click.Path(exists=False), default=None, help="Groundtruth .json file.")
@click.option("--verbose",     is_flag=True)
def evaluate(
    result_file: str,
    gt_file    : str,
    verbose    : bool,
):
    assert result_file is not None
    assert gt_file     is not None

    coco_gt = COCO(str(gt_file))
    coco_dt = coco_gt.loadRes(str(result_file))

    print(type(coco_dt))
    print(type(coco_gt))

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    evaluate()

# endregion
