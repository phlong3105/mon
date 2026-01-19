#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""COCO metric evaluator.

This module provides a runner for measuring COCO metrics.
"""

from __future__ import annotations

__all__ = [
    "COCOEvaluator",
]

import argparse
import json
import logging

import box
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mon.core import (
    bbox as B,
    BBoxFormat,
    console,
    create_device,
    create_progress_bar,
    image as I,
    load_config,
    Path,
)

current_file = Path(__file__).normalize(exist=True)
current_dir  = current_file.parents[0]


# ==============================================================================
# region CONTROL
# ==============================================================================

class COCOEvaluator:
    """A runner for measuring COCO metrics."""

    # --- Lifecycle & Initialization ---
    def __init__(self, cfg: box.Box):
        # Assign attributes
        self._cfg         = cfg
        self.verbose      = cfg.verbose
        self._arch        = cfg.arch
        self._model       = cfg.model
        self._data        = cfg.data
        self._device      = create_device(cfg.device)
        self._bbox_format = cfg.bbox_format
        self._save_txt    = cfg.save_txt
        self._exist_ok    = cfg.exist_ok

        # Resolve paths
        if cfg.input_json:
            self._input_json = Path(cfg.input_json).normalize(exist=True)
        elif cfg.input_dir and cfg.label_dir:
            self._input_dir  = Path(cfg.input_dir).normalize(exist=True)
            self._label_dir  = Path(cfg.label_dir).normalize(exist=True)
            self._input_json = self._label_dir.parent / f"{self._label_dir.stem}.json"
        else:
            raise ValueError(
                f"Either 'input_json' or ('input_dir' and 'label_dir') must be specified, but got: "
                f"input_json={cfg.input_json}, "
                f"input_dir={cfg.input_dir}, "
                f"label_dir={cfg.label_dir}."
            )

        if cfg.target_json:
            self._target_json = Path(cfg.target_json).normalize(exist=True)
        else:
            raise FileNotFoundError(f"Target JSON file not found: {cfg.target_json}.")

        self._remap = Path(cfg.remap).normalize(exist=True) if cfg.remap else None

        # Initialize states
        self._results = {}

    # --- Callable & Context Manager ---
    def run(self):
        """Run the metric measurement process."""
        # Summarize the current run
        if not self.verbose:
            logger = logging.getLogger()
            logger.disabled = True
        console.rule(f"[bold red] {self._model}")
        console.log(f"[bold green]Model : {self._model}")
        console.log(f"[bold red]Data  : {self._data}")
        console.log(f"[bold]Device: {self._device}")

        # Convert label files to a COCO JSON file
        if not self._exist_ok:
            self._input_json.unlink(missing_ok=True)
        if not self._input_json.is_json_file(exist=True):
            self._convert_label_to_coco()

        # Processing
        self._results = self._measure()

        # Print results
        self._print_results()

    def _measure(self) -> dict:
        """Measure the COCO metrics."""
        coco_gt   = COCO(str(self._target_json))
        coco_dt   = coco_gt.loadRes(str(self._input_json))
        imgIds    = sorted(coco_gt.getImgIds())

        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = imgIds
        # coco_eval.params.catIds = [1]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        results = {
            "AP"    : coco_eval.stats[0],
            "AP50"  : coco_eval.stats[1],
            "AP75"  : coco_eval.stats[2],
            "APs"   : coco_eval.stats[3],
            "APm"   : coco_eval.stats[4],
            "APl"   : coco_eval.stats[5],
            "AR@1"  : coco_eval.stats[6],
            "AR@10" : coco_eval.stats[7],
            "AR@100": coco_eval.stats[8],
            "ARs"   : coco_eval.stats[9],
            "ARm"   : coco_eval.stats[10],
            "ARl"   : coco_eval.stats[11],
        }
        return results

    def _convert_label_to_coco(self):
        """Convert label files to a COCO JSON file."""
        input_dir   = self._input_dir
        label_dir   = self._label_dir
        input_json  = self._input_json
        bbox_format = self._bbox_format
        remap       = self._remap

        # Create remap dictionary
        if remap and remap.is_file(exist=True):
            remap = load_config(config=remap)["remap"]
        else:
            remap = None

        # Determine bbox format
        if bbox_format != "coco":
            code = BBoxFormat(value=f"{bbox_format}2coco")
        else:
            code = None

        # Processing loop
        image_files = sorted([f for f in list(input_dir.rglob("*")) if f.is_image_file()])
        labels      = []
        with create_progress_bar() as pbar:
            for i, image_file in pbar.track(
                sequence    = enumerate(image_files),
                total       = len(image_files),
                description = f"[bright_yellow]Converting"
            ):
                # Append image
                h, w     = I.read_size(image_file)
                image_id = i

                # Append annotations
                label_file = label_dir / f"{image_file.stem}.txt"
                if not label_file.is_txt_file(exist=True):
                    continue

                bs = B.load(path=label_file, fmt=code, imgsz=(h, w))
                if len(bs) == 0:
                    continue

                for b in bs:
                    c = int(b[5])  # Class ID
                    if remap:
                        if c in remap:
                            c = int(remap[c])
                        else:
                            continue
                    labels.append({
                        "image_id"   : image_id,
                        "category_id": int(c),
                        "bbox"       : [
                            round(float(b[0]), 32),
                            round(float(b[1]), 32),
                            round(float(b[2]), 32),
                            round(float(b[3]), 32)
                        ],
                        "score"      : float(b[5]),
                    })

        # Write to JSON file
        with open(str(input_json), "w") as f:
            json.dump(labels, f, indent=None)

    def _print_results(self):
        """Print the measured results."""
        results = self._results

        message = ""
        # Headers
        for m, v in results.items():
            if v:
                message += f"{f'{m}':<10}\t"
        message += "\n"
        # Values
        for i, (m, v) in enumerate(results.items()):
            if v:
                if i == len(results) - 1:
                    message += f"{v:.10f}\n"
                else:
                    message += f"{v:.10f}\t"
        print(f"{message}\n")

    # --- CLI ---
    @staticmethod
    def parse_args() -> box.Box:
        """Parse command line arguments.

        Returns:
            Parsed arguments.
        """
        parser = argparse.ArgumentParser(description="metric_coco")
        parser.add_argument("--input-dir",   type=str, help="Input image directory.")
        parser.add_argument("--label-dir",   type=str, help="Input label directory.")
        parser.add_argument("--input-json",  type=str, help="Input JSON file.")
        parser.add_argument("--target-json", type=str, help="Ground-truth JSON file.")
        parser.add_argument("--result-file", type=str, help="Result file.")
        parser.add_argument("--remap",       type=str, help="Classes re-map definition file.")
        parser.add_argument("--arch",        type=str, help="Model's architecture.")
        parser.add_argument("--model",       type=str, help="Model's fullname.")
        parser.add_argument("--data",        type=str, help="Source data name.")
        parser.add_argument("--device",      type=str, help="Running devices.")
        parser.add_argument("--bbox-format", choices=["coco", "voc", "yolo"], default="yolo")
        parser.add_argument("--save-txt",    action="store_true")
        parser.add_argument("--exist-ok",    action="store_true")
        parser.add_argument("--verbose",     action="store_true")
        return box.Box(vars(parser.parse_args()))

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    args   = COCOEvaluator.parse_args()
    runner = COCOEvaluator(args)
    runner.run()

# endregion
