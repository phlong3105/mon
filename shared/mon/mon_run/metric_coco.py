#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Measures COCO metrics for a given model and dataset."""

import argparse
import json
import logging

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import mon

mon.disable_print()

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- COCO -----
def measure_metric(input_json: mon.Path, target_json: mon.Path):
    if not input_json and not mon.Path(input_json).is_json_file(exist=True):
        raise FileNotFoundError(f"``input_json`` does not exist: {input_json}.")
    if not target_json or not mon.Path(target_json).is_json_file(exist=True):
        raise FileNotFoundError(f"``target_json`` does not exist: {target_json}.")

    coco_gt   = COCO(str(target_json))
    coco_dt   = coco_gt.loadRes(str(input_json))
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


def convert_label_to_coco(
    input_dir  : mon.Path,
    label_dir  : mon.Path,
    input_json : mon.Path,
    remap      : mon.Path,
    bbox_format: str,
) -> mon.Path:
    if not input_dir or not input_dir.is_dir():
        raise ValueError(f"``input_dir`` does not exist: {input_dir}.")
    if not label_dir or not label_dir.is_dir():
        raise ValueError(f"``label_dir`` does not exist: {label_dir}.")

    # Parse files
    if not (input_json and input_json.is_json_file(exist=False)):
        input_json = label_dir.parent / f"{label_dir.stem}.json"
    input_json.parent.mkdir(parents=True, exist_ok=True)

    if remap and remap.is_file():
        remap = mon.rt.load_config(config=remap)["remap"]
    else:
        remap = None

    if bbox_format != "coco":
        code = mon.BBoxFormat.from_value(value=f"{bbox_format}2coco")
    else:
        code = None
    
    # COCO JSON Format
    annotations = []
    image_files = sorted([f for f in list(input_dir.rglob("*")) if f.is_image_file()])
    with mon.create_progress_bar() as pbar:
        for i, image_file in pbar.track(
            sequence    = enumerate(image_files),
            total       = len(image_files),
            description = f"[bright_yellow]Converting"
        ):
            # Append image
            h, w, _  = mon.image.read_shape(image_file)
            image_id = i

            # Append annotations
            label_file = label_dir / f"{image_file.stem}.txt"
            if not label_file.is_txt_file():
                continue

            bs = mon.hbb.load_hbb(path=label_file, fmt=code, imgsz=(h, w))
            if len(bs) == 0:
                continue

            for b in bs:
                c = int(b[4])  # Class ID
                if remap:
                    if c in remap:
                        c = int(remap[c])
                    else:
                        continue
                annotations.append({
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
        json.dump(annotations, f, indent=None)

    return input_json


# ----- Main -----
def main(
    input_dir  : mon.Path,
    label_dir  : mon.Path,
    input_json : mon.Path,
    target_json: mon.Path,
    result_file: mon.Path,
    remap      : mon.Path,
    arch       : str,
    model      : str,
    data       : str,
    device     : int | list[int] | str,
    bbox_format: str,
    save_txt   : bool,
    exist_ok   : bool,
    verbose    : bool,
):
    if not verbose:
        logger = logging.getLogger()
        logger.disabled = True
    mon.console.rule(f"[bold red] {model}")
    mon.console.log(f"[bold green]Model: {model}")
    mon.console.log(f"[bold red]Data : {data}")
    mon.console.log(f"[bold]Device: {device}")

    input_dir   = mon.Path(input_dir)   if input_dir   else None
    label_dir   = mon.Path(label_dir)   if label_dir   else None
    input_json  = mon.Path(input_json)  if input_json  else None
    target_json = mon.Path(target_json) if target_json else None
    result_file = mon.Path(result_file) if result_file else None
    remap       = mon.Path(remap)       if remap       else None

    if not target_json or not target_json.is_json_file(exist=True):
        raise FileNotFoundError(f"``target_json`` does not exist: {target_json}.")

    if not exist_ok and input_json and input_json.is_json_file():
        input_json.unlink(missing_ok=True)
    if input_json and input_json.is_json_file():
        results = measure_metric(input_json=input_json, target_json=target_json)
    elif input_dir and input_dir.is_dir() and label_dir and label_dir.is_dir():
        input_json = convert_label_to_coco(
            input_dir   = input_dir,
            label_dir   = label_dir,
            input_json  = input_json,
            remap       = remap,
            bbox_format = bbox_format,
        )
        results = measure_metric(input_json=input_json, target_json=target_json)
    else:
        raise RuntimeError(
            f"Either ``input_json`` or [``input_dir`` and ``label_dir``] must be provided, got:\n"
            f"input_json: {input_json}\n"
            f"input_dir : {input_dir}\n"
            f"label_dir : {label_dir}"
        )

    # Show results
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="metric_coco")
    parser.add_argument("--input-dir",     type=str, help="Input image directory.")
    parser.add_argument("--label-dir",     type=str, help="Input label directory.")
    parser.add_argument("--input-json",    type=str, help="Input JSON file.")
    parser.add_argument("--target-json",   type=str, help="Ground-truth JSON file.")
    parser.add_argument("--result-file",   type=str, help="Result file.")
    parser.add_argument("--remap",         type=str, help="Classes re-map definition file.")
    parser.add_argument("--arch",          type=str, help="Model's architecture.")
    parser.add_argument("--model",         type=str, help="Model's fullname.")
    parser.add_argument("--data",          type=str, help="Source data name.")
    parser.add_argument("--device",        type=str, help="Running devices.")
    parser.add_argument("--bbox-format",   choices=["coco", "voc", "yolo"], default="yolo")
    parser.add_argument("--save-txt",      action="store_true")
    parser.add_argument("--exist-ok",      action="store_true")
    parser.add_argument("--verbose",       action="store_true")
    args = parser.parse_args()
    main(**vars(args))
