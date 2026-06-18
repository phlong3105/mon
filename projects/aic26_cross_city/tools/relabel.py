#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Relabel Bounding Boxes.

This script provides a CLI for relabeling bounding boxes.
"""

from __future__ import annotations

__all__ = []

import argparse
import json
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from ultralytics import YOLO

from mon.core import (
    BBox,
    BBoxes,
    BBoxFormat,
    create_progress_bar,
    Path,
    resolve_project_root,
)
from mon.ops import read_imgsz, write_bbox

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]
project_root: Path = resolve_project_root(current_dir)
data_dir = project_root / "data" / "aic26_cross_city"

DEVICE = "cuda"
CLASSES = [
    "Car",
    "Pickup Truck",
    "Single Truck",
    "Combo Truck",
    "Heavy Duty Vehicle",
    "Trailer",
    "Motorcycle",
    "Bicycle",
    "Van",
    "Bus",
    "Person"
]
DESCRIPTIONS = [
    "Car: A sedan, hatchback, coupe, SUV, or police car driving on the road.",
    "Pickup Truck: A light duty utility truck with a distinct open cargo bed in the back.",
    "Single Truck: A large rigid single-unit cargo vehicle, delivery box truck, flatbed truck, or single-frame commercial truck without any trailer attached.",
    "Combo Truck: A large articulated semi-truck, tractor-trailer combination truck, big rig, or multi-axle commercial shipping truck pulling an attached trailer container.",
    "Heavy Duty Vehicle: A large specialized industrial vehicle on the road, including dump trucks, cement mixers, garbage trucks, road tractors, cranes, or construction machinery.",
    "Trailer: A non-motorized cargo utility trailer, container chassis, or vehicle transport trailer being towed behind another vehicle.",
    "Motorcycle: A motorized two-wheeled vehicle, motor scooter, or moped.",
    "Bicycle: A two-wheeled pedal bicycle or electric e-bike.",
    "Van: A large boxy passenger minivan, commercial cargo transit van, or step van.",
    "Bus: A large passenger transit bus, school bus, coach bus, or articulated public city bus.",
    "Person: A pedestrian walking on the street, a driver sitting inside a vehicle, or a human individual visible in a traffic camera frame."
]
SCHEMA_CONTEXT_PROMPT = """You are an advanced traffic video annotation engine. 
Your task is to detect ALL objects that belong to the following categories in the image. 

Class definitions:
- Car: A standard passenger vehicle primarily designed to transport up to 5–7 passengers. Includes sedans, hatchbacks, coupes, SUVs, and police cars.
- Pickup Truck: A light-duty vehicle with an enclosed cabin, and an open rear cargo bed.
- Single Truck: A rigid single-frame commercial truck, box truck, or delivery truck without a trailer.
- Combo Truck: An articulated semi-truck consisting of a tractor unit pulling one or more trailer units.
- Heavy Duty Vehicle: An Industrial machinery, construction vehicles, dump trucks, cement mixers, or road tractors.
- Trailer: A non-motorized towed utility or container trailers.
- Motorcycle: A motorized two-wheelers (scooters, moped). Do not include the rider's upper body inside the box.
- Bicycle: A human-powered or electric pedal bikes. Do not include the rider's body.
- Van: A medium-sized vehicle characterized by a box-shaped body designed for transporting goods or groups of people.
- Bus: A large passenger transit vehicles, school buses, or articulated transit city buses.
- Person: Pedestrians, drivers, riders, or humans visible in the scene (including real people featured on physical billboards/advertisements). Do not label statues or sculptures.

Output Format:
You must respond strictly with a valid JSON object matching this array layout, containing no conversational explanation text:
{"detections": [{"box_2d": [x1, y1, x2, y2], "label": "class_name"}]}"""


# ==============================================================================
# region FUNCTIONS
# ==============================================================================

def relabel_yolo_world(args: argparse.Namespace):
    """Uses Ultralytics YOLO-World to perform open-vocabulary grounding on raw
    image streams.
    """
    # Resolve paths
    images_dir = data_dir / args.data / "images"

    output_dir = data_dir / args.data / f"{args.label_dir}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Retrieve images
    image_files: list[Path] = sorted(images_dir.glob("*"))
    image_files = [f for f in image_files if f.is_image_file(exists=True)]

    # Initialize YOLO-World model (using the large or extra-large variant for
    # best zero-shot detail)
    model = YOLO("yolov8x-world.pt")
    model.set_classes(CLASSES)

    # Loop through each image
    with create_progress_bar() as pbar:
        for i, image_file in pbar.track(
            sequence=enumerate(image_files),
            total=len(image_files),
            description="[bright_yellow]Relabeling",
        ):
            imgsz = read_imgsz(path=image_file)

            # Execute model inference pass (disable plotting to speed up calculations)
            results = model.predict(str(image_file), conf=args.conf, verbose=False)[0]
            boxes = results.boxes.xywhn.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()

            # Convert bboxes
            bboxes = []
            for j in range(len(boxes)):
                bboxes.append(
                    BBox(
                        class_id=int(clss[j]),
                        bbox=boxes[j],
                        angle=0.0,
                        score=float(scores[j]),
                        track_id=-1,
                        imgsz=imgsz,
                    )
                )
            bboxes = BBoxes(bboxes=bboxes, imgsz=imgsz)

            # Save
            relabel_file = output_dir / f"{image_file.stem}.txt"
            write_bbox(bbox=bboxes, path=relabel_file, fmt=BBoxFormat.CXCYWHN)


def relabel_qwen(args: argparse.Namespace):
    """Uses Qwen3-VL to perform instruction-guided open-vocabulary grounding."""
    # Resolve paths
    images_dir = data_dir / args.data / "images"

    output_dir = data_dir / args.data / f"{args.label_dir}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Retrieve images
    image_files: list[Path] = sorted(images_dir.glob("*"))
    image_files = [f for f in image_files if f.is_image_file(exists=True)]

    # Initialize Qwen3-VL
    model_id = "Qwen/Qwen3-VL-32B-Instruct"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    # Loop through each image
    with create_progress_bar() as pbar:
        for i, image_file in pbar.track(
            sequence=enumerate(image_files),
            total=len(image_files),
            description="[bright_yellow]Relabeling",
        ):
            image = Image.open(image_file).convert("RGB")
            imgsz = read_imgsz(path=image_file)
            H, W = imgsz.hw

            # Build conversation stack using standard Hugging Face messages schema format
            prompt_content = f"{SCHEMA_CONTEXT_PROMPT}\nAnalyze this traffic image and extract the JSON payload now:"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_content}
                    ]
                }
            ]
            # Process prompt components into inputs tensors text format
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=image, padding=True, return_tensors="pt").to(DEVICE)

            try:
                # Forward inference generation run block without tracking gradients
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
                    # Trim down prompt context padding to isolate new tokens text responses block
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids
                        in zip(inputs.input_ids, generated_ids)
                    ]
                    response = processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]
                # Strip out markdown formatting block wraps if returned by text head parser
                clean_json_str = response.strip()
                if "```json" in clean_json_str:
                    clean_json_str = clean_json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_json_str:
                    clean_json_str = clean_json_str.split("```")[1].split("```")[0].strip()
                result_payload = json.loads(clean_json_str)
            except Exception as e:
                # Fallback to empty container to prevent breaking batch trace executions
                result_payload = {"detections": []}
                print(e)

            # Convert bboxes
            bboxes = []
            for det in result_payload.get("detections", []):
                label  = det.get("label")
                box_2d = det.get("box_2d")  # Returns [x1, y1, x2, y2] (0-1000 scale)

                if label not in CLASSES or not box_2d or len(box_2d) != 4:
                    continue

                class_id = CLASSES.index(label)
                x1, y1, x2, y2 = box_2d

                # Step A: Scale 0-1000 coordinates up into absolute image canvas pixel coordinates
                x1_px = (x1 / 1000.0) * W
                y1_px = (y1 / 1000.0) * H
                x2_px = (x2 / 1000.0) * W
                y2_px = (y2 / 1000.0) * H

                # Step B: Convert absolute dimensions to your localized normalized CXCYWH model structure format
                cx = (x1_px + x2_px) / 2.0 / W
                cy = (y1_px + y2_px) / 2.0 / H
                w_norm = max(0.0, (x2_px - x1_px) / W)
                h_norm = max(0.0, (y2_px - y1_px) / H)

                # Construct custom BBox instance matching your format parameters
                bboxes.append(
                    BBox(
                        class_id=class_id,
                        bbox=np.array([cx, cy, w_norm, h_norm]),
                        angle=0.0,
                        score=1.0,
                        track_id=-1,
                        imgsz=imgsz,
                    )
                )
            bboxes = BBoxes(bboxes=bboxes, imgsz=imgsz)

            # Save
            relabel_file = output_dir / f"{image_file.stem}.txt"
            write_bbox(bbox=bboxes, path=relabel_file, fmt=BBoxFormat.CXCYWHN)

# endregion


# ==============================================================================
# region MAIN
# ==============================================================================

def main(args: argparse.Namespace):
    """The main function."""
    if args.yolo:
        relabel_yolo_world(args)
    elif args.qwen:
        relabel_qwen(args)
    else:
        print("No valid action specified. Use --yolo or --qwen.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("main")
    parser.add_argument("--data", type=str, default="x/eccv_cross_city", help="Data source directory.")
    parser.add_argument("--label-dir", type=str, default="relabel", help="Relabel directory.")
    parser.add_argument("--conf", type=float, default=0.10, help="Confidence threshold.")
    parser.add_argument("--yolo", action="store_true", help="Use YOLO-World for relabeling.")
    parser.add_argument("--qwen", action="store_true", help="Use Qwen for relabeling.")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


if __name__ == "__main__":
    main(parse_args())

# endregion
