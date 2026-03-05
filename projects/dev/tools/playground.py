#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Playground."""
import cv2

# noinspection PyUnusedImports
import mon
from mon import Path

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]

_, dicm = mon.dataset.build_dataset(
    src="dicm",
    dataset_dir=current_dir / "data",
    split="test"
)
print(dicm)
print(dicm.modalities)
print(dicm.root)

for data in dicm:
    image = data["image"]
    depth = data["depth"]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image", image)
    cv2.imshow("Depth", depth)
    cv2.waitKey(0)
    print(image.shape, depth.shape)
