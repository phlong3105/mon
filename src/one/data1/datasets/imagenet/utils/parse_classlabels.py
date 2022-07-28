#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Parse `loc_synset_mapping.txt` to class_labels
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from munch import Munch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = str(Path(current_dir).parent)
file        = os.path.join(parent_dir, "det_synset_mapping.txt")

class_labels = []
colors      = []

with open(file) as f:
    for i, line in enumerate(f):
        words = line.split()
        color = list(np.random.choice(range(256), size=3))
        while color in colors:
            color = list(np.random.choice(range(256), size=3))
        colors.append(color)
        
        class_label = {
            "id"       : i,
            "synset_id": words[0],
            "name"     : line.replace(words[0], "").strip(),
            "color"    : color
        }
        class_label = Munch().fromDict(class_label)
        class_labels.append(class_label)


with open("class_labels.json", "w") as file_out:
    for l in class_labels:
        file_out.write(
            "{\"id\": \"" + f"{l.id}" + "\", \"synset_id\": \"" + f"{l.synset_id}" + "\", " +
            "\"name\": \"" + f"{l.name}" + "\", \"color\": [" + f"{l.color[0]}" + ", " + f"{l.color[1]}" + ", " + f"{l.color[2]}" + "]},\n"
        )
