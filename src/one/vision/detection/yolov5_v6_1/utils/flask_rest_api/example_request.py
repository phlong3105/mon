#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Perform test request.
"""

from __future__ import annotations

import pprint

import requests

DETECTION_URL = "http://localhost:5000/v1/object-detection/yolov5s"
TEST_IMAGE    = "zidane.jpg"
image_data    = open(TEST_IMAGE, "rb").read()
response      = requests.post(DETECTION_URL, files={"image": image_data}).json()
pprint.pprint(response)
