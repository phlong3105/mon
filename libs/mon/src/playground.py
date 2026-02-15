#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Playground."""

import os.path
import pathlib
import torch
from enum import StrEnum
import mon

device_manager = mon.DeviceManager()
print(device_manager.devices)
