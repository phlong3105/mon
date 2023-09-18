#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

import mon


class BaseParser:
    
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def parse(self):
        self.parser.add_argument("--data",       type=str, default="./data/test/*")
        self.parser.add_argument("--weights",    type=str, default="./checkpoints")
        self.parser.add_argument("--config",     type=str, default="./config.yaml", help="path to config")
        self.parser.add_argument("--image-size", type=int, default=512)
        self.parser.add_argument("--mode",       type=str, default="test", choices=["train", "test"])
        self.parser.add_argument("--output-dir", type=str, default=mon.RUN_DIR/"predict/kind")
        return self.parser.parse_args()
