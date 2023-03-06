#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements IO functions."""

from __future__ import annotations

__all__ = [
    "AICAutoCheckoutWriter", "AIC22AutoCheckoutWriter",
    "AIC23AutoCheckoutWriter"
]

import os
from abc import ABC
from operator import itemgetter
from timeit import default_timer as timer

import mon
from supr import obj

console = mon.console


# region Writer

class AICAutoCheckoutWriter(ABC):
    """Save product counting results.
    
    Args:
        destination: A path to the counting results file.
        camera_name: A camera name.
        start_time: The moment when the TexIO is initialized.
        subset: A subset name. One of: ['testA', 'testB'].
        exclude: A list of class ID to exclude from writing. Defaults to [116].
    """
    
    video_map = {}
    
    def __init__(
        self,
        destination: mon.Path,
        camera_name: str,
        subset     : str       = "testA",
        exclude    : list[int] = [116],  # Just to be sure
        start_time : float     = timer(),
    ):
        super().__init__()
        if subset not in self.video_map:
            raise ValueError(
                f"subset must be a valid key in video_map "
                f"({self.video_map.keys()}), but got {subset}."
            )
        if camera_name not in self.video_map[subset]:
            raise ValueError(
                f"camera_name must be a valid key in video_map[subset] "
                f"({self.video_map[subset].keys()}), but got {camera_name}."
            )
        
        self.destination = mon.Path(destination)
        self.camera_name = camera_name
        self.subset      = subset
        self.video_id    = self.video_map[subset][camera_name]
        self.exclude     = exclude
        self.start_time  = start_time
        self.results     = []
        self.init_writer()
    
    def __del__(self):
        """ Close the writer object."""
        pass
    
    def init_writer(self, destination: mon.Path | None = None):
        """Initialize the writer object.

        Args:
            destination: A path to the counting results file.
        """
        destination = destination or self.destination
        destination = mon.Path(destination)
        if destination.is_basename() or destination.is_stem():
            destination = f"{destination}.txt"
        elif destination.is_dir():
            destination = destination / f"{self.camera_name}.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.destination = destination
    
    def append_results(self, products: list[obj.Product]):
        """Write counting results.

        Args:
            products: A list of tracking :class:`data.Product` objects.
        """
        for p in products:
            class_id = p.majority_label_id
            if class_id in self.exclude:
                continue
            self.results.append((self.video_id, class_id + 1, int(p.timestamp)))
    
    def write_to_file(self):
        """Dump all content in :attr:`lines` to :attr:`output` file."""
        if not self.destination.is_txt_file(exist=False):
            self.init_writer()
        
        with open(self.destination, "w") as f:
            prev_id = 0
            for r in self.results:
                video_id  = r[0]
                class_id  = r[1]
                timestamp = r[2]
                if prev_id == class_id:
                    continue
                prev_id = class_id
                f.write(f"{video_id} {class_id } {timestamp}\n")
    
    @classmethod
    def merge_results(
        cls,
        output_dir : mon.Path | None = None,
        output_name: str | None      = "track4.txt",
        subset     : str             = "testA"
    ):
        """Merge all cameras' result files into one file.
        
        Args:
            output_dir: A directory to store the :attr:`output_name`.
            output_name: A result file name. Defaults to 'track4.txt'.
            subset: A subset name. One of: ['testA', 'testB'].
        """
        if subset not in cls.video_map:
            raise ValueError(
                f"subset must be a valid key in video_map "
                f"({cls.video_map.keys()}), but got {subset}."
            )
        
        output_dir      = output_dir or mon.Path().absolute()
        output_dir      = mon.Path(output_dir)
        output_name     = output_name or "track4"
        output_name     = mon.Path(output_name).stem
        output_name     = output_dir / f"{output_name}.txt"
        compress_writer = open(output_name, "w")
        
        # NOTE: Get results from each file
        for v_name, v_id in cls.video_map[subset].items():
            video_result_file = os.path.join(output_dir, f"{v_name}.txt")
            
            if not os.path.exists(video_result_file):
                console.log(f"Result of {video_result_file} does not exist!")
                continue
            
            # NOTE: Read result
            results = []
            with open(video_result_file) as f:
                line = f.readline()
                while line:
                    words  = line.split(" ")
                    result = {
                        "video_id" : int(words[0]),
                        "class_id" : int(words[1]),
                        "timestamp": int(words[2]),
                    }
                    if result["class_id"] != 116:
                        results.append(result)
                    line = f.readline()
            
            # NOTE: Sort result
            results = sorted(results, key=itemgetter("video_id"))
            
            # NOTE: write result
            for result in results:
                compress_writer.write(f"{result['video_id']} ")
                compress_writer.write(f"{result['class_id']} ")
                compress_writer.write(f"{result['timestamp']} ")
                compress_writer.write("\n")
        
        compress_writer.close()


class AIC22AutoCheckoutWriter(AICAutoCheckoutWriter):
    """Save product checkout results for AIC22 Multi-Class Product Counting &
    Recognition for Automated Retail Checkout.
    """
    
    video_map = {
        "testA": {
            "testA_1": 1,
            "testA_2": 2,
            "testA_3": 3,
            "testA_4": 4,
            "testA_5": 5,
        },
        "testB": {
            "testB_1": 1,
            "testB_2": 2,
            "testB_3": 3,
            "testB_4": 4,
            "testB_5": 5,
        },
    }


class AIC23AutoCheckoutWriter(AICAutoCheckoutWriter):
    """Save product checkout results for AIC23 Multi-Class Product Counting &
    Recognition for Automated Retail Checkout.
    """
    
    video_map = {
        "testA": {
            "testA_1": 1,
            "testA_2": 2,
            "testA_3": 3,
            "testA_4": 4,
        },
        "testB": {
            "testB_1": 1,
            "testB_2": 2,
            "testB_3": 3,
            "testB_4": 4,
        },
    }
    

# endregion
