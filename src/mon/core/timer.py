#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements a simple timer object."""

from __future__ import annotations

__all__ = [
    "Timer",
]

import time


class Timer:
    """A simple timer.
    
    Attributes:
        start_time: The start time of the current call.
        end_time: The end time of the current call.
        total_time: The total time of the timer.
        calls: The number of calls.
        diff_time: The difference time of the call.
        avg_time: The total average time.
    """
    
    def __init__(self):
        self.start_time = 0.0
        self.end_time   = 0.0
        self.total_time = 0.0
        self.calls      = 0
        self.diff_time  = 0.0
        self.avg_time   = 0.0
        self.duration   = 0.0
    
    def start(self):
        self.clear()
        self.tick()
    
    def end(self) -> float:
        self.tock()
        return self.avg_time
    
    def tick(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()
    
    def tock(self, average: bool = True) -> float:
        self.end_time    = time.time()
        self.diff_time   = self.end_time - self.start_time
        self.total_time += self.diff_time
        self.calls      += 1
        self.avg_time    = self.total_time / self.calls
        if average:
            self.duration = self.avg_time
        else:
            self.duration = self.diff_time
        return self.duration
    
    def clear(self):
        self.start_time = 0.0
        self.end_time   = 0.0
        self.total_time = 0.0
        self.calls      = 0
        self.diff_time  = 0.0
        self.avg_time   = 0.0
        self.duration   = 0.0
