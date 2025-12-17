#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for timer utilities.

This module implements a simple timer and time profiler classes for measuring
execution time of code segments.
"""

__all__ = [
    "Timer",
    "TimeProfiler",
]

import time


# ----- Timer -----
class Timer:
    """A simple timer.
    
    Attributes:
        start (float): Start time of the timer.
        end (float): End time of the timer.
        total (float): Total accumulated time.
        calls (int): Number of times the timer has been used.
        diff (float): Difference between end and start time.
        avg (float): Average time per call.
        duration (float): Duration of the last timing.
    """
    
    def __init__(self):
        self.start    = 0.0
        self.end      = 0.0
        self.total    = 0.0
        self.calls    = 0
        self.diff     = 0.0
        self.avg      = 0.0
        self.duration = 0.0
    
    @property
    def total_m(self) -> float:
        """Returns the total time in minutes.
        
        Returns:
            float: Total time in minutes.
        """
        return self.total / 60.0
    
    @property
    def total_h(self) -> float:
        """Returns the total time in hours.
        
        Returns:
            float: Total time in hours.
        """
        return self.total / 3600.0
    
    @property
    def avg_m(self) -> float:
        """Returns the average time in minutes.
        
        Returns:
            float: Average time in minutes.
        """
        return self.avg / 60.0
    
    @property
    def avg_h(self) -> float:
        """Returns the average time in hours.
        
        Returns:
            float: Average time in hours.
        """
        return self.avg / 3600.0
    
    @property
    def duration_m(self) -> float:
        """Returns the duration in minutes.
        
        Returns:
            float: Duration in minutes.
        """
        return self.duration / 60.0
    
    @property
    def duration_h(self) -> float:
        """Returns the duration in hours.
        
        Returns:
            float: Duration in hours.
        """
        return self.duration / 3600.0
    
    def start(self):
        """Starts the timer."""
        self.clear()
        self.tick()
    
    def end(self) -> float:
        """Ends the timer and returns the average time.
        
        Returns:
            float: Average time per call.
        """
        self.tock()
        return self.avg
    
    def tick(self):
        """Starts the timer."""
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start = time.time()
    
    def tock(self, average: bool = True) -> float:
        """Ends the timer and returns the duration.
        
        Args:
            average (bool): If True, returns the average time per call. If False,
                returns the duration of the last timing. Defaults to True.
        
        Returns:
            float: Duration or average time per call.
        """
        self.end    = time.time()
        self.diff   = self.end - self.start
        self.total += self.diff
        self.calls += 1
        self.avg    = self.total / self.calls
        if average:
            self.duration = self.avg
        else:
            self.duration = self.diff
        return self.duration
    
    def clear(self):
        """Clears the timer statistics."""
        self.start    = 0.0
        self.end      = 0.0
        self.total    = 0.0
        self.calls    = 0
        self.diff     = 0.0
        self.avg      = 0.0
        self.duration = 0.0


# ----- Time Profiler -----
class TimeProfiler:
    """A simple time profiler for measuring different stages of a process.
    
    Attributes:
        preprocess (Timer): Timer for the preprocessing stage.
        infer (Timer): Timer for the inference stage.
        postprocess (Timer): Timer for the postprocessing stage.
        total (Timer): Timer for the total process.
    """

    def __init__(self):
        self.preprocess  = Timer()
        self.infer       = Timer()
        self.postprocess = Timer()
        self.total       = Timer()

    @property
    def process_time(self) -> float:
        """Returns the average time taken by the profiler.
        
        Returns:
            float: Average process time.
        """
        return self.preprocess.total + self.infer.total + self.postprocess.total

    @property
    def avg_process_time(self) -> float:
        """Returns the average time taken by the profiler.
        
        Returns:
            float: Average process time.
        """
        return self.preprocess.avg + self.infer.avg + self.postprocess.avg

    def print(self):
        '''
        console.log(f"Total Time     : {self.total.total_time:09.6f} (s).")
        console.log(f"  - Preprocess : {self.preprocess.total_time:09.6f} (s).")
        console.log(f"  - Infer      : {self.infer.total_time:09.6f} (s).")
        console.log(f"  - Postprocess: {self.postprocess.total_time:09.6f} (s).")
        console.log(f"  - -----")
        console.log(f"  - Process    : {self.process_time:09.6f} (s).")
        '''

        results = {
            "Total"      : self.total.total,
            "Preprocess" : self.preprocess.total,
            "Infer"      : self.infer.total,
            "Postprocess": self.postprocess.total,
            "Process"    : self.process_time,
        }
        message = "           "
        # Headers
        for m, v in results.items():
            if v:
                message += f"{f'{m}':<10}\t"
        message += "\n           "
        # Values
        for i, (m, v) in enumerate(results.items()):
            if v:
                if i == len(results) - 1:
                    message += f"{v:<10.6f}\n"
                else:
                    message += f"{v:<10.6f}\t"
        print(f"{message}\n")
