#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements a simple timer and time profiler classes for measuring
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
    
    @property
    def total_time_m(self) -> float:
        return self.total_time / 60.0
    
    @property
    def total_time_h(self) -> float:
        return self.total_time / 3600.0
    
    @property
    def avg_time_m(self) -> float:
        return self.avg_time / 60.0
    
    @property
    def avg_time_h(self) -> float:
        return self.avg_time / 3600.0
    
    @property
    def duration_m(self) -> float:
        return self.duration / 60.0
    
    @property
    def duration_h(self) -> float:
        return self.duration / 3600.0
    
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


# ----- Time Profiler -----
class TimeProfiler:
    """A simple timer profiler for measuring the time taken by different parts
    of a process.
    """

    def __init__(self):
        self.preprocess  = Timer()
        self.infer       = Timer()
        self.postprocess = Timer()
        self.total       = Timer()

    @property
    def process_time(self) -> float:
        """Returns the average time taken by the profiler."""
        return self.preprocess.total_time + self.infer.total_time + self.postprocess.total_time

    @property
    def avg_process_time(self) -> float:
        """Returns the average time taken by the profiler."""
        return self.preprocess.avg_time + self.infer.avg_time + self.postprocess.avg_time

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
            "Total"      : self.total.total_time,
            "Preprocess" : self.preprocess.total_time,
            "Infer"      : self.infer.total_time,
            "Postprocess": self.postprocess.total_time,
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
