#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Timer and profiling utility collection.

This module provides lightweight timer and profiling helpers for measuring
elapsed time and profiling pipeline stages.
"""

__all__ = [
    "Timer",
    "TimeProfiler",
]

import time


# ==============================================================================
# CORE MEASUREMENT ENGINE
# ==============================================================================

# --- Atomic Timer (The Timer class for tracking intervals) ---
class Timer:
    """A lightweight timer for measuring elapsed time.

    Accumulate total time, count calls, and provide per-call average and last
    duration statistics.

    Attributes:
        start (float): Timestamp when the current interval started.
        end (float): Timestamp when the current interval ended.
        total (float): Accumulated total time across intervals.
        calls (int): Number of timing intervals recorded.
        diff (float): Duration of the most recent interval.
        avg (float): Running average duration per call.
        duration (float): Last reported duration (either last interval or
            average).
    """
    
    def __init__(self):
        """Initialize a new instance."""
        self.start    = 0.0
        self.end      = 0.0
        self.total    = 0.0
        self.calls    = 0
        self.diff     = 0.0
        self.avg      = 0.0
        self.duration = 0.0
    
    @property
    def total_m(self) -> float:
        """Return the total time in minutes."""
        return self.total / 60.0
    
    @property
    def total_h(self) -> float:
        """Return the total time in hours."""
        return self.total / 3600.0
    
    @property
    def avg_m(self) -> float:
        """Return the average time in minutes."""
        return self.avg / 60.0
    
    @property
    def avg_h(self) -> float:
        """Return the average time in hours."""
        return self.avg / 3600.0
    
    @property
    def duration_m(self) -> float:
        """Return the last duration in minutes."""
        return self.duration / 60.0
    
    @property
    def duration_h(self) -> float:
        """Return the last duration in hours."""
        return self.duration / 3600.0
    
    def start(self):
        """Start the timer and clear previous statistics.

        Reset accumulated statistics and record the start time.
        """
        self.clear()
        self.tick()
    
    def end(self) -> float:
        """End the timer and return the current average.

        Stop the current timing interval and return the average time per call.

        Returns:
            The current average duration per call.
        """
        self.tock()
        return self.avg
    
    def tick(self):
        """Record the start time for a timing interval.

        Begin a new timing segment without clearing previous data.
        """
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start = time.time()
    
    def tock(self, average: bool = True) -> float:
        """Stop the current timing interval and update statistics.

        Args:
            average: If True, set duration to the running average; otherwise set
                duration to the most recent interval.

        Returns:
            The resulting duration (either average or most recent interval).
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
        """Reset all timer statistics to zero.

        Clear start, end, total, calls, diff, avg, and duration values.
        """
        self.start    = 0.0
        self.end      = 0.0
        self.total    = 0.0
        self.calls    = 0
        self.diff     = 0.0
        self.avg      = 0.0
        self.duration = 0.0


# ==============================================================================
# PIPELINE INSTRUMENTATION
# ==============================================================================

# --- Stage Profiler ---
class TimeProfiler:
    """A profiler holding timers for different pipeline stages.

    Provide Timer instances for preprocess, infer, postprocess, and total
    measurements and helpers to compute aggregate process times.

    Attributes:
        preprocess (Timer): Timer for preprocessing stage.
        infer (Timer): Timer for inference stage.
        postprocess (Timer): Timer for postprocessing stage.
        total (Timer): Timer for the overall total stage.
    """

    def __init__(self):
        """Initialize a new instance."""
        self.preprocess  = Timer()
        self.infer       = Timer()
        self.postprocess = Timer()
        self.total       = Timer()

    @property
    def process_time(self) -> float:
        """Return the cumulative process time.

        Return the sum of preprocess, infer, and postprocess total times.
        """
        return self.preprocess.total + self.infer.total + self.postprocess.total

    @property
    def avg_process_time(self) -> float:
        """Return the cumulative average process time.

        Return the sum of preprocess, infer, and postprocess average times.
        """
        return self.preprocess.avg + self.infer.avg + self.postprocess.avg

    def print(self):
        """Print a formatted summary of collected timing statistics.

        Emit a simple tabular summary for total, preprocess, infer, and
        postprocess times to stdout.
        """
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
