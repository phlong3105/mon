#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Timer.

This module provides data structures for timing and profiling code execution.
"""

from __future__ import annotations

__all__ = [
    "Timer",
    "TimeProfiler",
]

import time

from rich.table import Table

from mon.core.console import console


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class Timer:
    """Lightweight timer for measuring elapsed time.

    Accumulate total time, count calls, and provide statistics like the per-call
    average and the duration of the last interval. Use as a context manager.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, name: str = ""):
        """Initialize a new instance.

        Args:
            name (str, optional): Optional name for the timer instance.
                Defaults to "".
        """
        # Assign attributes
        self.name = name
        self.start_time = 0.0
        self.end_time = 0.0
        self.total = 0.0
        self.calls = 0
        self.diff = 0.0
        self.avg = 0.0
        self.duration = 0.0

    # --- Properties ---
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
        """Return the last reported duration in minutes."""
        return self.duration / 60.0

    @property
    def duration_h(self) -> float:
        """Return the last reported duration in hours."""
        return self.duration / 3600.0

    # --- Callable & Context Manager ---
    def __enter__(self):
        """Start the timer when entering a context."""
        self.tick()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer when exiting a context."""
        self.tock()

    # --- Public Methods ---
    def start(self):
        """Start a new timing interval."""
        self.tick()

    def end(self) -> float:
        """End the current timing interval and return the average time.

        Returns:
            Average time.
        """
        self.tock()
        return self.avg

    def tick(self):
        """Record the start time of an interval."""
        self.start_time = time.perf_counter()

    def tock(self, average: bool = True) -> float:
        """Record the end time of an interval and update statistics.

        Args:
            average (bool, optional): If True, return the average duration of
                the last intervals. If False, return the duration of the most
                recent interval. Defaults to True.

        Returns:
            float: Duration of the last interval.
        """
        # Ensure all GPU operations are finished before stopping the clock.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except (ImportError, RuntimeError):
            pass

        self.end_time = time.perf_counter()
        self.diff = self.end_time - self.start_time
        self.total += self.diff
        self.calls += 1
        self.avg = self.total / self.calls
        self.duration = self.avg if average else self.diff
        return self.duration

    def restart(self):
        """Reset all timer statistics and start a new interval."""
        self.reset()
        self.tick()

    def reset(self):
        """Reset all timer statistics."""
        self.start_time = 0.0
        self.end_time = 0.0
        self.total = 0.0
        self.calls = 0
        self.diff = 0.0
        self.avg = 0.0
        self.duration = 0.0


class TimeProfiler:
    """Profiler that holds timers for different stages of a pipeline.

    Provide separate ``Timer`` instances for preprocessing, inference,
    postprocessing, and the total time.

    Attributes:
        preprocess (Timer): Timer for the preprocessing stage.
        infer (Timer): Timer for the inference stage.
        postprocess (Timer): Timer for the postprocessing stage.
        total (Timer): Timer for the total time.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self):
        """Initialize a new instance."""
        self.preprocess = Timer(name="Preprocess")
        self.infer = Timer(name="Infer")
        self.postprocess = Timer(name="Postprocess")
        self.total = Timer(name="Total")

    # --- Properties ---
    @property
    def process_time(self) -> float:
        """Return the cumulative process time."""
        return self.preprocess.total + self.infer.total + self.postprocess.total

    @property
    def avg_process_time(self) -> float:
        """Return the cumulative average process time."""
        return self.preprocess.avg + self.infer.avg + self.postprocess.avg

    # --- Public Methods ---
    def print(self):
        """Print a formatted summary of the timing statistics."""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Stage", style="dim")
        table.add_column("Time (s)", justify="right")

        # Define the order of timers to be printed
        timers_to_print = [
            (self.total.name, self.total.total),
            (self.preprocess.name, self.preprocess.total),
            (self.infer.name, self.infer.total),
            (self.postprocess.name, self.postprocess.total),
            ("Process", self.process_time),
        ]

        for name, time_val in timers_to_print:
            if time_val > 0:
                table.add_row(name, f"{time_val:.6f}")

        console.log(table)

    def print_copy(self):
        """Print a formatted summary of the timing statistics for easier
        copy-paste to other applications (e.g., Excel).
        """
        '''
        console.log(f"Total Time     : {self.total.total_time:09.6f} (s).")
        console.log(f"  - Preprocess : {self.preprocess.total_time:09.6f} (s).")
        console.log(f"  - Infer      : {self.infer.total_time:09.6f} (s).")
        console.log(f"  - Postprocess: {self.postprocess.total_time:09.6f} (
        s).")
        console.log(f"  - -----")
        console.log(f"  - Process    : {self.process_time:09.6f} (s).")
        '''

        results = {
            self.total.name: self.total.total,
            self.preprocess.name: self.preprocess.total,
            self.infer.name: self.infer.total,
            self.postprocess.name: self.postprocess.total,
            "Process": self.process_time,
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

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
