#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Timer and profiling utility collection.

This module provides lightweight timer and profiling helpers.
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
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

class Timer:
    """Lightweight timer for measuring elapsed time.

    Accumulate total time, count calls, and provide statistics like per-call
    average and the duration of the last interval. Use as a context manager.

    Attributes:
        _name (str | None): Optional name for the timer instance.
            Defaults to None.
        start_time (float): Timestamp when the current interval started.
            Defaults to 0.0.
        end_time (float): Timestamp when the current interval ended.
            Defaults to 0.0.
        total (float): Accumulated total time across all intervals.
            Defaults to 0.0.
        calls (int): Number of timing intervals recorded. Defaults to 0.
        diff (float): Duration of the most recent interval. Defaults to 0.0.
        avg (float): Running average duration per call. Defaults to 0.0.
        duration (float): Last reported duration, which can be either the most
            recent interval's duration or the running average. Defaults to 0.0.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, name: str | None = None):
        """Initialize a new instance.

        Args:
            name: Optional name for the timer. Defaults to None.
        """
        self._name = name
        self._reset_stats()

    # --- Properties ---
    @property
    def name(self) -> str:
        """Return the name of the timer."""
        return self._name

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
            average: If True, the ``duration`` attribute is set to the running
                average. Otherwise, it is set to the duration of the most
                recent interval. Defaults to True.

        Returns:
            Resulting duration.
        """
        # Ensure all GPU operations are finished before stopping the clock.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except (ImportError, RuntimeError):
            pass

        self.end_time = time.perf_counter()
        self.diff     = self.end_time - self.start_time
        self.total   += self.diff
        self.calls   += 1
        self.avg      = self.total / self.calls
        self.duration = self.avg if average else self.diff
        return self.duration

    def reset(self):
        """Reset all timer statistics and start a new interval."""
        self._reset_stats()
        self.tick()

    def _reset_stats(self):
        """Initialize or reset statistics."""
        self.start_time: float = 0.0
        self.end_time  : float = 0.0
        self.total     : float = 0.0
        self.calls     : int   = 0
        self.diff      : float = 0.0
        self.avg       : float = 0.0
        self.duration  : float = 0.0


class TimeProfiler:
    """Profiler that holds timers for different stages of a pipeline.

    Provide separate ``Timer`` instances for preprocessing, inference,
    postprocessing, and the total time.

    Attributes:
        preprocess (Timer): ``Timer`` for the preprocessing stage.
        infer (Timer): ``Timer`` for the inference stage.
        postprocess (Timer): ``Timer`` for the postprocessing stage.
        total (Timer): ``Timer`` for the overall process.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self):
        """Initialize a new instance."""
        self.preprocess  = Timer(name="Preprocess")
        self.infer       = Timer(name="Infer")
        self.postprocess = Timer(name="Postprocess")
        self.total       = Timer(name="Total")

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
            (self.total.name,       self.total.total),
            (self.preprocess.name,  self.preprocess.total),
            (self.infer.name,       self.infer.total),
            (self.postprocess.name, self.postprocess.total),
            ("Process",             self.process_time),
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
        console.log(f"  - Postprocess: {self.postprocess.total_time:09.6f} (s).")
        console.log(f"  - -----")
        console.log(f"  - Process    : {self.process_time:09.6f} (s).")
        '''

        results = {
            self.total.name      : self.total.total,
            self.preprocess.name : self.preprocess.total,
            self.infer.name      : self.infer.total,
            self.postprocess.name: self.postprocess.total,
            "Process"            : self.process_time,
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
