#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Runtime debugging utilities.

This module provides debugging utilities at runtime.
"""

from __future__ import annotations

__all__ = [
    "print_run_summary",
]

import box

from mon.core.console import console, log, pprint_dict


# ==============================================================================
# region BASIC LOGGING
# ==============================================================================

def print_run_summary(args: dict | box.Box, full: bool = False):
    """Print a concise summary of run arguments.

    Print a compact run summary or the full configuration when requested.

    Args:
        args: Arguments mapping (box.Box or dict).
        full: If True, pretty-print the full args and config. Defaults to False.
    """
    # Handle Full Configuration Output
    if full:
        console.rule("[bold yellow]Full Configuration")
        # Ensure we have a standard dict for pretty printing
        printable_args = args.to_dict() if hasattr(args, "to_dict") else dict(args)
        pprint_dict(printable_args)
        return

    # Handle Concise Summary Output
    # We use .get() defaults to prevent crashes if certain keys are missing
    name = args.get("fullname", "Unnamed Run")
    console.rule(f"[bold red]{name}")
    summary_fields = {
        "Machine" : args.get("hostname", "local"),
        "Device"  : args.get("device", "cpu"),
        "Task"    : args.get("task"),
        "Mode"    : args.get("mode"),
        "Data"    : args.get("data"),
        "Weights" : args.get("weights"),
        "Save Dir": args.get("save_dir"),
        "Config"  : args.get("config"),
    }
    for label, value in summary_fields.items():
        if value:  # Only log fields that have a value
            # Formatting paths to be cleaner strings
            display_val = str(value) if not isinstance(value, list) else f"{len(value)} files"
            log(f"{label:<10}: {display_val}")

    console.rule() # Add a closing line for visual polish

# endregion


# ==============================================================================
# region VISUALIZATION
# ==============================================================================


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
