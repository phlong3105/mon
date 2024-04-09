#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

from __future__ import annotations

__all__ = [
    "parse_menu_string",
]

from typing import Collection, Sequence


# region Parsing

def parse_menu_string(items: Sequence | Collection, num_columns: int = 4) -> str:
    s = f"\n  "
    for i, item in enumerate(items):
        s += f"{f'{i}.':>6} {item}\n  "
    s += f"{f'Other.':} (please specify)\n  "
    
    '''
    w, h = mon.get_terminal_size()
    w 	 = w if w >= 80 else 80
    items_per_row = w // (padding + 2)
    padding = math.floor(w / num_columns) - 8
    
    s   = f"\n  "
    row = f""
    for i, item in enumerate(items):
        if i > 0 and i % num_columns == 0:
            s   += f"{row}\n\t"
            row  = f""
        else:
            t    = f"{f'{i}.':>4}{item}"
            row += f"{t:<{padding}}"
    if row != "":
        s += f"{row}\n\t"
    '''
    
    return s

# endregion
