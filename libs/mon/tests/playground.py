#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Playground."""
import math
import os

# noinspection PyUnusedImports
import mon
from mon import Path

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]

from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl, BufferControl
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.styles import Style


class MultiSelect:
    def __init__(
        self,
        options:     list[str] | None             = None,
        title:       str                          = "Select options:",
        defaults:    list[str] | list[int] | None = None,
        multiselect: bool                         = True,
        strict:      bool                         = False,
    ):
        self.options       = options
        self.title         = title
        self.multiselect   = multiselect
        self.strict        = strict
        self.selected      = set()
        self.current       = 0
        self.error_msg     = ""
        self.custom_buffer = Buffer()

        if self.options is not None and defaults:
            options_lower = [o.lower() for o in self.options]
            _defaults = defaults[:1] if not multiselect else defaults
            for d in _defaults:
                if isinstance(d, int):
                    if 0 <= d < len(self.options):
                        self.selected.add(d)
                elif isinstance(d, str):
                    if d.isdigit():
                        idx = int(d) - 1
                        if 0 <= idx < len(self.options):
                            self.selected.add(idx)
                    else:
                        try:
                            self.selected.add(options_lower.index(d.lower()))
                        except ValueError:
                            pass

    # ------------------------------------------------------------------ #
    #  Parsing                                                             #
    # ------------------------------------------------------------------ #

    def parse_input(self, text: str) -> tuple[set, list]:
        indices       = set()
        custom        = []
        options_lower = [o.lower() for o in self.options] if self.options else []

        for part in text.split(","):
            part = part.strip()
            if not part:
                continue

            if self.options is None:
                custom.append(part)
            elif part.isdigit():
                idx = int(part) - 1
                if 0 <= idx < len(self.options):
                    indices.add(idx)
            else:
                try:
                    idx = options_lower.index(part.lower())
                    indices.add(idx)
                except ValueError:
                    if not self.strict:
                        custom.append(part)

        if not self.multiselect:
            indices = {max(indices)} if indices else set()
            custom  = custom[-1:]    if custom  else []

        return indices, custom

    def get_final(self) -> list[str] | str | None:
        indices, custom = self.parse_input(self.custom_buffer.text)

        if self.multiselect:
            predefined = [self.options[i] for i in sorted(self.selected | indices)] if self.options else []
            final      = predefined + [c for c in custom if c not in predefined]
            return final if final else None
        else:
            if indices and self.options is not None:
                return self.options[max(indices)]
            elif custom:
                return custom[-1]
            elif self.selected and self.options is not None:
                return self.options[min(self.selected)]
            return None

    def validate(self, final) -> str:
        if final is None or final == [] or final == "":
            return "Please enter at least one value."
        if self.strict and self.options is not None:
            options_lower = [o.lower() for o in self.options]
            values  = [final] if isinstance(final, str) else final
            invalid = [v for v in values if v.lower() not in options_lower]
            if invalid:
                return f"Invalid choice(s): {', '.join(invalid)}"
        return ""

    # ------------------------------------------------------------------ #
    #  Column layout                                                       #
    # ------------------------------------------------------------------ #

    def get_num_columns(self) -> int:
        if not self.options:
            return 1
        term_width  = os.get_terminal_size().columns
        max_opt_len = max(len(o) for o in self.options)
        num_digits  = len(str(len(self.options)))
        col_width   = max_opt_len + num_digits + 6  # "❯ ◉ N. " overhead
        num_cols    = max(1, term_width // col_width)
        return min(num_cols, len(self.options))

    def get_num_rows(self) -> int:
        if not self.options:
            return 0
        return math.ceil(len(self.options) / self.get_num_columns())

    # ------------------------------------------------------------------ #
    #  Rendering                                                           #
    # ------------------------------------------------------------------ #

    def get_header(self) -> list:
        if self.options is None:
            mode = "free input"
        else:
            mode = "multi-select" if self.multiselect else "single-select"
            if self.strict:
                mode += " strict"
        return [("class:header", f"{self.title} "), ("class:mode", f"[{mode}]\n")]

    def get_list_content(self) -> list:
        if not self.options:
            return []

        text = self.custom_buffer.text.strip()
        typed_indices, _ = self.parse_input(text) if text else (set(), [])

        num_cols   = self.get_num_columns()
        num_rows   = self.get_num_rows()
        term_width = os.get_terminal_size().columns
        col_width  = term_width // num_cols

        lines = []
        for row in range(num_rows):
            for col in range(num_cols):
                i = row + col * num_rows   # column-major: fill top-to-bottom first
                if i >= len(self.options):
                    lines.append(("", " " * col_width))
                    continue

                opt     = self.options[i]
                pointer = "❯ " if i == self.current else "  "
                check   = ("●" if not self.multiselect else "◉") if i in self.selected else "○"
                num     = f"{i+1}."
                cell    = f"{pointer}{check} {num} {opt}"
                cell    = cell.ljust(col_width)[:col_width]  # pad / truncate to col width

                if i in typed_indices and i == self.current:
                    style = "class:typed_focused"
                elif i in typed_indices:
                    style = "class:typed"
                elif i == self.current:
                    style = "class:focused"
                else:
                    style = ""

                lines.append((style, cell))

            lines.append(("", "\n"))

        return lines

    def get_input_preview(self) -> list:
        text = self.custom_buffer.text.strip()

        if self.options is None:
            if not text:
                return [("class:hint", "  nothing entered\n")]
            _, custom = self.parse_input(text)
            if not self.multiselect:
                custom = custom[-1:] if custom else []
            return [("class:preview", f"  → {', '.join(custom)}\n")] if custom else [("class:hint", "  nothing entered\n")]

        if not text and not self.selected:
            return [("class:hint", "  nothing selected\n")]

        if text:
            indices, custom = self.parse_input(text)
            active     = self.selected | indices
            predefined = [self.options[i] for i in sorted(active)]
            all_names  = predefined + [c for c in custom if c not in predefined]
        else:
            all_names = [self.options[i] for i in sorted(self.selected)]

        if not all_names:
            return [("class:hint", "  nothing selected\n")]
        return [("class:preview", f"  → {', '.join(all_names)}\n")]

    def get_error(self) -> list:
        if self.error_msg:
            return [("class:error", f"  ✗ {self.error_msg}\n")]

        if self.options is None:
            hint = "  Type value(s) separated by commas: " if self.multiselect else "  Type a value: "
        elif self.strict:
            hint = "  Enter number or option name only: "
        elif self.multiselect:
            hint = "  Enter numbers or values (e.g. 1,3,DeiT-S): "
        else:
            hint = "  Enter a number or value: "

        return [("class:hint", hint + "\n")]

    def get_footer(self) -> list:
        if self.options is None:
            keys = "  enter=confirm  esc=cancel\n"
        elif self.multiselect:
            keys = "  ↑↓←→=navigate  space=toggle  enter=confirm  esc=cancel\n"
        else:
            keys = "  ↑↓←→=navigate  space=select  enter=confirm  esc=cancel\n"
        return [("class:hint", keys)]

    # ------------------------------------------------------------------ #
    #  Key bindings                                                        #
    # ------------------------------------------------------------------ #

    def build_keybindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("up")
        def move_up(event):
            if self.options is not None:
                self.error_msg = ""
                num_cols = self.get_num_columns()
                num_rows = self.get_num_rows()
                row = self.current % num_rows
                col = self.current // num_rows
                row = (row - 1) % num_rows
                self.current = min(col * num_rows + row, len(self.options) - 1)

        @kb.add("down")
        def move_down(event):
            if self.options is not None:
                self.error_msg = ""
                num_cols = self.get_num_columns()
                num_rows = self.get_num_rows()
                row = self.current % num_rows
                col = self.current // num_rows
                row = (row + 1) % num_rows
                self.current = min(col * num_rows + row, len(self.options) - 1)

        @kb.add("left")
        def move_left(event):
            if self.options is not None:
                self.error_msg = ""
                num_rows = self.get_num_rows()
                col = self.current // num_rows
                row = self.current % num_rows
                col = (col - 1) % self.get_num_columns()
                self.current = min(col * num_rows + row, len(self.options) - 1)

        @kb.add("right")
        def move_right(event):
            if self.options is not None:
                self.error_msg = ""
                num_rows = self.get_num_rows()
                col = self.current // num_rows
                row = self.current % num_rows
                col = (col + 1) % self.get_num_columns()
                self.current = min(col * num_rows + row, len(self.options) - 1)

        @kb.add("space")
        def toggle_current(event):
            if self.options is not None:
                self.error_msg = ""
                if self.multiselect:
                    if self.current in self.selected:
                        self.selected.discard(self.current)
                    else:
                        self.selected.add(self.current)
                else:
                    self.selected = {self.current}

        @kb.add("enter")
        def confirm(event):
            final = self.get_final()
            error = self.validate(final)
            if error:
                self.error_msg = error
                return
            event.app.exit(result=final)

        @kb.add("escape")
        def cancel(event):
            event.app.exit(result=None)

        return kb

    # ------------------------------------------------------------------ #
    #  Layout & style                                                      #
    # ------------------------------------------------------------------ #

    def build_layout(self) -> Layout:
        input_window = Window(BufferControl(buffer=self.custom_buffer), height=1)

        rows = [Window(FormattedTextControl(self.get_header), height=1)]

        if self.options is not None:
            num_rows = self.get_num_rows()
            rows += [
                Window(FormattedTextControl(self.get_list_content), height=num_rows),
                Window(height=1),
            ]

        rows += [
            Window(FormattedTextControl(self.get_error), height=1),
            input_window,
            Window(FormattedTextControl(self.get_input_preview), height=1),
            Window(FormattedTextControl(self.get_footer), height=1),
        ]

        return Layout(HSplit(rows))

    @staticmethod
    def build_style() -> Style:
        return Style.from_dict({
            "header":        "bold",
            "mode":          "italic gray",
            "focused":       "bold cyan",
            "typed":         "bold yellow",
            "typed_focused": "bold yellow underline",
            "preview":       "bold green",
            "error":         "bold red",
            "hint":          "gray",
        })

    # ------------------------------------------------------------------ #
    #  Entry point                                                         #
    # ------------------------------------------------------------------ #

    def run(self) -> list[str] | str | None:
        app = Application(
            layout       = self.build_layout(),
            key_bindings = self.build_keybindings(),
            style        = self.build_style(),
            full_screen  = False,
        )
        return app.run()


result = MultiSelect(
    options = list(mon.DATASETS.keys()),
    multiselect = False,
    strict = True
).run()
print("Selected:", result)
