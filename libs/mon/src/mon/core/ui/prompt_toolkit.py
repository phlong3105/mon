#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Custom Prompts.

This module reimplements and extends the ``prompt_toolkit`` module.
"""

from __future__ import annotations

__all__ = [
    "ConfirmPrompt",
    "FloatPrompt",
    "IntPrompt",
    "PathPrompt",
    "Prompt",
]

import math
import os
from typing import Any, Generic, Literal, override, TypeVar

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.processors import BeforeInput, PasswordProcessor
from prompt_toolkit.styles import Style

from mon.core.path import Path
from mon.core.typing import PathLike
from mon.core.utils import truncate_string

# ==============================================================================
# region TYPE DEFINITIONS
# ==============================================================================

PromptType = TypeVar("PromptType")
DefaultType = TypeVar("DefaultType")

# endregion


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class PromptBase(Generic[PromptType]):
    """Interactive prompt base class supporting keyboard navigation, number
    shortcuts, direct input, and automatic column layout.

    Users can select values by:
        1. Arrow keys + space to navigate and toggle.
        2. Typing option numbers separated by commas (e.g. ``1,3``).
        3. Typing option names directly (case-insensitive match).
        4. Typing free-form values (when ``strict=False``).
        5. Left/right arrows to navigate across columns.

    Free-form tokens are passed through ``_convert()`` and cast to
    ``response_type``. Subclasses override ``response_type`` and ``_convert``
    to support types such as ``int``, ``float``, or ``Path``.

    Attributes:
        response_type (type): Target type for free-form tokens. Defaults to ``str``.
        validate_error_message (str): Shown when nothing is selected.
        invalid_choice_message (str): Shown when a strict-mode value is not in ``choices``.
        invalid_type_message (str): Shown when a free-form token cannot be converted.
    """

    response_type: type = str   # subclasses override this

    validate_error_message: str = "Please select or enter at least one value"
    invalid_choice_message: str = "Invalid choice(s)"
    invalid_type_message: str = "Invalid value(s)"
    skip_message: str = "Skip"

    choices: list[str] | None = None

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        prompt: str = "",
        *,
        choices: list[str] | None = None,
        defaults: str | int | list[str | int] | None = None,
        password: bool = False,
        multiselect: bool = False,
        strict: bool = False,
        skip: bool = False,
        case_sensitive: bool = True,
        truncate_length: int | None = None,
        truncate_side: Literal["left", "middle", "right"] = "middle",
        show_default: bool = True,
        show_choices: bool = True,
        show_column: bool = False,
    ):
        """Initialize a new instance.

        Args:
            prompt (str, optional): Prompt text shown as the header.
                Defaults to "".
            choices (List[str], optional): List of selectable options. If
                None, only free-form direct input is accepted. Defaults to None.
            defaults (str | int | list[str | int], optional): Pre-selected values.
                Each item may be a 0-based ``int`` index, a 1-based number string
                (``"1"``), or the choice value itself. Defaults to None.
            password (bool, optional): Mask typed characters in the input
                buffer. The live preview shows a count instead of values.
                Defaults to False.
            multiselect (bool, optional): Allow selecting multiple values.
                Defaults to False.
            strict (bool, optional): When True, only values present in
                ``choices`` are accepted; free-form input is rejected.
                Ignored when ``choices=None``. Defaults to False.
            skip (bool, optional): Allow skipping the prompt with a special
                "Skip" option. When enabled, pressing ``Escape`` returns None.
                Defaults to False.
            case_sensitive (bool, optional): Whether name matching against
                ``choices`` is case-sensitive. Defaults to True.
            truncate_length (int | None, optional): Maximum display length
                for paths in the choices list and header. None means no
                truncation. Defaults to None.
            truncate_side (Literal["left", "middle", "right"], optional):
                Which side to truncate on. ``"middle"`` preserves both the
                root and the filename, which is usually most readable for
                paths. Defaults to ``"middle"``.
            show_default (bool, optional): Display pre-selected values in
                the header. Defaults to True.
            show_choices (bool, optional): Render the choices list. When
                False the list is hidden but choices are still used for
                input resolution and validation. Defaults to True.
            show_column (bool, optional): Render the choices list in columns.
                Defaults to False.
        """
        # Assign attributes
        self.prompt = prompt
        if choices is not None:
            self.choices = [self.skip_message] + list(choices) if skip else list(choices)
        self.password = password
        self.multiselect = multiselect
        self.strict = strict
        self.skip = skip
        self.case_sensitive = case_sensitive
        self.truncate_length = truncate_length
        self.truncate_side = truncate_side
        self.show_default = show_default
        self.show_choices = show_choices
        self.show_column = show_column

        # Allocate resources
        self._selected: set = set()
        self._current: int = 0
        self._error_msg: str = ""
        self._default_text: str = ""
        self._custom_tokens: list = []
        self._custom_buffer: Buffer = Buffer()

        self._init_defaults(defaults=defaults)

    def _init_defaults(self, defaults: str | int | list[str | int] | None):
        """Pre-populate the selected set from ``defaults``.

        Each default is resolved in order:
            1. ``int``       → treated as a 0-based index.
            2. digit ``str`` → treated as a 1-based index.
            3. other ``str`` → matched against ``choices`` by name.

        In single-select mode only the first default is applied.

        Args:
            defaults: Default values or indices.
        """
        if not defaults:
            if self.multiselect:
                return
            else:
                defaults = 0 if self.skip or self.choices else 1

        # Normalize
        defaults = [defaults] if isinstance(defaults, (str, int)) else defaults
        candidates = defaults[:1] if not self.multiselect else defaults

        # Free-input mode — store as pre-filled buffer text
        if self.choices is None:
            # Pre-fill the buffer with default value(s)
            self._default_text = ", ".join(str(d) for d in candidates)
            return

        # Choices mode — resolve to indices as before
        options_cmp = self.choices if self.case_sensitive else [c.lower() for c in self.choices]

        for d in candidates:
            if isinstance(d, int):
                if 0 <= d < len(self.choices):
                    self._selected.add(d)
            elif isinstance(d, str):
                if d.isdigit():
                    idx = int(d) - 1
                    if 0 <= idx < len(self.choices):
                        self._selected.add(idx)
                else:
                    query = d if self.case_sensitive else d.lower()
                    try:
                        self._selected.add(options_cmp.index(query))
                    except ValueError:
                        pass

        # Visualized the given defaults
        if self._selected:
            # Move the cursor to the first default if any were resolved
            self._current = min(self._selected)

            # Pre-fill buffer to match resolved defaults —
            # _on_text_changed is not yet registered so write directly
            if self.multiselect:
                text = ", ".join(str(i + 1) for i in sorted(self._selected))
            else:
                text = str(min(self._selected) + 1)
            self._custom_buffer.set_document(Document(text=text, cursor_position=len(text)))

    # --- Callable & Context Manager ---
    def __call__(self) -> Any:
        """Run the interactive prompt loop.

        Returns:
            - ``response_type`` if ``multiselect=False`` and confirmed.
            - list[``response_type``] if ``multiselect=True`` and confirmed.
            - None if canceled via ``Escape``.
        """
        app = Application(
            layout=self._make_prompt(),
            key_bindings=self._build_keybindings(),
            style=self._build_style(),
            full_screen=False,
        )
        return app.run()

    # --- Creation ---
    @classmethod
    def ask(
        cls,
        prompt: str = "",
        *,
        choices: list[str | int] | None = None,
        defaults: str | int | list[str | int] | None = None,
        password: bool = False,
        multiselect: bool = False,
        strict: bool = False,
        skip: bool = False,
        case_sensitive: bool = True,
        truncate_length: int | None = None,
        truncate_side: Literal["left", "middle", "right"] = "middle",
        show_default: bool = True,
        show_choices: bool = True,
        show_column: bool = False,
    ) -> str | list[str] | None:
        """Construct and run the prompt, returning the result.

        Args:
            prompt (str, optional): Prompt text shown as the header.
                Defaults to "".
            choices (List[str], optional): List of selectable options. If
                None, only free-form direct input is accepted.
                Defaults to None.
            defaults (str | int | list[str | int], optional): Pre-selected values.
                Each item may be a 0-based ``int`` index, a 1-based number string
                (``"1"``), or the choice value itself. Defaults to None.
            password (bool, optional): Mask typed characters in the input
                buffer. The live preview shows a count instead of values.
                Defaults to False.
            multiselect (bool, optional): Allow selecting multiple values.
                Defaults to False.
            strict (bool, optional): When True, only values present in
                ``choices`` are accepted; free-form input is rejected.
                Ignored when ``choices=None``. Defaults to False.
            skip (bool, optional): Allow skipping the prompt with a special
                "Skip" option. When enabled, pressing ``Escape`` returns None.
                Defaults to False.
            case_sensitive (bool, optional): Whether name matching against
                ``choices`` is case-sensitive. Defaults to True.
            truncate_length (int | None, optional): Maximum display length
                for paths in the choices list and header. None means no
                truncation. Defaults to None.
            truncate_side (Literal["left", "middle", "right"], optional):
                Which side to truncate on. ``"middle"`` preserves both the
                root and the filename, which is usually most readable for
                paths. Defaults to ``"middle"``.
            show_default (bool, optional): Display pre-selected values in
                the header. Defaults to True.
            show_choices (bool, optional): Render the choices list. When
                False the list is hidden but choices are still used for
                input resolution and validation. Defaults to True.
            show_column (bool, optional): Render the choices list in columns.
                Defaults to False.

        Returns:
            str | List[str] | None: One of the following values:

                - ``response_type`` if ``multiselect=False`` and confirmed.
                - list[``response_type``] if ``multiselect=True`` and confirmed.
                - None if canceled via ``Escape``.
        """
        return cls(
            prompt,
            choices=choices,
            defaults=defaults,
            password=password,
            multiselect=multiselect,
            strict=strict,
            skip=skip,
            case_sensitive=case_sensitive,
            truncate_length=truncate_length,
            truncate_side=truncate_side,
            show_default=show_default,
            show_choices=show_choices,
            show_column=show_column,
        )()

    # --- Validation ---
    def _validate(self, final: Any) -> str:
        """Validate the resolved result.

        Checks (in order):
            1. Non-empty — uses ``validate_error_message``.
            2. Type conversion failure — uses ``invalid_type_message``.
            3. Strict-mode membership — uses ``invalid_choice_message``.

        Args:
            final (Any): Value returned by ``_get_final()``.

        Returns:
            str: Non-empty error message if invalid, else "".
        """
        if final is None:
            # Check if skip was intentionally selected
            if self.skip:
                indices, custom = self._parse_input(self._custom_buffer.text)
                active = self._selected | indices
                if 0 in active:
                    return ""   # ← skip selected, None is valid

            text = self._custom_buffer.text.strip()
            if text:
                _, custom = self._parse_input(text)
                if custom:
                    return (
                        f"{self.invalid_type_message}: cannot convert to "
                        f"'{self.response_type.__name__}'"
                    )
            return self.validate_error_message

        if self.strict and self.choices:
            options_cmp = (
                self.choices if self.case_sensitive
                else [c.lower() for c in self.choices]
            )
            values = [final] if not isinstance(final, list) else final
            invalid = [
                str(v) for v in values
                if (str(v) if self.case_sensitive else str(v).lower())
                not in options_cmp
            ]
            if invalid:
                return f"{self.invalid_choice_message}: {', '.join(invalid)}"

        return ""

    # --- Retrieval ---
    def _get_final(self) -> Any:
        """Compute the confirmed result from the current selection state.

        Predefined choices are returned as plain strings. Free-form tokens are
        passed through ``_convert()`` to cast them to ``response_type``.

        Returns:
            response_type | list[response_type] | None: Returns None on
                conversion failure (caught by ``_validate()``).
        """
        indices, custom = self._parse_input(self._custom_buffer.text)

        # Convert free-form tokens — return None on failure so _validate
        # can produce a user-facing error message.
        try:
            custom = [self._convert(c) for c in custom]
        except (ValueError, TypeError):
            return None

        if self.multiselect:
            predefined = (
                [self.choices[i] for i in sorted(self._selected | indices)
                 if not (self.skip and i == 0)]   # ← exclude skip from multiselect
                if self.choices else []
            )
            # If skip was explicitly selected, return None immediately
            if self.skip and 0 in (self._selected | indices):
                return None
            final = predefined + [c for c in custom if c not in predefined]
            return final if final else None
        else:
            idx = max(indices) if indices and self.choices else None
            if idx is None and self._selected and self.choices:
                idx = min(self._selected)

            # Skip selected → return None
            if self.skip and idx == 0:
                return None

            if idx is not None and self.choices:
                return self.choices[idx]
            elif custom:
                return custom[-1]
            return None

    # --- Processing ---
    def _parse_input(self, text: str) -> tuple[set, list]:
        """Parse comma-separated input into ``(matched_indices, custom_values)``.

        Resolution order for each token:
            1. Pure digit string → 1-based index into ``choices``.
            2. Name match in ``choices`` (respects ``case_sensitive``).
            3. Free-form string — appended to ``custom`` only when ``strict=False``.

        In single-select mode only the last resolved value is kept.

        Args:
            text (str): Raw text from the input buffer.

        Returns:
            tuple[set, list]: ``(matched_indices, custom_strings)``
        """
        indices: set = set()
        custom: list = []

        options_cmp: list = (
            (self.choices if self.case_sensitive else [c.lower() for c in self.choices])
            if self.choices else []
        )

        for part in text.split(","):
            part = part.strip()
            if not part:
                continue

            if self.choices is None:
                # No choices list — everything is free-form
                custom.append(part)
            elif part.isdigit():
                idx = int(part) - 1
                if 0 <= idx < len(self.choices):
                    indices.add(idx)
            else:
                query = part if self.case_sensitive else part.lower()
                try:
                    indices.add(options_cmp.index(query))
                except ValueError:
                    if not self.strict:
                        custom.append(part)

        if not self.multiselect:
            indices = {max(indices)} if indices else set()
            custom = custom[-1:] if custom else []

        return indices, custom

    def _convert(self, value: str) -> Any:
        """Convert a single free-form token to ``response_type``.

        Subclasses override this to add custom conversion logic or richer
        error messages.

        Args:
            value (str): Raw string token from the input buffer.

        Returns:
            Any: Value cast to ``response_type``.

        Raises:
            ValueError: If the conversion fails.
        """
        if self.response_type is str:
            return value
        try:
            return self.response_type(value)
        except (ValueError, TypeError):
            raise ValueError(
                f"'{value}' cannot be converted to "
                f"'{self.response_type.__name__}'"
            )

    # --- Visualization ---
    def _make_prompt(self) -> Layout:
        """Assemble the full prompt layout.

        Returns:
            Layout: prompt_toolkit Layout instance.
        """
        # Move the cursor to the last typed index as user types
        self._custom_buffer.on_text_changed += self._on_text_changed

        # Pre-fill buffer with default text for free-input mode
        if self._default_text:
            self._custom_buffer.set_document(
                Document(
                    text=self._default_text,
                    cursor_position=len(self._default_text),
                )
            )

        input_window = Window(
            BufferControl(
                buffer=self._custom_buffer,
                input_processors=[
                    BeforeInput("  "),
                    *([PasswordProcessor()] if self.password else []),
                ],
            ),
            height=1,
        )

        rows = [Window(FormattedTextControl(self._render_header), height=1)]

        if self.choices is not None and self.show_choices:
            num_rows = self._get_num_rows()
            reserved = 6  # header, spacer, label, input, preview, footer
            list_height = min(num_rows, max(1, os.get_terminal_size().lines - reserved))
            rows += [
                Window(FormattedTextControl(self._render_choices_list), height=list_height),
                Window(height=1),
            ]

        rows += [
            Window(FormattedTextControl(self._render_input_label), height=1),
            input_window,
            Window(FormattedTextControl(self._render_input_preview), height=1),
            Window(FormattedTextControl(self._render_footer), height=1),
        ]

        return Layout(HSplit(rows))

    def _render_header(self) -> list:
        """Render the header line with prompt text and mode tag."""
        if self.choices is None:
            mode = "free input"
        else:
            mode = "multi-select" if self.multiselect else "single-select"
            if self.strict:
                mode += " · strict"
            if not self.show_choices:
                mode += " · hidden"

        parts: list = [
            ("class:header", self.prompt),
            ("class:mode", f" [{mode}]"),
        ]

        # if self.show_default and self._selected and self.choices:
        #     defaults_str = ", ".join(self._truncate(self.choices[i]) for i in sorted(self._selected))
        #     parts.append(("class:default", f" ({defaults_str})"))

        parts.append(("", "\n"))
        return parts

    def _render_choices_list(self) -> list:
        """Render the choices list in columns."""
        if not self.choices:
            return []

        text = self._custom_buffer.text.strip()
        typed_indices, _ = self._parse_input(text) if text else (set(), [])
        all_selected = self._selected | typed_indices

        display = self._render_display()
        num_cols = self._get_num_columns(display=display)
        num_rows = self._get_num_rows()
        col_width = os.get_terminal_size().columns // num_cols

        reserved = 6
        visible_rows = max(1, os.get_terminal_size().lines - reserved)
        current_row = self._current % num_rows
        scroll = max(0, min(current_row - visible_rows // 2, num_rows - visible_rows))
        visible_end = scroll + visible_rows

        lines: list = []
        for row in range(scroll, min(num_rows, visible_end)):
            for col in range(num_cols):
                i = row + col * num_rows
                if i >= len(self.choices):
                    lines.append(("", " " * col_width))
                    continue

                pointer = "❯ " if i == self._current else "  "
                if self.multiselect:
                    check = "◉" if i in all_selected else "○"
                else:
                    check = "◉" if i == self._current else "○"
                cell = f"{pointer}{check} {i + 1}. {display[i]}".ljust(col_width)[:col_width]

                if i in typed_indices and i == self._current:
                    style = "class:typed_focused"
                elif i == self._current:
                    style = "class:focused"
                elif i in all_selected:
                    style = "class:typed"
                else:
                    style = ""

                lines.append((style, cell))
            lines.append(("", "\n"))

        return lines

    def _render_display(self) -> list:
        """Render the display of the choices."""
        return [self._truncate(c) for c in self.choices]

    def _render_input_label(self) -> list:
        """Render the input hint / error line above the buffer."""
        if self._error_msg:
            return [("class:error", f"  ✗ {self._error_msg}\n")]

        type_hint = "" if self.response_type is str else f" [{self.response_type.__name__}]"

        if self.choices is None:
            hint = (
                f"  Type value(s) separated by commas{type_hint}: "
                if self.multiselect else
                f"  Type a value{type_hint}: "
            )
        elif self.strict:
            hint = "  Enter an option number or name: "
        elif self.multiselect:
            hint = f"  Enter option numbers or values{type_hint}: "
        else:
            hint = f"  Enter an option number or value{type_hint}: "

        return [("class:hint", hint + "\n")]

    def _render_input_preview(self) -> list:
        """Render a live preview of the currently resolved selection.

        In password mode the preview shows a value count instead of the
        actual values.
        """
        # Password mode — never reveal typed content
        if self.password:
            n = len([p for p in self._custom_buffer.text.split(",") if p.strip()])
            return [
                ("class:preview" if n else "class:hint",
                 f"  → {n} value(s) entered\n" if n else "  nothing entered\n")
            ]

        text = self._custom_buffer.text.strip()

        # Free-input mode (no choices list)
        if self.choices is None:
            if not text:
                return [("class:hint", "  nothing entered\n")]
            _, custom = self._parse_input(text)
            if not self.multiselect:
                custom = custom[-1:] if custom else []
            return (
                [("class:preview", f"  → {', '.join(str(c) for c in custom)}\n")]
                if custom else [("class:hint", "  nothing entered\n")]
            )

        # Choices mode — merge space-toggled + typed input + free-form custom
        if not text and not self._selected:
            return [("class:hint", "  nothing selected\n")]

        indices, custom = self._parse_input(text) if text else (set(), [])
        predefined = [self.choices[i] for i in sorted(self._selected | indices)]
        all_names = predefined + [c for c in custom if c not in predefined]

        # In single-select mode only show the final resolved value
        if not self.multiselect:
            final = self._get_final()
            return (
                [("class:preview", f"  → {final}\n")]
                if final is not None else [("class:hint", "  nothing selected\n")]
            )

        return (
            [("class:preview", f"  → {', '.join(str(v) for v in all_names)}\n")]
            if all_names else [("class:hint", "  nothing selected\n")]
        )

    def _render_footer(self) -> list:
        """Render the key-hint footer."""
        if self.choices is None or not self.show_choices:
            keys = "  (enter=confirm  esc=cancel)"
        elif self.multiselect:
            keys = "  (arrows=navigate  space=toggle  enter=confirm  esc=cancel)"
        else:
            keys = "  (arrows=navigate  space=select  enter=confirm  esc=cancel)"
        return [("class:hint", keys + "\n")]

    def _on_text_changed(self, _):
        """Sync selection state and cursor when buffer text changes."""
        text = self._custom_buffer.text.strip()

        if not text:
            self._selected.clear()
            self._custom_tokens = []
            self._current = 0
            return

        indices, custom = self._parse_input(text)

        # Update both selected indices and custom tokens from buffer
        self._selected = indices.copy()
        self._custom_tokens = custom  # ← always track latest custom tokens

        # Move cursor to last resolved index
        if indices:
            self._current = (max(indices) if not self.multiselect else sorted(indices)[-1])

    def _sync_buffer(self) -> None:
        """Sync the buffer from ``_selected`` + ``_custom_tokens``.

        Uses the separately tracked ``_custom_tokens`` so free-form input
        survives space-toggle operations without being lost.
        """
        if not self.choices:
            return

        if self.multiselect:
            parts  = [str(i + 1) for i in sorted(self._selected)]
            parts += self._custom_tokens  # ← use tracked tokens, not re-parsed
        else:
            if self._selected:
                parts = [str(min(self._selected) + 1)]
            elif self._custom_tokens:
                parts = self._custom_tokens[-1:]  # single-select: keep last custom
            else:
                parts = []

        text = ", ".join(parts)

        # Detach to avoid re-triggering _on_text_changed during programmatic update
        self._custom_buffer.on_text_changed -= self._on_text_changed
        self._custom_buffer.set_document(Document(text=text, cursor_position=len(text)))
        self._custom_buffer.on_text_changed += self._on_text_changed

    @staticmethod
    def _build_style() -> Style:
        """Return the prompt color scheme."""
        return Style.from_dict(
            {
                "header": "bold",
                "mode": "gray",
                "default": "gray",
                "focused": "bold yellow",
                "typed": "bold yellow",
                "typed_focused": "bold yellow",
                "preview": "bold cyan",
                "error": "bold red",
                "hint": "gray",
            },
        )

    # --- Utilities ---
    def _build_keybindings(self) -> KeyBindings:
        """Build and return the key binding registry."""
        kb = KeyBindings()

        @kb.add("up")
        def _move_up(event):
            if not self.choices or not self.show_choices:
                return
            self._error_msg = ""
            num_rows = self._get_num_rows()
            col, row = divmod(self._current, num_rows)
            row = (row - 1) % num_rows
            self._current = min(col * num_rows + row, len(self.choices) - 1)
            if not self.multiselect:
                self._selected = {self._current}
                self._sync_buffer()

        @kb.add("down")
        def _move_down(event):
            if not self.choices or not self.show_choices:
                return
            self._error_msg = ""
            num_rows = self._get_num_rows()
            col, row = divmod(self._current, num_rows)
            row = (row + 1) % num_rows
            self._current = min(col * num_rows + row, len(self.choices) - 1)
            if not self.multiselect:
                self._selected = {self._current}
                self._sync_buffer()

        @kb.add("left")
        def _move_left(event):
            if not self.choices or not self.show_choices:
                return
            self._error_msg = ""
            num_rows = self._get_num_rows()
            num_cols = self._get_num_columns()
            col, row = divmod(self._current, num_rows)
            col = (col - 1) % num_cols
            self._current = min(col * num_rows + row, len(self.choices) - 1)
            if not self.multiselect:
                self._selected = {self._current}
                self._sync_buffer()

        @kb.add("right")
        def _move_right(event):
            if not self.choices or not self.show_choices:
                return
            self._error_msg = ""
            num_rows = self._get_num_rows()
            num_cols = self._get_num_columns()
            col, row = divmod(self._current, num_rows)
            col = (col + 1) % num_cols
            self._current = min(col * num_rows + row, len(self.choices) - 1)
            if not self.multiselect:
                self._selected = {self._current}
                self._sync_buffer()

        @kb.add("space")
        @kb.add("tab")
        def _toggle(event):
            if not self.choices or not self.show_choices:
                return
            self._error_msg = ""
            if self.multiselect:
                if self._current in self._selected:
                    self._selected.discard(self._current)
                else:
                    self._selected.add(self._current)
            else:
                self._selected = {self._current}
            # Sync buffer to reflect current selection
            self._sync_buffer()

        @kb.add("enter")
        def _confirm(event):
            final = self._get_final()
            error = self._validate(final)
            if error:
                self._error_msg = error
                return
            event.app.exit(result=final)

        @kb.add("escape")
        def _cancel(event):
            event.app.exit(result=None)

        @kb.add("c-c")
        def _interrupt(event):
            raise KeyboardInterrupt

        return kb

    def _get_num_columns(self, display: list[str] | None = None) -> int:
        """Compute how many columns fit in the current terminal width.

        Args:
            display (List[str] | None): Pre-truncated labels. If None, falls
                back to ``self.choices``.

        Returns:
            int: Number of columns (at least 1).
        """
        if not self.choices or not self.show_column:
            return 1

        labels = display if display is not None else self.choices
        num_digits = len(str(len(self.choices)))
        col_width = max(len(str(l)) for l in labels) + num_digits + 6  # "❯ ◉ N. " overhead
        return min(max(1, os.get_terminal_size().columns // col_width), len(self.choices))

    def _get_num_rows(self) -> int:
        """Compute the number of rows needed given the column count."""
        if not self.choices:
            return 0
        return math.ceil(len(self.choices) / self._get_num_columns())

    def _truncate(self, value: str | None) -> str:
        """Apply truncation to a display string if configured.

        Args:
            value (str): Original string.

        Returns:
            str: Truncated string, or original if no truncation is set.
        """
        if not value or value == self.skip_message or not self.truncate_length :
            return value or ""
        return truncate_string(
            value=value,
            max_length=self.truncate_length,
            side=self.truncate_side,
        )

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

class Prompt(PromptBase[str]):
    """A prompt that returns a str.

    Example:
        >>> name = Prompt.ask("Enter your name")
    """

    response_type = str


class IntPrompt(PromptBase[int]):
    """A prompt that returns an integer.

    Example:
        >>> burrito_count = IntPrompt.ask("How many burritos do you want to order")
    """

    response_type = int
    validate_error_message: str = "Please enter at least one integer"


class FloatPrompt(PromptBase[float]):
    """A prompt that returns a float.

    Example:
        >>> temperature = FloatPrompt.ask("Enter desired temperature")
    """

    response_type = float
    validate_error_message: str = "Please enter at least one float"


class PathPrompt(PromptBase[Path]):
    """A prompt that returns Paths."""

    response_type = Path
    validate_error_message = "Please select or enter at least one path"
    invalid_type_message = "Not a valid path"

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        prompt: str = "",
        *,
        choices: list[str] | None = None,
        defaults: str | int | list[str | int] | None = None,
        password: bool = False,
        multiselect: bool = False,
        strict: bool = False,
        skip: bool = False,
        case_sensitive: bool = True,
        truncate_length: int | None = None,
        truncate_side: Literal["left", "middle", "right"] = "middle",
        commonpath: PathLike | None = None,
        show_default: bool = True,
        show_choices: bool = True,
        show_column: bool = False,
    ):
        """Initialize a new instance.

        Args:
            prompt (str, optional): Prompt text shown as the header.
            Defaults to "".
            choices (List[str], optional): List of selectable options. If
                None, only free-form direct input is accepted.
                Defaults to None.
            defaults (str | int | list[str | int], optional): Pre-selected values.
                Each item may be a 0-based ``int`` index, a 1-based number string
                (``"1"``), or the choice value itself. Defaults to None.
            password (bool, optional): Mask typed characters in the input
                buffer. The live preview shows a count instead of values.
                Defaults to False.
            multiselect (bool, optional): Allow selecting multiple values.
                Defaults to False.
            strict (bool, optional): When True, only values present in
                ``choices`` are accepted; free-form input is rejected.
                Ignored when ``choices=None``. Defaults to False.
            skip (bool, optional): Allow skipping the prompt with a special
                "Skip" option. When enabled, pressing ``Escape`` returns None.
                Defaults to False.
            case_sensitive (bool, optional): Whether name matching against
                ``choices`` is case-sensitive. Defaults to True.
            truncate_length (int | None, optional): Maximum display length
                for paths in the choices list and header. None means no
                truncation. Defaults to None.
            truncate_side (Literal["left", "middle", "right"], optional):
                Which side to truncate on. ``"middle"`` preserves both the
                root and the filename, which is usually most readable for
                paths. Defaults to ``"middle"``.
            commonpath (PathLike, optional): If provided, trim this part from
                the start of each choice for display. Defaults to None.
            show_default (bool, optional): Display pre-selected values in
                the header. Defaults to True.
            show_choices (bool, optional): Render the choices list. When
                False the list is hidden but choices are still used for
                input resolution and validation. Defaults to True.
            show_column (bool, optional): Render the choices list in columns.
                Defaults to False.
        """
        # Assign attributes
        self.commonpath = commonpath

        # Continue the initialization chain
        super().__init__(
            prompt,
            choices=choices,
            defaults=defaults,
            password=password,
            multiselect=multiselect,
            strict=strict,
            skip=skip,
            case_sensitive=case_sensitive,
            truncate_length=truncate_length,
            truncate_side=truncate_side,
            show_default=show_default,
            show_choices=show_choices,
            show_column=show_column,
        )

    # --- Creation ---
    @classmethod
    def ask(
        cls,
        prompt: str = "",
        *,
        choices: list[str] | None = None,
        defaults: str | int | list[str | int] | None = None,
        password: bool = False,
        multiselect: bool = False,
        strict: bool = False,
        skip: bool = False,
        case_sensitive: bool = True,
        truncate_length: int | None = None,
        truncate_side: Literal["left", "middle", "right"] = "middle",
        commonpath: PathLike | None = None,
        show_default: bool = True,
        show_choices: bool = True,
        show_column: bool = False,
    ) -> str | list[str] | None:
        """Construct and run the prompt, returning the result.

        Args:
            prompt (str, optional): Prompt text shown as the header.
            Defaults to "".
            choices (List[str], optional): List of selectable options. If
                None, only free-form direct input is accepted.
                Defaults to None.
            defaults (str | int | list[str | int], optional): Pre-selected values.
                Each item may be a 0-based ``int`` index, a 1-based number string
                (``"1"``), or the choice value itself. Defaults to None.
            password (bool, optional): Mask typed characters in the input
                buffer. The live preview shows a count instead of values.
                Defaults to False.
            multiselect (bool, optional): Allow selecting multiple values.
                Defaults to False.
            strict (bool, optional): When True, only values present in
                ``choices`` are accepted; free-form input is rejected.
                Ignored when ``choices=None``. Defaults to False.
            skip (bool, optional): Allow skipping the prompt with a special
                "Skip" option. When enabled, pressing ``Escape`` returns None.
                Defaults to False.
            case_sensitive (bool, optional): Whether name matching against
                ``choices`` is case-sensitive. Defaults to True.
            truncate_length (int | None, optional): Maximum display length
                for paths in the choices list and header. None means no
                truncation. Defaults to None.
            truncate_side (Literal["left", "middle", "right"], optional):
                Which side to truncate on. ``"middle"`` preserves both the
                root and the filename, which is usually most readable for
                paths. Defaults to ``"middle"``.
            commonpath (PathLike, optional): If provided, trim this part from
                the start of each choice for display. Defaults to None.
            show_default (bool, optional): Display pre-selected values in
                the header. Defaults to True.
            show_choices (bool, optional): Render the choices list. When
                False the list is hidden but choices are still used for
                input resolution and validation. Defaults to True.
            show_column (bool, optional): Render the choices list in columns.
                Defaults to False.

        Returns:
            str | List[str] | None: One of the following values:

                - ``response_type`` if ``multiselect=False`` and confirmed.
                - list[``response_type``] if ``multiselect=True`` and confirmed.
                - None if canceled via ``Escape``.
        """
        _prompt = cls(
            prompt,
            choices=choices,
            defaults=defaults,
            password=password,
            multiselect=multiselect,
            strict=strict,
            skip=skip,
            case_sensitive=case_sensitive,
            truncate_length=truncate_length,
            truncate_side=truncate_side,
            commonpath=commonpath,
            show_default=show_default,
            show_choices=show_choices,
            show_column=show_column,
        )
        return _prompt()

    # --- Visualization ---
    @override
    def _render_display(self) -> list:
        """Render the display of the choices."""
        display = []
        if self.commonpath:
            for c in self.choices:
                if c == self.skip_message:
                    display.append(self.skip_message)
                else:
                    display.append(Path(c).unique_path_from(self.commonpath))
        display = [self._truncate(d) for d in display]
        return display


class ConfirmPrompt(Prompt):
    """A yes / no confirmation prompt that returns a boolean.

    The user can confirm or cancel via keyboard navigation, direct input
    (``y`` / ``n`` / ``yes`` / ``no``), or number shortcuts (``1`` / ``2``).

    Example:
        >>> if ConfirmPrompt.ask("Continue training?", default=True):
        ...     run_training()
    """

    response_type = bool
    validate_error_message = "Please enter y or n"
    invalid_choice_message = "Please enter y or n"

    _YES = {"y", "yes", "1"}
    _NO = {"n", "no", "2"}

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        prompt: str = "",
        *,
        defaults: bool | None = None,
        show_default: bool = True,
    ):
        """Initialize a new instance.

        Args:
            prompt (str, optional): Prompt text. Defaults to "".
            defaults (bool | None, optional): Pre-selected answer.
                True pre-selects **Yes**, False pre-selects **No**, None means
                no pre-selection (answer required). Defaults to None.
            show_default (bool, optional): Show the pre-selected answer in
                the header. Defaults to True.
        """
        # Resolve default → index into ["Yes", "No"]
        if defaults:
            defaults = [0]  # "Yes"
        elif not defaults:
            defaults = [1]  # "No"

        # Continue the initialization chain
        super().__init__(
            prompt,
            choices=["Yes", "No"],
            defaults=defaults,
            password=False,
            multiselect=False,
            strict=True,
            case_sensitive=False,
            show_default=show_default,
            show_choices=True,
            show_column=False,
        )

    # --- Creation ---
    @classmethod
    def ask(
        cls,
        prompt: str = "",
        *,
        defaults: bool | None = None,
        show_default: bool = True,
    ) -> bool | None:
        """Construct and run the prompt, returning the result.

        Args:
            prompt (str, optional): Prompt text. Defaults to "".
            defaults (bool | None, optional): Pre-selected answer.
                True pre-selects **Yes**, False pre-selects **No**, None means
                no pre-selection (answer required). Defaults to None.
            show_default (bool, optional): Show the pre-selected answer in
                the header. Defaults to True.

        Returns:
            bool | None: True if the user confirmed Yes, False if the user,
                None if canceled via Escape.
        """
        _prompt = cls(prompt, defaults=defaults, show_default=show_default)
        return _prompt()

    # --- Validation ---
    @override
    def _validate(self, final: Any) -> str:
        """Validate that a yes/no answer was given.

        Args:
            final: Value from ``_get_final()``.

        Returns:
            str: Error message, or ``""`` if valid.
        """
        if final is None:
            text = self._custom_buffer.text.strip()
            if text:
                return self.invalid_choice_message
            return self.validate_error_message
        return ""  # True and False are both valid

    # --- Retrieval ---
    @override
    def _get_final(self) -> bool | None:
        """Resolve the confirmed answer to a boolean

        Choice-list selection (arrow + space) returns True for index 0 (Yes) and
        False for index 1 (No). Free-form tokens are passed through ``_convert()``.

        Returns:
            bool | None
        """
        indices, custom = self._parse_input(self._custom_buffer.text)

        # Free-form input takes priority over list selection
        if custom:
            try:
                return self._convert(custom[-1])
            except (ValueError, TypeError):
                return None

        # Resolve from list index or space-toggle selection
        idx = None
        if indices:
            idx = max(indices)
        elif self._selected:
            idx = min(self._selected)

        if idx is not None:
            return idx == 0  # 0 = "Yes" → True, 1 = "No" → False

        return None

    # --- Processing ---
    @override
    def _convert(self, value: str) -> bool:
        """Convert free-form token to boolean.

        Accepts:
            - ``y``, ``yes``, ``1`` → True
            - ``n``, ``no``, ``2`` → False (case-insensitive).

        Args:
            value (str): Raw token from the input buffer.

        Returns:
            bool: Converted value.

        Raises:
            ValueError: If the token is not a recognized yes/no value.
        """
        v = value.strip().lower()
        if v in self._YES:
            return True
        if v in self._NO:
            return False
        raise ValueError(f"'{value}' is not a valid yes/no response")

    # --- Visualization ---
    @override
    def _render_header(self) -> list:
        """Render header with ``[Y/n]`` / ``[y/N]`` / ``[y/n]`` style tag.

        Returns:
            list: ``(style, text)`` tuples for prompt_toolkit.
        """
        if self._selected:
            idx = min(self._selected)
            yn = "Y/n" if idx == 0 else "y/N"
        else:
            yn = "y/n"

        return [
            ("class:header", self.prompt),
            ("class:mode", f" [{yn}]"),
            ("", "\n"),
        ]

    @override
    def _render_footer(self) -> list:
        """Render footer with confirm-specific hints.

        Returns:
            list: ``(style, text)`` tuples for prompt_toolkit.
        """
        return [
            (
                "class:hint",
                "  arrows=navigate  space=select  enter=confirm  esc=cancel\n"
            )
        ]

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
