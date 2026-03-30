#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Custom Rich Prompts.

This module reimplements and extends the ``rich.prompt`` module.
"""

from __future__ import annotations

__all__ = [
    "Confirm",
    "FloatPrompt",
    "IntPrompt",
    "InvalidResponse",
    "OptionPrompt",
    "PathPrompt",
    "Prompt",
    "PromptBase",
    "PromptError",
]

from typing import (
    Any,
    Generic,
    List,
    Literal, Optional,
    overload,
    override, TextIO,
    TypeVar,
    Union,
)

import rich
from rich import get_console
from rich.columns import Columns
from rich.progress import Console
from rich.text import Text, TextType

from mon.core.path import Path
from mon.core.typing import PathLike
from mon.core.utils import is_int, is_valid_str, to_list, truncate_string

# ==============================================================================
# region TYPE DEFINITIONS
# ==============================================================================

PromptType = TypeVar("PromptType")
DefaultType = TypeVar("DefaultType")

# endregion


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class PromptError(Exception):
    """Exception base class for prompt related errors."""
    pass


class InvalidResponse(PromptError):
    """Exception to indicate a response was invalid. Raise this within
    process_response() to indicate an error and provide an error message.

    Args:
        message (Union[str, Text]): Error message.
    """

    def __init__(self, message: TextType) -> None:
        self.message = message

    def __rich__(self) -> TextType:
        return self.message


class PromptBase(Generic[PromptType]):
    """Ask the user for input until a valid response is received. This is the
    base class, see one of the concrete classes for examples.
    """

    response_type: type = str

    validate_error_message = "[prompt.invalid]Please enter a valid value"
    illegal_choice_message = (
        "[prompt.invalid.choice]Please select one of the available options"
    )
    prompt_suffix = ": "

    choices: Optional[List[str]] = None

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        prompt: TextType = "",
        *,
        console: Optional[Console] = None,
        password: bool = False,
        choices: Optional[List[str]] = None,
        case_sensitive: bool = True,
        show_default: bool = True,
        show_choices: bool = True,
    ):
        """Initialize a new instance.

        Args:
            prompt (TextType, optional): Prompt text. Defaults to "".
            console (Console, optional): A Console instance or None to use
                global console. Defaults to None.
            password (bool, optional): Enable password input. Defaults to False.
            choices (List[str], optional): A list of valid choices.
                Defaults to None.
            case_sensitive (bool, optional): Matching of choices should be
                case-sensitive. Defaults to True.
            show_default (bool, optional): Show default in prompt.
                Defaults to True.
            show_choices (bool, optional): Show choices in prompt.
                Defaults to True.
        """
        self.console = console or get_console()
        self.prompt = (
            Text.from_markup(prompt, style="prompt")
            if isinstance(prompt, str)
            else prompt
        )
        self.password = password
        if choices is not None:
            self.choices = choices
        self.case_sensitive = case_sensitive
        self.show_default = show_default
        self.show_choices = show_choices

    # --- Callable & Context Manager ---
    @overload
    def __call__(self, *, stream: Optional[TextIO] = None) -> PromptType:
        ...

    @overload
    def __call__(
        self, *, default: DefaultType, stream: Optional[TextIO] = None
    ) -> Union[PromptType, DefaultType]:
        ...

    def __call__(self, *, default: Any = ..., stream: Optional[TextIO] = None) -> Any:
        """Run the prompt loop.

        Args:
            default (Any, optional): Optional default value.

        Returns:
            PromptType: Processed value.
        """
        while True:
            self.pre_prompt()
            prompt = self.make_prompt(default)
            value = self.get_input(self.console, prompt, self.password, stream=stream)
            if value == "" and default != ...:
                return default
            try:
                return_value = self.process_response(value)
            except InvalidResponse as error:
                self.on_validate_error(value, error)
                continue
            else:
                return return_value

    # --- Creation ---
    @classmethod
    @overload
    def ask(
        cls,
        prompt: TextType = "",
        *,
        console: Optional[Console] = None,
        password: bool = False,
        choices: Optional[List[str]] = None,
        case_sensitive: bool = True,
        show_default: bool = True,
        show_choices: bool = True,
        default: DefaultType,
        stream: Optional[TextIO] = None,
    ) -> Union[DefaultType, PromptType]:
        ...

    @classmethod
    @overload
    def ask(
        cls,
        prompt: TextType = "",
        *,
        console: Optional[Console] = None,
        password: bool = False,
        choices: Optional[List[str]] = None,
        case_sensitive: bool = True,
        show_default: bool = True,
        show_choices: bool = True,
        stream: Optional[TextIO] = None,
    ) -> PromptType:
        ...

    @classmethod
    def ask(
        cls,
        prompt: TextType = "",
        *,
        console: Optional[Console] = None,
        password: bool = False,
        choices: Optional[List[str]] = None,
        case_sensitive: bool = True,
        show_default: bool = True,
        show_choices: bool = True,
        default: Any = ...,
        stream: Optional[TextIO] = None,
    ) -> Any:
        """Shortcut to construct and run a prompt loop and return the result.

        Example:
            >>> filename = Prompt.ask("Enter a filename")

        Args:
            prompt (TextType, optional): Prompt text. Defaults to "".
            console (Console, optional): A Console instance or None to use
                global console. Defaults to None.
            password (bool, optional): Enable password input. Defaults to False.
            choices (List[str], optional): A list of valid choices.
                Defaults to None.
            case_sensitive (bool, optional): Matching of choices should be
                case-sensitive. Defaults to True.
            show_default (bool, optional): Show default in prompt.
                Defaults to True.
            show_choices (bool, optional): Show choices in prompt.
                Defaults to True.
            stream (TextIO, optional): Optional text file open for reading to
                get input. Defaults to None.
        """
        _prompt = cls(
            prompt,
            console=console,
            password=password,
            choices=choices,
            case_sensitive=case_sensitive,
            show_default=show_default,
            show_choices=show_choices,
        )
        return _prompt(default=default, stream=stream)

    # --- Validation ---
    def check_choice(self, value: str) -> bool:
        """Check value is in the list of valid choices.

        Args:
            value (str): Value entered by user.

        Returns:
            bool: True if choice was valid, otherwise False.
        """
        assert self.choices is not None
        if self.case_sensitive:
            return value.strip() in self.choices
        return value.strip().lower() in [choice.lower() for choice in self.choices]

    def on_validate_error(self, value: str, error: InvalidResponse) -> None:
        """Called to handle validation error.

        Args:
            value (str): String entered by user.
            error (InvalidResponse): Exception instance the initiated the error.
        """
        self.console.print(error)

    # --- Retrieval ---
    @classmethod
    def get_input(
        cls,
        console: Console,
        prompt: TextType,
        password: bool,
        stream: Optional[TextIO] = None,
    ) -> str:
        """Get input from user.

        Args:
            console (Console): Console instance.
            prompt (TextType): Prompt text.
            password (bool): Enable password entry.

        Returns:
            str: String from user.
        """
        return console.input(prompt, password=password, stream=stream)

    # --- Processing ---
    def process_response(self, value: str) -> PromptType:
        """Process response from user, convert to prompt type.

        Args:
            value (str): String typed by user.

        Raises:
            InvalidResponse: If ``value`` is invalid.

        Returns:
            PromptType: The value to be returned from ask method.
        """
        value = value.strip()
        try:
            return_value: PromptType = self.response_type(value)
        except ValueError:
            raise InvalidResponse(self.validate_error_message)

        if self.choices is not None:
            if not self.check_choice(value):
                raise InvalidResponse(self.illegal_choice_message)

            if not self.case_sensitive:
                # return the original choice, not the lower case version
                return_value = self.response_type(
                    self.choices[
                        [choice.lower() for choice in self.choices].index(value.lower())
                    ]
                )
        return return_value

    # --- Visualization ---
    def pre_prompt(self) -> None:
        """Hook to display something before the prompt."""

    def make_prompt(self, default: DefaultType) -> Text:
        """Make prompt text.

        Args:
            default (DefaultType): Default value.

        Returns:
            Text: Text to display in prompt.
        """
        prompt = self.prompt.copy()
        prompt.end = ""

        if self.show_choices and self.choices:
            _choices = "/".join(self.choices)
            choices = f"[{_choices}]"
            prompt.append(" ")
            prompt.append(choices, "prompt.choices")

        if (
            default != ...
            and self.show_default
            and isinstance(default, (str, self.response_type))
        ):
            prompt.append(" ")
            _default = self.render_default(default)
            prompt.append(_default)

        prompt.append(self.prompt_suffix)

        return prompt

    def render_default(self, default: DefaultType) -> Text:
        """Turn the supplied default in to a Text instance.

        Args:
            default (DefaultType): Default value.

        Returns:
            Text: Text containing rendering of default value.
        """
        return Text(f"({default})", "prompt.default")

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

    @classmethod
    def ask(
        cls,
        prompt: TextType = "",
        *,
        console: Optional[Console] = None,
        password: bool = False,
        choices: Optional[List[str]] = None,
        case_sensitive: bool = True,
        show_default: bool = True,
        show_choices: bool = True,
        default: Any = ...,
        stream: Optional[TextIO] = None,
    ) -> Any:
        """Shortcut to construct and run a prompt loop and return the result.

        Example:
            >>> filename = Prompt.ask("Enter a filename")

        Args:
            prompt (TextType, optional): Prompt text. Defaults to "".
            console (Console, optional): A Console instance or None to use
                global console. Defaults to None.
            password (bool, optional): Enable password input. Defaults to False.
            choices (List[str], optional): A list of valid choices.
                Defaults to None.
            case_sensitive (bool, optional): Matching of choices should be
                case-sensitive. Defaults to True.
            show_default (bool, optional): Show default in prompt.
                Defaults to True.
            show_choices (bool, optional): Show choices in prompt.
                Defaults to True.
            stream (TextIO, optional): Optional text file open for reading to
                get input. Defaults to None.
        """
        # Prepare choices
        choices = to_list(choices) if choices else None

        # Create prompt
        _prompt = cls(
            prompt,
            console=console,
            password=password,
            choices=choices,
            case_sensitive=case_sensitive,
            show_default=show_default,
            show_choices=show_choices,
        )

        # Prepare default value
        default = str(default) if default else ""

        # Run prompt
        response = _prompt(default=default, stream=stream)

        # Process response
        if isinstance(response, (list, tuple)):
            response = response[0] if len(response) == 1 else response

        return response

    def process_response(self, value: str) -> PromptType:
        """Process response from user, convert to prompt type.

        Args:
            value (str): String typed by user.

        Raises:
            InvalidResponse: If ``value`` is invalid.

        Returns:
            PromptType: The value to be returned from ask method.
        """
        value = value.strip() if isinstance(value, str) else value

        if self.choices:
            if not value and not self.allow_empty:
                raise InvalidResponse(self.illegal_choice_message)

            # Split input to support multi-index/multi-value selection (e.g., "0,2")
            input_parts = to_list(value, sep=",|;")
            processed_values = []

            for part in input_parts:
                part = part.strip()
                # Check if part is a valid index
                if is_int(part):
                    idx = int(part)
                    if 0 <= idx < len(self.choices):
                        processed_values.append(self.choices[idx])
                    else:
                        raise IndexError(f"Index {idx} out of range for choices "
                                         f"of size {len(self.choices)}.")
                # Check if part is a direct choice match
                elif self.check_choice(part):
                    processed_values.append(part)
                # Handle free-form input (if allowed) or error
                else:
                    processed_values.append(part)

            # Return a single value if only one was selected, else the list
            return processed_values[0] if len(processed_values) == 1 else processed_values

        return value


class OptionPrompt(PromptBase[str]):
    """Ask the user for input until a valid response is received.

    There are three ways the user can input values:
        1. Direct input
        2. Enter an option from a list of choices
        3. Enter an index corresponding to a choice from the list
    """

    response_type: type = str

    validate_error_message = "[prompt.invalid]Please enter a valid value"
    illegal_choice_message = (
        "[prompt.invalid.choice]Please select one of the available options"
    )
    prompt_suffix = ": "

    choices: Optional[List[str]] = None

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        prompt: TextType = "",
        *,
        console: Optional[Console] = None,
        password: bool = False,
        choices: Optional[List[str]] = None,
        case_sensitive: bool = True,
        show_default: bool = True,
        show_choices: bool = True,
        column_first: bool = False,
        truncate_length: int | None = None,
        truncate_side: Literal["left", "middle", "right"] = "right",
        multiselect: bool = False,
        allow_empty: bool = False,
    ):
        """Initialize a new instance.

        Args:
            prompt (TextType, optional): Prompt text. Defaults to "".
            console (Console, optional): A Console instance or None to use
                global console. Defaults to None.
            password (bool, optional): Enable password input. Defaults to False.
            choices (List[str], optional): A list of valid choices.
                Defaults to None.
            case_sensitive (bool, optional): Matching of choices should be
                case-sensitive. Defaults to True.
            show_default (bool, optional): Show default in prompt.
                Defaults to True.
            show_choices (bool, optional): Show choices in prompt.
                Defaults to True.
            column_first (bool, optional): If True, print choices in
                column-first. Defaults to False.
            truncate_length (int | None, optional): Truncate long choices to
                this length for display. Defaults to None (no truncation).
            truncate_side (Literal["left", "middle", "right"], optional): If
                ``truncate_length`` is set, which side to truncate on.
                Defaults to "right".
            multiselect (bool, optional): If True, allow multiple selections.
                Defaults to False.
            allow_empty (bool, optional): If True, allow an empty string as a
                valid response. Defaults to False.
        """
        # Assign attributes
        self.column_first = column_first
        self.truncate_length = truncate_length
        self.truncate_side = truncate_side
        self.multiselect = multiselect
        self.allow_empty = allow_empty

        # Continue the initialization chain
        super().__init__(
            prompt,
            console=console,
            password=password,
            choices=choices,
            case_sensitive=case_sensitive,
            show_default=show_default,
            show_choices=show_choices,
        )

    # --- Callable & Context Manager ---
    @override
    def __call__(
        self,
        *,
        default: Any = ..., stream: Optional[TextIO] = None
    ) -> Any:
        """Run the prompt loop.

        Args:
            default (Any, optional): Optional default value.

        Returns:
            PromptType: Processed value.
        """
        while True:
            self.pre_prompt()
            prompt = self.make_prompt(default)
            value = self.get_input(self.console, prompt, self.password, stream=stream)
            if value == "" and default != ...:
                # return default
                value = default
            try:
                return_value = self.process_response(value)
            except (InvalidResponse, IndexError) as error:
                self.on_validate_error(value, error)
                continue
            else:
                return return_value

    # --- Creation ---
    @override
    @classmethod
    def ask(
        cls,
        prompt: TextType = "",
        *,
        console: Optional[Console] = None,
        password: bool = False,
        choices: Optional[List[str]] = None,
        case_sensitive: bool = True,
        show_default: bool = True,
        show_choices: bool = True,
        column_first: bool = False,
        truncate_length: int | None = None,
        truncate_side: Literal["left", "middle", "right"] = "right",
        multiselect: bool = False,
        allow_empty: bool = False,
        default: Any = ...,
        stream: Optional[TextIO] = None,
    ) -> Any:
        """Shortcut to construct and run a prompt loop and return the result.

        Example:
            >>> filename = Prompt.ask("Enter a filename")

        Args:
            prompt (TextType, optional): Prompt text. Defaults to "".
            console (Console, optional): A Console instance or None to use
                global console. Defaults to None.
            password (bool, optional): Enable password input. Defaults to False.
            choices (List[str], optional): A list of valid choices.
                Defaults to None.
            case_sensitive (bool, optional): Matching of choices should be
                case-sensitive. Defaults to True.
            show_default (bool, optional): Show default in prompt.
                Defaults to True.
            show_choices (bool, optional): Show choices in prompt.
                Defaults to True.
            column_first (bool, optional): If True, print choices in
                column-first. Defaults to False.
            truncate_length (int | None, optional): Truncate long choices to
                this length for display. Defaults to None (no truncation).
            truncate_side (Literal["left", "middle", "right"], optional): If
                ``truncate_length`` is set, which side to truncate on.
                Defaults to "right".
            multiselect (bool, optional): If True, allow multiple selections.
                Defaults to False.
            allow_empty (bool, optional): If True, allow an empty string as a
                valid response. Defaults to False.
            stream (TextIO, optional): Optional text file open for reading to
                get input. Defaults to None.
        """
        # 1. Prepare choices
        choices = to_list(choices) if choices else None

        # 2. Prepare default value
        if isinstance(default, str) and not is_valid_str(default):
            default = None

        # 3. Create prompt
        _prompt = cls(
            prompt,
            console=console,
            password=password,
            choices=choices,
            case_sensitive=case_sensitive,
            show_default=show_default,
            show_choices=show_choices,
            column_first=column_first,
            truncate_length=truncate_length,
            truncate_side=truncate_side,
            multiselect=multiselect,
            allow_empty=allow_empty,
        )
        return _prompt(default=default, stream=stream)

    # --- Validation ---
    @override
    def check_choice(self, value: str) -> bool:
        """Check value is in the list of valid choices.

        Args:
            value (str): Value entered by user.

        Returns:
            bool: True if choice was valid, otherwise False.
        """
        # Modified Code
        assert self.choices is not None
        if self.case_sensitive:
            return value in self.choices
        return value.lower() in [choice.lower() for choice in self.choices]

    # --- Processing ---
    @override
    def process_response(self, value: str) -> PromptType:
        """Process response from user, convert to prompt type.

        Args:
            value (str): String typed by user.

        Raises:
            InvalidResponse: If ``value`` is invalid.

        Returns:
            PromptType: The value to be returned from ask method.
        """
        if self.choices:
            return self.process_choices_response(value)
        elif self.multiselect:
            return self.process_inputs_response(value)
        else:
            return value.strip() if isinstance(value, str) else value

    def process_choices_response(self, value: str) -> List[str]:
        """Process response from user, convert to prompt type."""
        value = value.strip() if isinstance(value, str) else value

        # Handle empty string
        if not value and not self.allow_empty:
            raise InvalidResponse(self.illegal_choice_message)

        # Handle multiple selections (e.g., user input = "0,1,2")
        # Split value into individual selections
        input_parts = to_list(value, sep=",|;")
        processed_values = []

        for part in input_parts:
            part = part.strip() if isinstance(part, str) else part

            if is_int(part):
                # If part is an index, return the corresponding choice
                idx = int(part)
                if 0 <= idx < len(self.choices):
                    processed_values.append(self.choices[idx])
                else:
                    raise IndexError(
                        f"Index {idx} out of range for choices of size "
                        f"{len(self.choices)}."
                    )
            elif self.check_choice(part):
                # If part is a direct choice match, return it
                processed_values.append(part)
            elif is_valid_str(part):
                # If part is not a valid choice, return it as-is
                processed_values.append(part)

        # Return
        if self.multiselect:
            # If multiselect is enabled, return a list of processed values
            # return processed_values[0] if len(processed_values) == 1 else processed_values
            return processed_values
        else:
            # If multiselect is disabled, return the first processed value
            return processed_values[0]

    def process_inputs_response(self, value: str) -> List[str]:
        """Process response from user, convert to prompt type."""
        value = value.strip() if isinstance(value, str) else value

        # Handle empty string
        if not value and not self.allow_empty:
            raise InvalidResponse(self.illegal_choice_message)

        # Handle multiple selections (e.g., user input = "0,1,2")
        # Split value into individual selections
        input_parts = to_list(value, sep=",|;")
        processed_values = []

        for part in input_parts:
            part = part.strip() if isinstance(part, str) else part

            if is_valid_str(part):
                # If part is not a valid choice, return it as-is
                processed_values.append(part)

        # Return
        if self.multiselect:
            # If multiselect is enabled, return a list of processed values
            # return processed_values[0] if len(processed_values) == 1 else processed_values
            return processed_values
        else:
            # If multiselect is disabled, return the first processed value
            return processed_values[0]

    # --- Visualization ---
    @override
    def make_prompt(self, default: DefaultType) -> Text:
        """Make prompt text.

        Args:
            default (DefaultType): Default value.

        Returns:
            Text: Text to display in prompt.
        """
        # Modified Code
        if self.show_choices and self.choices and len(self.choices) > 0:
            rich.print(self.prompt)
            self.render_choices()
            prompt = Text.from_markup("", style="prompt")
        else:
            prompt = self.prompt.copy()
        prompt.end = ""

        if (
            default != ...
            and self.show_default
            and isinstance(default, (int, str, self.response_type))
        ):
            prompt.append(" ")
            _default = self.render_default(default)
            prompt.append(_default)

        prompt.append(self.prompt_suffix)

        return prompt

    def render_choices(self):
        """Print available choices in columns."""
        choices_ = []
        for i, choice in enumerate(self.choices):
            if self.truncate_length:
                choice = truncate_string(
                    value=choice,
                    max_length=self.truncate_length,
                    side=self.truncate_side,
                )
            choices_.append(f"{f'{i}.':>6} {choice}")
        columns = Columns(choices_, equal=True, column_first=self.column_first)
        rich.print(columns)

    @override
    def render_default(self, default: DefaultType) -> Text:
        """Render the default as (y) or (n) rather than True/False."""
        if isinstance(default, int):
            if not (0 <= default < len(self.choices)):
                raise ValueError(f"Default index {default} out of range.")
            return Text(f"({self.choices[default]})", "prompt.default")
        else:
            default = default.strip()
            return Text(f"({default})", "prompt.default")


class PathPrompt(OptionPrompt):
    """A prompt that returns a path."""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        prompt: TextType = "",
        *,
        console: Optional[Console] = None,
        password: bool = False,
        choices: Optional[List[str]] = None,
        case_sensitive: bool = True,
        show_default: bool = True,
        show_choices: bool = True,
        column_first: bool = False,
        truncate_length: int | None = None,
        truncate_side: Literal["left", "middle", "right"] = "middle",
        commonpath: PathLike | None = None,
        multiselect: bool = False,
        allow_empty: bool = False,
    ):
        """Initialize a new instance.

        Args:
            prompt (TextType, optional): Prompt text. Defaults to "".
            console (Console, optional): A Console instance or None to use
                global console. Defaults to None.
            password (bool, optional): Enable password input. Defaults to False.
            choices (List[str], optional): A list of valid choices.
                Defaults to None.
            case_sensitive (bool, optional): Matching of choices should be
                case-sensitive. Defaults to True.
            show_default (bool, optional): Show default in prompt.
                Defaults to True.
            show_choices (bool, optional): Show choices in prompt.
                Defaults to True.
            column_first (bool, optional): If True, print choices in
                column-first. Defaults to False.
            truncate_length (int | None, optional): Truncate long choices to
                this length for display. Defaults to None (no truncation).
            truncate_side (Literal["left", "middle", "right"], optional): If
                ``truncate_length`` is set, which side to truncate on.
                Defaults to "right".
            commonpath (PathLike, optional): If provided, trim this part from
                the start of each choice for display. Defaults to None.
            multiselect (bool, optional): If True, allow multiple selections.
                Defaults to False.
            allow_empty (bool, optional): If True, allow an empty string as a
                valid response. Defaults to False.
        """
        # Assign attributes
        self.commonpath = commonpath

        # Continue the initialization chain
        super().__init__(
            prompt,
            console=console,
            password=password,
            choices=choices,
            case_sensitive=case_sensitive,
            show_default=show_default,
            show_choices=show_choices,
            column_first=column_first,
            truncate_length=truncate_length,
            truncate_side=truncate_side,
            multiselect=multiselect,
            allow_empty=allow_empty,
        )

    # --- Creation ---
    @override
    @classmethod
    def ask(
        cls,
        prompt: TextType = "",
        *,
        console: Optional[Console] = None,
        password: bool = False,
        choices: Optional[List[str]] = None,
        case_sensitive: bool = True,
        show_default: bool = True,
        show_choices: bool = True,
        column_first: bool = False,
        truncate_length: int | None = None,
        truncate_side: Literal["left", "middle", "right"] = "middle",
        commonpath: PathLike | None = None,
        multiselect: bool = False,
        allow_empty: bool = False,
        default: Any = ...,
        stream: Optional[TextIO] = None,
    ) -> Any:
        """Shortcut to construct and run a prompt loop and return the result.

        Example:
            >>> filename = Prompt.ask("Enter a filename")

        Args:
            prompt (TextType, optional): Prompt text. Defaults to "".
            console (Console, optional): A Console instance or None to use
                global console. Defaults to None.
            password (bool, optional): Enable password input. Defaults to False.
            choices (List[str], optional): A list of valid choices.
                Defaults to None.
            case_sensitive (bool, optional): Matching of choices should be
                case-sensitive. Defaults to True.
            show_default (bool, optional): Show default in prompt.
                Defaults to True.
            show_choices (bool, optional): Show choices in prompt.
                Defaults to True.
            column_first (bool, optional): If True, print choices in
                column-first. Defaults to False.
            truncate_length (int | None, optional): Truncate long choices to
                this length for display. Defaults to None (no truncation).
            truncate_side (Literal["left", "middle", "right"], optional): If
                ``truncate_length`` is set, which side to truncate on.
                Defaults to "right".
            commonpath (PathLike, optional): If provided, trim this part from
                the start of each choice for display. Defaults to None.
            multiselect (bool, optional): If True, allow multiple selections.
                Defaults to False.
            allow_empty (bool, optional): If True, allow an empty string as a
                valid response. Defaults to False.
            stream (TextIO, optional): Optional text file open for reading to
                get input. Defaults to None.
        """
        # 1. Prepare choices
        choices = to_list(choices) if choices else None

        # 2. Prepare default value
        if isinstance(default, str) and not is_valid_str(default):
            default = None

        # 3. Create prompt
        _prompt = cls(
            prompt,
            console=console,
            password=password,
            choices=choices,
            case_sensitive=case_sensitive,
            show_default=show_default,
            show_choices=show_choices,
            column_first=column_first,
            truncate_length=truncate_length,
            truncate_side=truncate_side,
            commonpath=commonpath,
            multiselect=multiselect,
            allow_empty=allow_empty,
        )
        return _prompt(default=default, stream=stream)

    # --- Visualization ---
    @override
    def render_choices(self):
        """Print available choices in columns."""
        choices_ = []
        for i, choice in enumerate(self.choices):
            if self.commonpath:
                choice = Path(choice).unique_path_from(self.commonpath)
            if self.truncate_length:
                choice = truncate_string(
                    value=choice,
                    max_length=self.truncate_length,
                    side=self.truncate_side,
                )
            choices_.append(f"{f'{i}.':>6} {choice}")
        columns = Columns(choices_, equal=True, column_first=self.column_first)
        rich.print(columns)


class IntPrompt(PromptBase[int]):
    """A prompt that returns an integer.

    Example:
        >>> burrito_count = IntPrompt.ask("How many burritos do you want to order")
    """

    response_type = int
    validate_error_message = "[prompt.invalid]Please enter a valid integer number"


class FloatPrompt(PromptBase[float]):
    """A prompt that returns a float.

    Example:
        >>> temperature = FloatPrompt.ask("Enter desired temperature")
    """

    response_type = float
    validate_error_message = "[prompt.invalid]Please enter a number"


class Confirm(PromptBase[bool]):
    """A yes / no confirmation prompt.

    Example:
        >>> if Confirm.ask("Continue"):
                run_job()
    """

    response_type = bool
    validate_error_message = "[prompt.invalid]Please enter Y or N"
    choices: List[str] = ["y", "n"]

    # --- Processing ---
    @override
    def process_response(self, value: str) -> bool:
        """Convert choices to a bool."""
        value = value.strip().lower()
        if value not in self.choices:
            raise InvalidResponse(self.validate_error_message)
        return value == self.choices[0]

    # --- Visualization ---
    @override
    def render_default(self, default: DefaultType) -> Text:
        """Render the default as (y) or (n) rather than True/False."""
        yes, no = self.choices
        return Text(f"({yes})" if default else f"({no})", style="prompt.default")

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    value = OptionPrompt.ask(
        "Select Weights",
        choices=[
            "/Volumes/ssd_01/10_workspace/11_code/mon/zoo/cv/classify/mobileone/mobileone_s0/imagenet1k_v1/mobileone_s0_imagenet1k_v1.pth.tar",
            "/Volumes/ssd_01/10_workspace/11_code/mon/zoo/cv/classify/mobileone/mobileone_s2/imagenet1k_v1/mobileone_s2_imagenet1k_v1.pth.tar",
            "/Volumes/ssd_01/10_workspace/11_code/mon/zoo/cv/classify/mobileone/mobileone_s4/imagenet1k_v1/mobileone_s4_imagenet1k_v1.pth.tar",
        ],
        default=0,
        column_first=True,
        truncate_length=60,
        truncate_side="middle",
        multiselect=True,
    )
    print(value)

# endregion
