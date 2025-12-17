#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for interactive CLI runtime menu.

This module implements an interactive command-line interface (CLI) menu using Rich
prompts. It allows users to select options for tasks, models, configurations, and
various runtime parameters in a guided manner.
"""

__all__ = [
    "RunCLI",
]

from typing import Any, Collection, Sequence

import box
from rich import prompt

from mon.core.console import console, rprint_dict
from mon.core.pathlib import Path
from mon.core.rich import SelectionOrInputPrompt
from mon.core.utils import is_int, to_int, to_list, to_str
from .options import CLI_OPTIONS, DEFAULT_ARGS
from .utils import (
    list_archs,
    list_config_files,
    list_datasets,
    list_models,
    list_tasks,
    list_weights_files,
    load_config,
    parse_model_dir,
    parse_weights_file,
)


# ----- Base Prompts -----
class Prompt:
    """A class that wraps around core.rich.prompt with additional values parsing
    functionality.
    """
    
    def __init__(self, text: str, default: str, choices: Sequence | Collection = None):
        """Initializes the Prompt instance.
        
        Args:
            text (str): The prompt text to display to the user.
            default (str): The default value if the user provides no input.
            choices (Sequence or Collection, optional): A list of choices to 
                present to the user. Defaults to None.
        """
        self.text    = text
        self.default = default
        self.choices = choices
        self.value   = None
    
    @property
    def default(self) -> str:
        """Getter for the default value.
        
        Returns:
            str: The default value.
        """
        return self._default
    
    @default.setter
    def default(self, default: str):
        """Setter for the default value.
        
        Args:
            default (str): The default value to set.
        """
        self._default = str(default) if default else ""
    
    @property
    def value(self) -> str:
        """Getter for the user's input value.
        
        Returns:
            str: The user's input value.
        """
        return self._value
    
    @value.setter
    def value(self, value: str):
        """Setter for the user's input value.
        
        Args:
            value (str): The user's input value to set.
        """
        if value:
            value = value[0] if isinstance(value, list | tuple) and len(value) == 1 else value
        else:
            value = ""
        self._value = value
        
    @property
    def choices(self) -> list[str]:
        """Getter for the list of choices to display.
        
        Returns:
            list[str]: The list of choices.
        """
        return self._choices
    
    @choices.setter
    def choices(self, choices: Sequence | Collection = None):
        """Setter for the list of choices to display.
        
        Args:
            choices (Sequence or Collection, optional): The list of choices to set.
                Defaults to None.
        """
        self._choices = to_list(choices) or None
    
    def prompt(self) -> Any:
        """Prompts the user for a choice.
        
        Returns:
            Any: The user's selected or input value.
        """
        kwargs = {
            "prompt"        : self.text,
            "case_sensitive": True,
            "show_default"  : True,
            "show_choices"  : True,
            "allow_empty"   : False,
            "column_first"  : False,
            "default"       : self.default,
        }
        if self._choices and len(self._choices) > 0:
            kwargs["choices"] = self._choices
        self.value = SelectionOrInputPrompt().ask(**kwargs)
        return self.value


class Confirm:
    """A class that wraps around core.rich.prompt.Confirm."""
    
    def __init__(self, text: str, default: bool = True):
        """Initializes the Confirm instance.
        
        Args:
            text (str): The prompt text to display to the user.
            default (bool, optional): The default value if the user provides no
                input. Defaults to True.
        """
        self.text    = text
        self.default = default
        self.value   = default
    
    def prompt(self) -> bool:
        """Prompts the user for a confirmation (yes/no)."""
        self.value = prompt.Confirm().ask(prompt=self.text, default=self.default)
        return self.value


class NumberPrompt:
    """A class that wraps around core.rich.prompt.IntPrompt."""
    
    def __init__(self, text: str, default: int = -1):
        """Initializes the NumberPrompt instance.
        
        Args:
            text (str): The prompt text to display to the user.
            default (int, optional): The default value if the user provides no
                input. Defaults to -1.
        """
        self.text    = text
        self.default = default
        self.value   = default

    @property
    def default(self):
        """Getter for the default value.
        
        Returns:
            int: The default value.
        """
        return self._default
    
    @default.setter
    def default(self, default: int):
        """Setter for the default value.
        
        Args:
            default (int): The default value to set.
        """
        default       = default[0] if isinstance(default, list | tuple) else default
        default       = to_int(default)
        self._default = default if isinstance(default, int | float) else -1

    @property
    def value(self) -> int:
        """Getter for the user's input value.
        
        Returns:
            int: The user's input value.
        """
        return self._value
    
    @value.setter
    def value(self, value: int):
        """Setter for the user's input value.
        
        Args:
            value (int): The user's input value to set.
        """
        value       = value[0] if isinstance(value, list | tuple) else value
        value       = to_int(value)
        self._value = None if isinstance(value, int | float) and value < 0 else value
        
    def prompt(self) -> int:
        """Prompts the user for a number."""
        self.value = prompt.IntPrompt().ask(prompt=self.text, default=self.default)
        return self.value


# ----- Predefined Prompts -----
class TaskPrompt(Prompt):
    """A prompt for selecting a task."""
    
    def __init__(
        self,
        project_root: str | Path,
        text        : str = CLI_OPTIONS["task"]["prompt_text"],
        default     : str = CLI_OPTIONS["task"]["default"],
        choices     : Sequence | Collection = None,
    ):
        choices = choices or list_tasks(project_root=project_root)
        super().__init__(text=text, default=default, choices=choices)


class ArchPrompt(Prompt):
    """A prompt for selecting a model architecture."""
    
    def __init__(
        self,
        task        : str,
        mode        : str,
        project_root: str | Path,
        text        : str = CLI_OPTIONS["arch"]["prompt_text"],
        default     : str = CLI_OPTIONS["arch"]["default"],
        choices     : Sequence | Collection = None,
    ):
        choices = choices or list_archs(task=task, mode=mode, project_root=project_root)
        super().__init__(text=text, default=default, choices=choices)


class ModelPrompt(Prompt):
    """A prompt for selecting a model."""
    
    def __init__(
        self,
        task        : str,
        mode        : str,
        arch        : str,
        project_root: str | Path,
        text        : str = CLI_OPTIONS["model"]["prompt_text"],
        default     : str = CLI_OPTIONS["model"]["default"],
        choices     : Sequence | Collection = None,
    ):
        choices = choices or list_models(task=task, mode=mode, arch=arch, project_root=project_root)
        super().__init__(text=text, default=default, choices=choices)


class ConfigPrompt(Prompt):
    """A prompt for selecting a configuration file."""
    
    def __init__(
        self,
        project_root: str | Path,
        arch        : str,
        model       : str,
        text        : str = CLI_OPTIONS["config"]["prompt_text"],
        default     : str = CLI_OPTIONS["config"]["default"],
        choices     : Sequence | Collection = None,
    ):
        choices = choices or list_config_files(
            project_root  = project_root,
            model_root    = parse_model_dir(arch, model),
            model         = model,
            absolute_path = True
        )
        choices = [str(c) for c in choices]
        super().__init__(text=text, default=default, choices=choices)


class WeightsPrompt(Prompt):
    """A prompt for selecting a weights file."""
    
    def __init__(
        self,
        model       : str,
        project_root: str | Path,
        text        : str = CLI_OPTIONS["weights"]["prompt_text"],
        default     : str = CLI_OPTIONS["weights"]["default"],
        choices     : Sequence | Collection = None,
    ):
        default = (parse_weights_file(project_root, default))
        default = str(default) if default else None
        choices = choices or list_weights_files(model=model, project_root=project_root)
        choices = [str(c) for c in choices]
        super().__init__(text=text, default=default, choices=choices)
    
    @property
    def value(self):
        """Getter for the user's input value.
        
        Returns:
            Any: The user's input value.
        """
        return self._value
    
    @value.setter
    def value(self, value: Any):
        """Setter for the user's input value.
        
        Args:
            value (Any): The user's input value to set.
        """
        value = value if value not in [None, ""] else None
        if value:
            if isinstance(value, str):
                value = to_list(value)
            if self.choices and len(self.choices) > 0:
                value = [self.choices[int(w)] if is_int(w) else w for w in value]
                value = [w.replace("'", "") for w in value]
            value = value[0] if len(value) == 1 else value
        self._value = value

    def prompt(self) -> Any:
        """Prompts the user for a choice.
        
        Returns:
            Any: The user's selected or input value.
        """
        kwargs = {
            "prompt"        : self.text,
            "case_sensitive": True,
            "show_default"  : True,
            "show_choices"  : True,
            "allow_empty"   : True,
            "default"       : self.default,
        }
        if self._choices and len(self._choices) > 0:
            kwargs["choices"] = self._choices
        self.value = SelectionOrInputPrompt().ask(**kwargs)
        return self.value


class DataPrompt(Prompt):
    """A prompt for selecting a dataset."""
    
    def __init__(
        self,
        task        : str,
        project_root: str | Path,
        text        : str = CLI_OPTIONS["data"]["prompt_text"],
        default     : str = CLI_OPTIONS["data"]["default"],
        choices     : Sequence | Collection = None,
    ):
        default = to_str(default, sep=", ")
        # default = wrap_str(default, max_length=get_terminal_size()[0])
        choices = choices or list_datasets(task=task, mode="predict", project_root=project_root)
        super().__init__(text=text, default=default, choices=choices)
    
    @property
    def value(self) -> str:
        """Getter for the user's input value.
        
        Returns:
            str: The user's input value.
        """
        return self._value
    
    @value.setter
    def value(self, value: str):
        """Setter for the user's input value.
        
        Args:
            value (str): The user's input value to set.
        """
        if value:
            value = to_list(value)
        else:
            value = []
        self._value = value


class FullnamePrompt(Prompt):
    """A prompt for specifying a model fullname."""
    
    def __init__(
        self,
        config : str,
        model  : str,
        text   : str = CLI_OPTIONS["fullname"]["prompt_text"],
        default: str = CLI_OPTIONS["fullname"]["default"],
    ):
        default = default or (Path(config).stem if config not in [None, "None", ""] else model)
        super().__init__(text=text, default=default)


class DevicePrompt(Prompt):
    """A prompt for selecting a device."""
    
    def __init__(
        self,
        model  : str,
        mode   : str,
        task   : str,
        text   : str  = CLI_OPTIONS["device"]["prompt_text"],
        default: str  = CLI_OPTIONS["device"]["default"],
        choices: list = CLI_OPTIONS["device"]["choices"],
    ):
        default = default or "cuda:0"
        choices = choices or CLI_OPTIONS["device"]["choices"]
        super().__init__(text=text, default=default, choices=choices)


# ----- Interactive CLI -----
class RunCLI:
    """An interactive CLI menu for selecting runtime options."""
    
    def __init__(self, defaults: dict = None):
        """Initializes the RunCLI instance.
        
        Args:
            defaults (dict, optional): A dictionary of default argument values.
                Defaults to None.
        """
        self._index = 0
        self._args  = DEFAULT_ARGS
        self._args.update(defaults or {})
        self._config_args = {}
    
    def __len__(self) -> int:
        """Returns the number of options in the menu."""
        return 27

    @property
    def args(self) -> dict:
        """Getter for the selected arguments.
        
        Returns:
            dict: The selected arguments.
        """
        return self._args
    
    @property
    def config_args(self) -> dict:
        """Getter for the loaded configuration arguments.
        
        Returns:
            dict: The loaded configuration arguments.
        """
        return self._config_args
    
    def _next(self):
        """Moves to the next option, wrapping around if needed."""
        self._index = (self._index + 1) % self.__len__()

    def _prev(self):
        """Moves to the previous option, wrapping around if needed."""
        self._index = (self._index - 1) % self.__len__()

    def _display_prompt(self):
        """Displays the prompt for the current option and handles user input."""
        if self._index == 0:
            # clear_terminal()
            console.rule(f"[bold red]Input Prompts")
        else:
            console.rule()

        if self._index == 0:  # Task
            self._args["task"] = TaskPrompt(
                project_root = self._args["root"],
                default      = self._args["task"],
            ).prompt()
        if self._index == 1:  # Mode
            self._args["mode"] = Prompt(
                text    = CLI_OPTIONS["mode"]["prompt_text"],
                default = self._args["mode"],
                choices = CLI_OPTIONS["mode"]["choices"],
            ).prompt()
        if self._index == 2:  # Arch
            self._args["arch"] = ArchPrompt(
                task         = self._args["task"],
                mode         = self._args["mode"],
                project_root = self._args["root"],
                default      = self._args["arch"],
            ).prompt()
        if self._index == 3:  # Model
            self._args["model"] = ModelPrompt(
                task         = self._args["task"],
                mode         = self._args["mode"],
                arch         = self._args["arch"],
                project_root = self._args["root"],
                default      = self._args["model"],
            ).prompt()
        if self._index == 4:  # Config
            self._args["config"] = ConfigPrompt(
                project_root = self._args["root"],
                arch         = self._args["arch"],
                model        = self._args["model"],
                default      = self._args["config"],
            ).prompt()
            self._config_args = load_config(self._args["config"], False)
        if self._index == 5:  # Weights
            self._args["weights"] = WeightsPrompt(
                model        = self._args["model"],
                project_root = self._args["root"],
                default      = self._args["weights"] or self._config_args.get("weights"),
            ).prompt()
        if self._index == 6:  # Data
            if self._args["mode"] not in ["predict"]:
                self._next()
            else:
                self._args["data"] = DataPrompt(
                    task         = self._args["task"],
                    project_root = self._args["root"],
                    default      = self._args["data"],
                ).prompt()
        if self._index == 7:  # Fullname
            self._args["fullname"] = FullnamePrompt(
                config  = self._args["config"],
                model   = self._args["model"],
                default = self._args["fullname"] or self._config_args.get("fullname"),
            ).prompt()
        if self._index == 8:  # Device
            self._args["device"] = DevicePrompt(
                model   = self._args["model"],
                mode    = self._args["mode"],
                task    = self._args["task"],
                default = self._args["device"],
            ).prompt()
        if self._index == 9:  # Seed
            self._args["seed"] = NumberPrompt(
                text    = CLI_OPTIONS["seed"]["prompt_text"],
                default = self._args["seed"] or self._config_args.get("seed"),
            ).prompt()
        if self._index == 10:  # Image Size
            if self._args["mode"] not in ["predict", "speed"]:
                self._next()
            else:
                self._args["imgsz"] = NumberPrompt(
                    text    = CLI_OPTIONS["imgsz"]["prompt_text"],
                    default = self._args["imgsz"] or self._config_args.get("imgsz"),
                ).prompt()
        if self._index == 11:  # Epochs
            if self._args["mode"] not in ["train"]:
                self._next()
            else:
                self._args["epochs"] = NumberPrompt(
                    text    = CLI_OPTIONS["epochs"]["prompt_text"],
                    default = self._args["epochs"] or self._config_args.get("epochs"),
                ).prompt()
        if self._index == 12:  # Batch Size
            if self._args["mode"] not in ["train"]:
                self._next()
            else:
                self._args["batch_size"] = NumberPrompt(
                    text    = CLI_OPTIONS["batch_size"]["prompt_text"],
                    default = self._args["batch_size"] or self._config_args.get("batch_size"),
                ).prompt()
        if self._index == 13:  # torchrun
            if self._args["mode"] not in ["train"]:
                self._next()
            else:
                self._args["torchrun"] = Confirm(
                    text    = CLI_OPTIONS["torchrun"]["prompt_text"],
                    default = self._args["torchrun"] or self._config_args.get("torchrun", False),
                ).prompt()
        if self._index == 14:  # Master Port
            if self._args["mode"] not in ["train"] or not self._args["torchrun"]:
                self._next()
            else:
                self._args["master_port"] = NumberPrompt(
                    text    = CLI_OPTIONS["master_port"]["prompt_text"],
                    default = self._args["master_port"] or self._config_args.get("master_port"),
                ).prompt()
        if self._index == 15:  # Master Address
            if self._args["mode"] not in ["train"] or not self._args["torchrun"]:
                self._next()
            else:
                self._args["master_addr"] = Prompt(
                    text    = CLI_OPTIONS["master_addr"]["prompt_text"],
                    default = self._args["master_addr"] or self._config_args.get("master_addr"),
                ).prompt()
        if self._index == 16:  # Resize
            if self._args["mode"] not in ["predict", "speed"]:
                self._next()
            else:
                self._args["resize"] = Confirm(
                    text    = CLI_OPTIONS["resize"]["prompt_text"],
                    default = self._args["resize"] or self._config_args.get("resize", False),
                ).prompt()
        if self._index == 17:  # Benchmark
            self._args["benchmark"] = Confirm(
                text    = CLI_OPTIONS["benchmark"]["prompt_text"],
                default = self._args["benchmark"] or self._config_args.get("benchmark", False),
            ).prompt()
        if self._index == 18:  # Save Result
            if self._args["mode"] in ["speed"]:
                self._args["save_result"] = Confirm(
                    text    = CLI_OPTIONS["save_result"]["prompt_text"],
                    default = self._args["save_result"],
                ).prompt()
            else:
                self._args["save_result"] = Confirm(
                    text    = CLI_OPTIONS["save_result"]["prompt_text"],
                    default = self._args["save_result"] or self._config_args.get("save_result", False),
                ).prompt()
        if self._index == 19:  # Save Image
            if self._args["mode"] in ["speed"]:
                self._args["save_image"] = Confirm(
                    text    = CLI_OPTIONS["save_image"]["prompt_text"],
                    default = self._args["save_image"],
                ).prompt()
            else:
                self._args["save_image"] = Confirm(
                    text    = CLI_OPTIONS["save_image"]["prompt_text"],
                    default = self._args["save_image"] or self._config_args.get("save_image", False),
                ).prompt()
        if self._index == 20:  # Save Debug
            if self._args["mode"] in ["speed"]:
                self._args["save_debug"] = Confirm(
                    text    = CLI_OPTIONS["save_debug"]["prompt_text"],
                    default = self._args["save_debug"],
                ).prompt()
            else:
                self._args["save_debug"] = Confirm(
                    text    = CLI_OPTIONS["save_debug"]["prompt_text"],
                    default = self._args["save_debug"] or self._config_args.get("save_debug", False),
                ).prompt()
        if self._index == 21:  # Use Fullname
            self._args["use_fullname"] = Confirm(
                text    = CLI_OPTIONS["use_fullname"]["prompt_text"],
                default = self._args["use_fullname"] or self._config_args.get("use_fullname", False),
            ).prompt()
        if self._index == 22:  # Keep Subdirs
            self._args["keep_subdirs"] = Confirm(
                text    = CLI_OPTIONS["keep_subdirs"]["prompt_text"],
                default = self._args["keep_subdirs"] or self._config_args.get("keep_subdirs", False),
            ).prompt()
        if self._index == 23:  # Save Nearby
            if self._args["mode"] not in ["predict"]:
                self._next()
            else:
                self._args["save_nearby"] = Confirm(
                    text    = CLI_OPTIONS["save_nearby"]["prompt_text"],
                    default = self._args["save_nearby"] or self._config_args.get("save_nearby", False),
                ).prompt()
        if self._index == 24:  # Exist OK?
            self._args["exist_ok"] = Confirm(
                text    = CLI_OPTIONS["exist_ok"]["prompt_text"],
                default = self._args["exist_ok"] or self._config_args.get("exist_ok", False),
            ).prompt()
        if self._index == 25:  # Use Verbose
            self._args["verbose"] = Confirm(
                text    = CLI_OPTIONS["verbose"]["prompt_text"],
                default = self._args["verbose"] or self._config_args.get("verbose", False),
            ).prompt()
        if self._index == 26:  # Finish
            rprint_dict(self._args, title="Input Arguments")
            finish = Confirm(text="Finish/Re-input", default=True).prompt()
            if finish:
                self._index = self.__len__()
    
    def prompt(self) -> dict | box.Box:
        """Runs the interactive menu and return the selected option.
        
        Returns:
            dict or box.Box: The selected arguments.
        """
        while True:
            self._display_prompt()
            if self._index == self.__len__():
                return self.args
            self._next()
