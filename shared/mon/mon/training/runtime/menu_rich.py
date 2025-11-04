#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements a rich-text CLI menu using ``rich`` package."""

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
    """Wrap around ``core.rich.prompt`` with additional values parsing functionality."""
    
    def __init__(self, text: str, default: str, choices: Sequence | Collection = None):
        self.text    = text
        self.default = default
        self.choices = choices
        self.value   = None
    
    @property
    def default(self):
        """Returns the default value."""
        return self._default
    
    @default.setter
    def default(self, default: str):
        """Sets the default value."""
        self._default = str(default) if default else ""
    
    @property
    def value(self):
        """Returns the current input value."""
        return self._value
    
    @value.setter
    def value(self, value: str):
        """Parses the user's input.
        
        If the choice is an integer (i.e., list index), it returns the corresponding
        option from the list of options. Otherwise, it returns the choice as is.
        """
        if value:
            value = value[0] if isinstance(value, list | tuple) and len(value) == 1 else value
        else:
            value = ""
        self._value = value
        
    @property
    def choices(self) -> list[str]:
        """List of choices to display."""
        return self._choices
    
    @choices.setter
    def choices(self, choices: Sequence | Collection = None):
        """Set list of choices to display."""
        self._choices = to_list(choices) or None
    
    def prompt(self) -> Any:
        """Prompts the user for a choice."""
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
    """Wrap around ``core.rich.prompt`` with additional values parsing functionality."""
    
    def __init__(self, text: str, default: bool = True):
        self.text    = text
        self.default = default
        self.value   = default
    
    def prompt(self) -> bool:
        self.value = prompt.Confirm().ask(prompt=self.text, default=self.default)
        return self.value


class NumberPrompt:
    """Wrap around ``core.rich.prompt`` with additional values parsing functionality."""
    
    def __init__(self, text: str, default: int = -1):
        self.text    = text
        self.default = default
        self.value   = default

    @property
    def default(self):
        return self._default
    
    @default.setter
    def default(self, default: int):
        default       = default[0] if isinstance(default, list | tuple) else default
        default       = to_int(default)
        self._default = default if isinstance(default, int | float) else -1

    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, value: int):
        value       = value[0] if isinstance(value, list | tuple) else value
        value       = to_int(value)
        self._value = None if isinstance(value, int | float) and value < 0 else value
        
    def prompt(self) -> int:
        self.value = prompt.IntPrompt().ask(prompt=self.text, default=self.default)
        return self.value


# ----- Predefined Prompts -----
class TaskPrompt(Prompt):
    
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
        return self._value
    
    @value.setter
    def value(self, value: Any):
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
        """Prompts the user for a choice."""
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
    def value(self):
        return self._value
    
    @value.setter
    def value(self, value: str):
        if value:
            value = to_list(value)
        else:
            value = []
        self._value = value


class FullnamePrompt(Prompt):
    
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
    
    def __init__(self, defaults: dict = None):
        self.index = 0
        self.args  = DEFAULT_ARGS
        self.args.update(defaults or {})
        self.config_args = {}

    def __len__(self):
        return 27

    def cycle_next(self):
        """Move to the next option, wrapping around if needed."""
        self.index = (self.index + 1) % self.__len__()

    def cycle_prev(self):
        """Move to the previous option, wrapping around if needed."""
        self.index = (self.index - 1) % self.__len__()

    def display_prompt(self):
        if self.index == 0:
            # clear_terminal()
            console.rule(f"[bold red]Input Prompts")
        else:
            console.rule()

        if self.index == 0:  # Task
            self.args["task"] = TaskPrompt(
                project_root = self.args["root"],
                default      = self.args["task"],
            ).prompt()
        if self.index == 1:  # Mode
            self.args["mode"] = Prompt(
                text    = CLI_OPTIONS["mode"]["prompt_text"],
                default = self.args["mode"],
                choices = CLI_OPTIONS["mode"]["choices"],
            ).prompt()
        if self.index == 2:  # Arch
            self.args["arch"] = ArchPrompt(
                task         = self.args["task"],
                mode         = self.args["mode"],
                project_root = self.args["root"],
                default      = self.args["arch"],
            ).prompt()
        if self.index == 3:  # Model
            self.args["model"] = ModelPrompt(
                task         = self.args["task"],
                mode         = self.args["mode"],
                arch         = self.args["arch"],
                project_root = self.args["root"],
                default      = self.args["model"],
            ).prompt()
        if self.index == 4:  # Config
            self.args["config"] = ConfigPrompt(
                project_root = self.args["root"],
                arch         = self.args["arch"],
                model        = self.args["model"],
                default      = self.args["config"],
            ).prompt()
            self.config_args = load_config(self.args["config"], False)
        if self.index == 5:  # Weights
            self.args["weights"] = WeightsPrompt(
                model        = self.args["model"],
                project_root = self.args["root"],
                default      = self.args["weights"] or self.config_args.get("weights"),
            ).prompt()
        if self.index == 6:  # Data
            if self.args["mode"] not in ["predict"]:
                self.cycle_next()
            else:
                self.args["data"] = DataPrompt(
                    task         = self.args["task"],
                    project_root = self.args["root"],
                    default      = self.args["data"],
                ).prompt()
        if self.index == 7:  # Fullname
            self.args["fullname"] = FullnamePrompt(
                config  = self.args["config"],
                model   = self.args["model"],
                default = self.args["fullname"] or self.config_args.get("fullname"),
            ).prompt()
        if self.index == 8:  # Device
            self.args["device"] = DevicePrompt(
                model   = self.args["model"],
                mode    = self.args["mode"],
                task    = self.args["task"],
                default = self.args["device"],
            ).prompt()
        if self.index == 9:  # Seed
            self.args["seed"] = NumberPrompt(
                text    = CLI_OPTIONS["seed"]["prompt_text"],
                default = self.args["seed"] or self.config_args.get("seed"),
            ).prompt()
        if self.index == 10:  # Image Size
            if self.args["mode"] not in ["predict", "speed"]:
                self.cycle_next()
            else:
                self.args["imgsz"] = NumberPrompt(
                    text    = CLI_OPTIONS["imgsz"]["prompt_text"],
                    default = self.args["imgsz"] or self.config_args.get("imgsz"),
                ).prompt()
        if self.index == 11:  # Epochs
            if self.args["mode"] not in ["train"]:
                self.cycle_next()
            else:
                self.args["epochs"] = NumberPrompt(
                    text    = CLI_OPTIONS["epochs"]["prompt_text"],
                    default = self.args["epochs"] or self.config_args.get("epochs"),
                ).prompt()
        if self.index == 12:  # Batch Size
            if self.args["mode"] not in ["train"]:
                self.cycle_next()
            else:
                self.args["batch_size"] = NumberPrompt(
                    text    = CLI_OPTIONS["batch_size"]["prompt_text"],
                    default = self.args["batch_size"] or self.config_args.get("batch_size"),
                ).prompt()
        if self.index == 13:  # torchrun
            if self.args["mode"] not in ["train"]:
                self.cycle_next()
            else:
                self.args["torchrun"] = Confirm(
                    text    = CLI_OPTIONS["torchrun"]["prompt_text"],
                    default = self.args["torchrun"] or self.config_args.get("torchrun", False),
                ).prompt()
        if self.index == 14:  # Master Port
            if self.args["mode"] not in ["train"] or not self.args["torchrun"]:
                self.cycle_next()
            else:
                self.args["master_port"] = NumberPrompt(
                    text    = CLI_OPTIONS["master_port"]["prompt_text"],
                    default = self.args["master_port"] or self.config_args.get("master_port"),
                ).prompt()
        if self.index == 15:  # Master Address
            if self.args["mode"] not in ["train"] or not self.args["torchrun"]:
                self.cycle_next()
            else:
                self.args["master_addr"] = Prompt(
                    text    = CLI_OPTIONS["master_addr"]["prompt_text"],
                    default = self.args["master_addr"] or self.config_args.get("master_addr"),
                ).prompt()
        if self.index == 16:  # Resize
            if self.args["mode"] not in ["predict", "speed"]:
                self.cycle_next()
            else:
                self.args["resize"] = Confirm(
                    text    = CLI_OPTIONS["resize"]["prompt_text"],
                    default = self.args["resize"] or self.config_args.get("resize", False),
                ).prompt()
        if self.index == 17:  # Benchmark
            self.args["benchmark"] = Confirm(
                text    = CLI_OPTIONS["benchmark"]["prompt_text"],
                default = self.args["benchmark"] or self.config_args.get("benchmark", False),
            ).prompt()
        if self.index == 18:  # Save Result
            if self.args["mode"] in ["speed"]:
                self.args["save_result"] = Confirm(
                    text    = CLI_OPTIONS["save_result"]["prompt_text"],
                    default = self.args["save_result"],
                ).prompt()
            else:
                self.args["save_result"] = Confirm(
                    text    = CLI_OPTIONS["save_result"]["prompt_text"],
                    default = self.args["save_result"] or self.config_args.get("save_result", False),
                ).prompt()
        if self.index == 19:  # Save Image
            if self.args["mode"] in ["speed"]:
                self.args["save_image"] = Confirm(
                    text    = CLI_OPTIONS["save_image"]["prompt_text"],
                    default = self.args["save_image"],
                ).prompt()
            else:
                self.args["save_image"] = Confirm(
                    text    = CLI_OPTIONS["save_image"]["prompt_text"],
                    default = self.args["save_image"] or self.config_args.get("save_image", False),
                ).prompt()
        if self.index == 20:  # Save Debug
            if self.args["mode"] in ["speed"]:
                self.args["save_debug"] = Confirm(
                    text    = CLI_OPTIONS["save_debug"]["prompt_text"],
                    default = self.args["save_debug"],
                ).prompt()
            else:
                self.args["save_debug"] = Confirm(
                    text    = CLI_OPTIONS["save_debug"]["prompt_text"],
                    default = self.args["save_debug"] or self.config_args.get("save_debug", False),
                ).prompt()
        if self.index == 21:  # Use Fullname
            self.args["use_fullname"] = Confirm(
                text    = CLI_OPTIONS["use_fullname"]["prompt_text"],
                default = self.args["use_fullname"] or self.config_args.get("use_fullname", False),
            ).prompt()
        if self.index == 22:  # Keep Subdirs
            self.args["keep_subdirs"] = Confirm(
                text    = CLI_OPTIONS["keep_subdirs"]["prompt_text"],
                default = self.args["keep_subdirs"] or self.config_args.get("keep_subdirs", False),
            ).prompt()
        if self.index == 23:  # Save Nearby
            if self.args["mode"] not in ["predict"]:
                self.cycle_next()
            else:
                self.args["save_nearby"] = Confirm(
                    text    = CLI_OPTIONS["save_nearby"]["prompt_text"],
                    default = self.args["save_nearby"] or self.config_args.get("save_nearby", False),
                ).prompt()
        if self.index == 24:  # Exist OK?
            self.args["exist_ok"] = Confirm(
                text    = CLI_OPTIONS["exist_ok"]["prompt_text"],
                default = self.args["exist_ok"] or self.config_args.get("exist_ok", False),
            ).prompt()
        if self.index == 25:  # Use Verbose
            self.args["verbose"] = Confirm(
                text    = CLI_OPTIONS["verbose"]["prompt_text"],
                default = self.args["verbose"] or self.config_args.get("verbose", False),
            ).prompt()
        if self.index == 26:  # Finish
            rprint_dict(self.args, title="Input Arguments")
            finish = Confirm(text="Finish/Re-input", default=True).prompt()
            if finish:
                self.index = self.__len__()
            
    def prompt_args(self) -> dict | box.Box:
        """Run the interactive menu and return the selected option."""
        while True:
            self.display_prompt()
            if self.index == self.__len__():
                return self.args
            self.cycle_next()
