#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.utilities.imports import _RICH_AVAILABLE
from pytorch_lightning.utilities.model_summary import get_human_readable_count

from one.core import CALLBACKS
from one.core import console

if _RICH_AVAILABLE:
    from rich.table import Table

__all__ = [
    "RichModelSummary"
]


@CALLBACKS.register(name="rich_model_summary")
class RichModelSummary(ModelSummary):
    r"""Generates a summary of all layers in a
    :class:`~pytorch_lightning.core.lightning.LightningModule` with `rich text
    formatting <https://github.com/willmcgugan/rich>`_.

    Install it with pip:

    .. code-block:: bash

        pip install rich

    .. code-block:: python

        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import RichModelSummary

        trainer = Trainer(callbacks=RichModelSummary())

    You could also enable `RichModelSummary` using the
    :class:`~pytorch_lightning.callbacks.RichProgressBar`

    .. code-block:: python

        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import RichProgressBar

        trainer = Trainer(callbacks=RichProgressBar())

    Args:
        max_depth (int):
            The maximum depth of layer nesting that the summary will include.
            A value of 0 turns the layer summary off.

    Raises:
        ModuleNotFoundError:
            If required `rich` package is not installed on the device.
    """

    def __init__(self, max_depth: int = 1):
        if not _RICH_AVAILABLE:
            raise ModuleNotFoundError(
                "`RichProgressBar` requires `rich` to be installed. "
                "Install it by running `pip install -U rich`."
            )
        super().__init__(max_depth)

    @staticmethod
    def summarize(
        summary_data        : list[tuple[str, list[str]]],
        total_parameters    : int,
        trainable_parameters: int,
        model_size          : float,
    ):
        # NOTE: Print Modules and Layers
        table = Table(header_style="bold magenta")
        table.add_column(" ", style="dim")
        table.add_column("Name", justify="left", no_wrap=True)
        table.add_column("Type")
        table.add_column("Params", justify="right")

        column_names = list(zip(*summary_data))[0]

        for column_name in ["In sizes", "Out sizes"]:
            if column_name in column_names:
                table.add_column(column_name, justify="right", style="white")

        rows = list(zip(*(arr[1] for arr in summary_data)))
        for row in rows:
            table.add_row(*row)

        console.log(table)

        parameters = []
        for param in [trainable_parameters,
                      total_parameters - trainable_parameters,
                      total_parameters, model_size]:
            parameters.append("{:<{}}".format(
                get_human_readable_count(int(param)), 10)
            )
        
        # NOTE: Print Parameters
        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_column()

        grid.add_row(f"[bold]Trainable params[/]: {parameters[0]}")
        grid.add_row(f"[bold]Non-trainable params[/]: {parameters[1]}")
        grid.add_row(f"[bold]Total params[/]: {parameters[2]}")
        grid.add_row(f"[bold]Total estimated model params size (MB)[/]: {parameters[3]}")

        console.log(grid)
