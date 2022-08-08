#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import BackboneFinetuning
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import ModelPruning
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.callbacks import QuantizationAwareTraining
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.callbacks.progress.rich_progress import \
    BatchesProcessedColumn
from pytorch_lightning.callbacks.progress.rich_progress import CustomBarColumn
from pytorch_lightning.callbacks.progress.rich_progress import CustomProgress
from pytorch_lightning.callbacks.progress.rich_progress import MetricsTextColumn
from pytorch_lightning.callbacks.progress.rich_progress import \
    ProcessingSpeedColumn
from pytorch_lightning.utilities.imports import _RICH_AVAILABLE
from pytorch_lightning.utilities.model_summary import get_human_readable_count
from rich.progress import SpinnerColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn

from one.constants import CALLBACKS
from one.core import console
from one.core import GPUMemoryUsageColumn

if _RICH_AVAILABLE:
    from rich.table import Table


class RichModelSummary(ModelSummary):
    """Generates a summary of all layers in a
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
        max_depth (int): The maximum depth of layer nesting that the summary
            will include. A value of 0 turns the layer summary off.

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
        
        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_column()

        grid.add_row(f"[bold]Trainable params[/]: {parameters[0]}")
        grid.add_row(f"[bold]Non-trainable params[/]: {parameters[1]}")
        grid.add_row(f"[bold]Total params[/]: {parameters[2]}")
        grid.add_row(f"[bold]Total estimated model params size (MB)[/]: {parameters[3]}")

        console.log(grid)


class RichProgressBar(callbacks.RichProgressBar):
    """
    Override `pytorch_lightning.callbacks.progress.rich_progress` to add some
    customizations.
    """
    
    def _init_progress(self, trainer):
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            self._console = console
            # self._console: Console = Console(**self._console_kwargs)
            self._console.clear_live()
            self._metric_component = MetricsTextColumn(
                trainer = trainer,
                style   = self.theme.metrics
            )
            self.progress = CustomProgress(
                *self.configure_columns(trainer),
                self._metric_component,
                auto_refresh = False,
                disable      = self.is_disabled,
                console      = self._console,
            )
            self.progress.start()
            # progress has started
            self._progress_stopped = False
            
    def configure_columns(self, trainer) -> list:
        return [
            TextColumn(
                console.get_datetime().strftime("[%x %H:%M:%S:%f]"),
                justify = "left",
                style   = "log.time"
            ),
            TextColumn("[progress.description][{task.description}]"),
            CustomBarColumn(
                complete_style = self.theme.progress_bar,
                finished_style = self.theme.progress_bar_finished,
                pulse_style    = self.theme.progress_bar_pulse,
            ),
            BatchesProcessedColumn(style="progress.download"),
            "•",
            GPUMemoryUsageColumn(),
            "•",
            ProcessingSpeedColumn(style="progress.data.speed"),
            "•",
            TimeRemainingColumn(),
            ">",
            TimeElapsedColumn(),
            SpinnerColumn(),
        ]


CALLBACKS.register(name="backbone_finetuning",             module=BackboneFinetuning)
CALLBACKS.register(name="device_stats_monitor",            module=DeviceStatsMonitor)
CALLBACKS.register(name="early_stopping",                  module=EarlyStopping)
CALLBACKS.register(name="gradient_accumulation_scheduler", module=GradientAccumulationScheduler)
CALLBACKS.register(name="learning_rate_monitor",           module=LearningRateMonitor)
CALLBACKS.register(name="model_checkpoint",                module=ModelCheckpoint)
CALLBACKS.register(name="model_pruning",                   module=ModelPruning)
CALLBACKS.register(name="model_summary",                   module=ModelSummary)
CALLBACKS.register(name="quantization_aware_training",     module=QuantizationAwareTraining)
CALLBACKS.register(name="rich_model_summary",              module=RichModelSummary)
CALLBACKS.register(name="rich_progress_bar",               module=RichProgressBar)
CALLBACKS.register(name="stochastic_weight_averaging",     module=StochasticWeightAveraging)
