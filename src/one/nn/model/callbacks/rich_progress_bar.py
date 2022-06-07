#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.progress.rich_progress import \
	BatchesProcessedColumn
from pytorch_lightning.callbacks.progress.rich_progress import CustomBarColumn
from pytorch_lightning.callbacks.progress.rich_progress import CustomProgress
from pytorch_lightning.callbacks.progress.rich_progress import MetricsTextColumn
from pytorch_lightning.callbacks.progress.rich_progress import \
	ProcessingSpeedColumn
from rich.progress import SpinnerColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn

from one.core import CALLBACKS
from one.core import console
from one.core import GPUMemoryUsageColumn

__all__ = [
    "RichProgressBar"
]


# MARK: - RichProgressBar

@CALLBACKS.register(name="rich_progress_bar")
class RichProgressBar(callbacks.RichProgressBar):
	"""Override `pytorch_lightning.callbacks.progress.rich_progress` to add
	some customizations.
	"""
	
	# MARK: Configure
	
	def _init_progress(self, trainer):
		if self.is_enabled and (self.progress is None or self._progress_stopped):
			self._reset_progress_bar_ids()
			self._console = console
			# self._console: Console = Console(**self._console_kwargs)
			self._console.clear_live()
			self._metric_component = MetricsTextColumn(trainer, self.theme.metrics)
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
