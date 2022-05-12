#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import os
import socket
import time
from functools import wraps
from typing import Any
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.base import rank_zero_experiment
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.event_pb2 import SessionLog
from tensorboard.summary.writer import event_file_writer
from tensorboard.summary.writer.event_file_writer import _AsyncWriter
from tensorboard.summary.writer.record_writer import RecordWriter
from torch.utils import tensorboard

from one.core import Callable
from one.core import LOGGERS

__all__ = [
    "TensorBoardLogger",
]


# MARK: - EventFileWriter


class EventFileWriter(event_file_writer.EventFileWriter):
    
    # MARK: - Magic Functions
    
    def __init__(
        self,
        logdir         : str,
        max_queue_size : int = 10,
        flush_secs     : int = 120,
        filename_suffix: str = ""
    ):
        self._logdir = logdir
        tf.io.gfile.makedirs(logdir)
        self._file_name = (
            os.path.join(
                logdir, "events.out.tfevents.%s" % (socket.gethostname())
            )
            + filename_suffix
        )  # noqa E128
        self._general_file_writer = tf.io.gfile.GFile(self._file_name, "wb")
        self._async_writer = _AsyncWriter(
            RecordWriter(self._general_file_writer), max_queue_size, flush_secs
        )
    
        # Initialize an event measurement.
        _event = event_pb2.Event(
            wall_time=time.time(), file_version="brain.Event:2"
        )
        self.add_event(_event)
        self.flush()


# MARK: - FileWriter

class FileWriter(tensorboard.FileWriter):
    
    def __init__(
        self,
        log_dir        : str,
        max_queue      : int = 10,
        flush_secs     : int = 120,
        filename_suffix: str = ""
    ):
        """Creates a `FileWriter` and an event file.
        On construction the writer creates a new event file in `log_dir`.
        The other arguments to the constructor control the asynchronous writes
        to the event file.

        Args:
          log_dir: A string. Directory where event file will be written.
          max_queue: Integer. Size of the queue for pending events and
            summaries before one of the 'add' calls forces a flush to disk.
            Default is ten items.
          flush_secs: Number. How often, in seconds, to flush the
            pending events and summaries to disk. Default is every two minutes.
          filename_suffix: A string. Suffix added to all event filenames
            in the log_dir directory. More details on filename construction in
            tensorboard.summary.writer.event_file_writer.EventFileWriter.
        """
        # Sometimes PosixPath is passed in and we need to coerce it to
        # a string in all cases
        # TODO: See if we can remove this in the future if we are
        # actually the ones passing in a PosixPath
        log_dir = str(log_dir)
        self.event_writer = EventFileWriter(
            log_dir, max_queue, flush_secs, filename_suffix)


# MARK: - SummaryWriter

class SummaryWriter(tensorboard.SummaryWriter):
    
    def _get_file_writer(self):
        """Returns the default FileWriter measurement. Recreates it if closed."""
        if self.all_writers is None or self.file_writer is None:
            self.file_writer = FileWriter(
                self.log_dir, self.max_queue, self.flush_secs,
                self.filename_suffix
            )
            self.all_writers = {self.file_writer.get_logdir(): self.file_writer}
            if self.purge_step is not None:
                most_recent_step = self.purge_step
                self.file_writer.add_event(
                    Event(step=most_recent_step, file_version="brain.Event:2")
                )
                self.file_writer.add_event(
                    Event(step=most_recent_step, session_log=SessionLog(status=SessionLog.START))
                )
                self.purge_step = None
        return self.file_writer


# MARK: - TensorBoardLogger

def rank_zero_only(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Optional[Any]:
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped_fn


# TODO: this should be part of the cluster environment
def _get_rank() -> int:
    rank_keys = ("RANK", "SLURM_PROCID", "LOCAL_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


# add the attribute to the function but don't overwrite in case Trainer has already set it
rank_zero_only.rank = getattr(rank_zero_only, "rank", _get_rank())


@LOGGERS.register(name="tensorboard",        force=True)
@LOGGERS.register(name="tensorboard_logger", force=True)
@LOGGERS.register(name="TensorBoardLogger",  force=True)
class TensorBoardLogger(pl.loggers.TensorBoardLogger):
    
    @property
    @rank_zero_experiment
    def experiment(self) -> SummaryWriter:
        r"""
        Actual tensorboard object. To use TensorBoard features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example:
            self.logger.experiment.some_tensorboard_function()
        """
        if self._experiment is not None:
            return self._experiment

        assert rank_zero_only.rank == 0, "tried to init log dirs in non global_rank=0"
        if self.root_dir:
            self._fs.makedirs(self.root_dir, exist_ok=True)
        self._experiment = SummaryWriter(log_dir=self.log_dir, **self._kwargs)
        return self._experiment
