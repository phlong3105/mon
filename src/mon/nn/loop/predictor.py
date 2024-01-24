#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the inferring procedure."""

from __future__ import annotations

__all__ = [
    "Inferrer",
]

import datetime
from abc import ABC, abstractmethod
from typing import Any

import torch

from mon import core
from mon.globals import ModelPhase
from mon.nn import model

console = core.console


class Inferrer(ABC):
    """The base class for all inferrers."""
    
    def __init__(
        self,
        source     : core.Path | None = None,
        root       : core.Path        = core.Path() / "infer",
        project    : str              = "",
        name       : str              = "exp",
        max_samples: int | None       = None,
        batch_size : int              = 1,
        image_size : int | list[int]  = 256,
        device     : int | str        = "cpu",
        phase      : ModelPhase | str = "training",
        tensorrt   : bool             = True,
        save       : bool             = True,
        verbose    : bool             = True,
    ):
        self.source      = core.Path(source) if source is not None else source
        self.root        = core.Path(root)
        self.project     = project
        self.shape       = image_size
        self.max_samples = max_samples
        self.batch_size  = batch_size
        self.device      = core.select_device(device=device, batch_size=batch_size)
        self.phase       = phase
        self.tensorrt    = tensorrt
        self.save        = save
        self.verbose     = verbose
        
        self.model: model.Model | None = None
        self.data_loader = None
        self.data_writer = None
        self.logger      = None
        
        if self.project is not None and self.project != "":
            self.root = self.root / self.project
        next_version    = core.get_next_version(path=self.root, prefix=name)
        self.name       = f"{name}-{next_version}"
        self.output_dir = self.root / self.name
        
        console.log(f"Using: {self.device}.")
    
    @property
    def phase(self) -> ModelPhase:
        return self._phase
    
    @phase.setter
    def phase(self, phase: ModelPhase | str = "training"):
        self._phase = ModelPhase.from_value(value=phase)
    
    @property
    def root(self) -> core.Path:
        return self._root
    
    @root.setter
    def root(self, root: core.Path):
        self._root = core.Path(root)
    
    @abstractmethod
    def init_data_loader(self):
        """Initialize the :attr:`data_loader`."""
        pass
    
    @abstractmethod
    def init_data_writer(self):
        """Initialize the :attr:`data_writer`."""
        pass
    
    def init_logger(self):
        """Initialize the :attr:`logger`."""
        self.logger = open(self.output_dir / "log.txt", "a", encoding="utf-8")
        self.logger.write(
            f"\n================================================================================\n"
        )
        self.logger.write(
            f"{datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}\n\n"
        )
        self.logger.flush()
    
    @abstractmethod
    def run(self, model: model.Model, source: Any):
        """Start the inference loop.
        
        The loop must follow these steps:
        1. Get the input.
        2. Preprocess the input.
        3. Run model forward pass.
        4. Post-process the prediction.

        Args:
            model: A model.
            source: A data source. It can be:
                - :class:`mon.nn.data.DataModule`,
                - :class:`mon.nn.data.Dataset`,
                - :class:`torch.utils.data.Dataloader`,
                - A path to the data-source.
        """
        pass
    
    @abstractmethod
    def preprocess(self, input: torch.Tensor):
        """Pre-process an input.

        Args:
            input: An input of shape :math:`[B, C, H, W]`.

        Returns:
            The processed input of shape :math:`[B, C H, W]`.
        """
        pass
    
    @abstractmethod
    def postprocess(
        self,
        input: torch.Tensor,
        pred : torch.Tensor,
        *args, **kwargs
    ) -> torch.Tensor:
        """Post-process a prediction.

        Args:
            input: The input of shape :math:`[B, C, H, W]`.
            pred: The prediction of shape :math:`[B, C, H, W]`.

        Returns:
            The post-processed prediction of shape :math:`[B, C, H, W]`.
        """
        pass
    
    def on_run_start(self):
        """Call before :meth:`run` starts."""
        if self.save:
            core.mkdirs(paths=[self.output_dir], replace=True)
        
        self.init_data_loader()
        self.init_data_writer()
        self.init_logger()
        
        self.model.phase = self.phase
        self.model.to(self.device)
    
    @abstractmethod
    def on_run_end(self):
        """Call after :meth`run` finishes."""
        pass


'''
class VisionInferrer(Inferrer):
    """
    Online vision inference pipeline.
    """
    
    def __init__(
        self,
        source     : Path_ | None = None,
        root       : Path_ | None = RUNS_DIR / "infer",
        project    : str          = "",
        name       : str          = "exp",
        max_samples: int   | None = None,
        batch_size : int          = 1,
        shape      : Ints  | None = None,
        device     : int   | str  = "cpu",
        phase      : ModelPhase_  = "training",
        tensorrt   : bool         = True,
        save       : bool         = True,
        verbose    : bool         = True,
        *args, **kwargs
    ):
        super().__init__(
            source=source,
            root=root,
            project=project,
            name=name,
            max_samples=max_samples,
            batch_size=batch_size,
            shape=shape,
            device=device,
            phase=phase,
            tensorrt=tensorrt,
            save=save,
            verbose=verbose,
            *args, **kwargs
        )
    
    def init_data_loader(self):
        """
        Initialize data loader.
        """
        import one.vision.acquisition as io
        if isinstance(self.source, (DataLoader, DataModule)):
            pass
        elif is_image_file(self.source) or is_dir(self.source):
            self.data_loader = io.ImageLoader(
                source=self.source,
                max_samples=self.max_samples,
                batch_size=self.batch_size,
            )
        elif is_video_file(self.source):
            self.data_loader = io.VideoLoaderCV(
                source=self.source,
                max_samples=self.max_samples,
                batch_size=self.batch_size,
            )
        else:
            raise RuntimeError()
    
    def init_data_writer(self):
        """
        Initialize data writer.
        """
        import one.vision.acquisition as io
        if self.save:
            if is_image_file(self.source) or is_dir(self.source):
                self.data_writer = io.ImageWriter(dst=self.output_dir)
            elif is_video_file(self.source) \
                and isinstance(self.data_loader, io.VideoLoaderCV):
                self.data_writer = io.VideoWriterCV(
                    dst=self.output_dir,
                    shape=self.data_loader.shape,
                    frame_rate=30,
                )
            else:
                raise RuntimeError()
    
    def run(self, model: Model, source: Path_):
        self.model = model
        self.source = source
        self.on_run_start()
        
        # Setup online learning
        if self.phase == ModelPhase.TRAINING:
            optimizer = torch.optim.SGD(
                params=model.parameters(),
                lr=0.01,
                weight_decay=0.01,
            )
        else:
            optimizer = None
        
        # Print info
        self.logger.write(f"{'Model':<22}: {model.name}\n")
        self.logger.write(f"{'Data':<22}: {model.fullname}\n")
        if hasattr(model, "params"):
            self.logger.write(f"{'Parameters':<22}: {model.params}\n")
        # if self.shape is not None and is_sequence_of_length(self.shape, 3):
        #     self.logger.write(f"{'MACs':<21}: {model.macs(self.shape)}\n")
        self.logger.write(f"{'Device':<22}: {self.model.device}\n")
        self.logger.write(f"{'TensorRT':<22}: {self.tensorrt}\n")
        self.logger.write(f"{'Image Size':<22}: {self.shape}\n")
        self.logger.flush()
        
        step_times = []
        used_memory = []
        start_time = timer()
        with progress_bar() as pbar:
            for batch_idx, batch in pbar.track(
                enumerate(self.data_loader),
                total=int(len(self.data_loader) / self.batch_size),
                description=f"[bright_yellow] Processing"
            ):
                # Frame capture
                images, indexes, files, rel_paths = batch
                
                # Pre-process
                input, size0, size1 = self.preprocess(images)
                
                # Process
                step_start_time = timer()
                if model.phase == ModelPhase.TRAINING:
                    pred, loss = self.model.forward_loss(
                        input=input, target=None
                        )
                else:
                    pred, loss = self.model.forward(input=input), None
                
                if torch.cuda.is_available():
                    total, used, free = get_gpu_memory()
                    used_memory.append(used)
                
                step_end_time = timer()
                step_times.append(step_end_time - step_start_time)
                
                # Online learning
                if optimizer is not None and loss is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()
                
                # Post-process
                pred = self.postprocess(
                    input=input,
                    pred=pred,
                    size0=size0,
                    size1=size1,
                )
                
                # Debug
                if self.verbose:
                    self.model.show_results(
                        input=images,
                        pred=pred,
                        max_n=self.batch_size,
                        nrow=self.batch_size,
                        save=False,
                    )
                if self.save:
                    self.data_writer.write_batch(
                        images=pred,
                        files=rel_paths,
                        denormalize=True,
                    )
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        
        end_time = timer()
        console.log(
            f"{'Used Memory':<22}: "
            f"{(sum(used_memory) / len(used_memory)):.9f} GB"
        )
        console.log(
            f"{'Total Time':<22}: {(end_time - start_time):.9f} seconds"
        )
        console.log(
            f"{'Average Time':<22}: "
            f"{((end_time - start_time) / len(step_times)):.9f} seconds"
        )
        console.log(
            f"{'Average Time (forward)':<22}: "
            f"{(sum(step_times) / len(step_times)):.9f} seconds"
        )
        
        self.logger.write(
            f"{'Used Memory':<22}: "
            f"{(sum(used_memory) / len(used_memory)):.9f} GB\n"
        )
        self.logger.write(
            f"{'Total Time':<22}: {(end_time - start_time):.9f} seconds\n"
        )
        self.logger.write(
            f"{'Average Time':<22}: "
            f"{((end_time - start_time) / len(step_times)):.9f} seconds\n"
        )
        self.logger.write(
            f"{'Average Time (forward)':<22}: "
            f"{(sum(step_times) / len(step_times)):.9f} seconds\n"
        )
        self.logger.flush()
        self.logger.close()
        self.on_run_end()
    
    def preprocess(self, input: torch.Tensor) -> tuple[torch.Tensor, Ints,
    Ints]:
        """
        Preprocessing input.

        Args:
            input (torch.Tensor): Input of shape [B, C, H, W].

        Returns:
            input (torch.Tensor): Processed input image as  [B, C H, W].
        	size0 (Ints_): The original images' sizes.
            size1 (Ints_): The resized images' sizes.
        """
        from one.vision.acquisition import get_image_size
        from one.vision.transformation import resize
        
        size0 = get_image_size(input)
        if self.shape:
            new_size = to_size(self.shape)
            if size0 != new_size:
                input = resize(
                    image=input,
                    size=new_size,
                    interpolation=InterpolationMode.BICUBIC
                )
            # images = [resize(i, self.shape) for i in images]
            # images = torch.stack(input)
        size1 = get_image_size(input)
        
        input = input.to(self.device)
        return input, size0, size1
    
    # noinspection PyMethodOverriding
    def postprocess(
        self,
        input: torch.Tensor,
        pred : torch.Tensor,
        size0: Ints,
        size1: Ints,
    ) -> torch.Tensor:
        """
        Postprocessing prediction.

        Args:
            input (torch.Tensor): Input of shape [B, C, H, W].
            pred (torch.Tensor): Prediction of shape [B, C, H, W].
            size0 (Ints_): The original images' sizes.
            size1 (Ints_): The resized images' sizes.

        Returns:
            pred (torch.Tensor): Results of shape [B, C, H, W].
        """
        from one.vision.transformation import resize
        
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
        
        if size0 != size1:
            pred = pred if isinstance(pred, torch.Tensor) else
            torch.from_numpy(pred)
            pred = resize(
                image=pred,
                size=size0,
                interpolation=InterpolationMode.BICUBIC
            )
        return pred
    
    def on_run_end(self):
        """
        Call after `run()` finishes.
        """
        if self.verbose:
            cv2.destroyAllWindows()
        if self.save and self.data_writer:
            self.data_writer.close()
'''
