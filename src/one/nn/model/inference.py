#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Inference Pipeline.
"""

from __future__ import annotations

import os
import threading
from queue import Queue
from typing import Any
from typing import Optional
from typing import Union

import cv2
import numpy as np
import torch
from one.imgproc import resize
from one.io import create_dirs
from one.io import ImageLoader
from one.io import ImageWriter
from torch import Tensor
from torchvision.transforms import functional as F

from one.core import Arrays
from one.core import console
from one.core import get_image_hw
from one.core import Int2T
from one.core import Int3T
from one.core import InterpolationMode
from one.core import progress_bar
from one.core import select_device
from one.core import to_image
from one.nn.model.utils import get_next_version

__all__ = [
    "Inference"
]


# MARK: - Inference

class Inference:
    """Inference class defines the prediction loop for image data: images,
    folders, video, etc.
    
    Attributes:
        default_root_dir (str):
            Root dir to save predicted data.
        output_dir (str):
            Output directory to save predicted images.
        model (nn.Module):
            Model to run.
        data (str):
            Data source. Can be a path or pattern to image/video/directory.
        data_loader (Any):
            Data loader object.
        shape (tuple, optional):
            Input and output shape of the image as [H, W, C]. If `None`,
            use the input image shape.
        batch_size (int):
            Batch size. Default: `1`.
        device (int, str, optional):
            Will be mapped to either gpus, tpu_cores, num_processes or ipus,
            based on the accelerator type.
        verbose (bool):
            Verbosity mode. Default: `False`.
        save_image (bool):
            Save predicted images. Default: `False`.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        default_root_dir: str,
        version         : Union[int, str, None] = None,
        shape           : Optional[Int3T]        = None,
        batch_size      : int                   = 1,
        device          : Union[int, str, None] = 0,
        verbose         : bool                  = True,
        save_image      : bool                  = False,
        *args, **kwargs
    ):
        super().__init__()
        self.default_root_dir = default_root_dir
        self.shape            = shape
        self.batch_size       = batch_size
        self.device           = select_device(device=device)
        self.verbose          = verbose
        self.save_image       = save_image
        self.model            = None
        self.data             = None
        self.data_loader      = None
        self.image_writer     = None
        
        self.init_output_dir(version=version)
        
    # MARK: Configure
    
    def init_output_dir(self, version: Union[int, str, None] = None):
        """Configure output directory base on the given version.
        
        Args:
            version (int, str, optional):
                Experiment version. If version is not specified the logger
                inspects the save directory for existing versions, then
                automatically assigns the next available version. If it is a
                string then it is used as the run-specific subdirectory name,
                otherwise `version_${version}` is used.
        """
        if version is None:
            version = get_next_version(root_dir=self.default_root_dir)
        if isinstance(version, int):
            version = f"version_{version}"
        version = version.lower()
        
        self.output_dir = os.path.join(self.default_root_dir, version)
        console.log(f"Output directory at: {self.output_dir}.")
    
    def init_data_loader(self):
        """Configure the data loader object."""
        self.data_loader = ImageLoader(
            data=self.data, batch_size=self.batch_size
        )
        
    def init_data_writer(self):
        """Configure the data writer object."""
        self.image_writer = ImageWriter(dst=self.output_dir)
        
    def validate_attributes(self):
        """Validate all attributes' values before run loop start."""
        if self.model is None:
            raise ValueError(f"`model` must be defined.")
        if self.data_loader is None:
            raise ValueError(f"`data_loader` must be defined.")
        if self.save_image and self.image_writer is None:
            raise ValueError(f"`image_writer` must be defined.")
        
    # MARK: Run
    
    def run(self, model: Any, data: str):
        """Main prediction loop.
        
        Args:
            model (nn.Module):
                Model to run.
            data (str):
                Data source. Can be a path or pattern to image/video/directory.
        """
        self.model = model
        self.data  = data
        
        self.run_routine_start()
        
        # NOTE: Mains loop
        with progress_bar() as pbar:
            for batch_idx, batch in pbar.track(
                enumerate(self.data_loader),
                total=len(self.data_loader),
                description=f"[bright_yellow]{self.model.fullname}"
            ):
                images, indexes, files, rel_paths = batch
                input, size0, size1 = self.preprocess(images)
                pred                = self.model.forward(input)
                results             = self.postprocess(pred, size0, size1)
                
                if self.verbose:
                    self.show_results(results=results)
                if self.save_image:
                    self.image_writer.write_batch(
                        images=results, image_files=rel_paths
                    )
                    
        self.run_routine_end()
        
    def run_routine_start(self):
        """When run routine starts we build the `output_dir` on the fly."""
        create_dirs(paths=[self.output_dir])
        self.init_data_loader()
        self.init_data_writer()
        self.validate_attributes()

        self.model.to(self.device)
        self.model.eval()
        
        if self.verbose:
            cv2.namedWindow(
                "results", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO
            )

    def preprocess(self, images: Arrays) -> tuple[Tensor, Int2T, Int2T]:
        """Preprocessing input.

        Args:
            images (Arrays):
                Input images as [B, H, W, C].

        Returns:
        	x (Tensor):
        	    Input image as  [B, C H, W].
        	size0 (Int2T):
                The original images' sizes.
            size1 (Int2T):
                The resized images' sizes.
        """
        # NOTE: THIS PIPELINE IS FASTER
        size0  = get_image_hw(images)
        if self.shape:
            images = [resize(i, self.shape) for i in images]
        images = [F.to_tensor(i) for i in images]
        images = torch.stack(images)
        size1  = get_image_hw(images)
        images = images.to(self.device)
        
        """
        # NOTE: THIS PIPELINE IS SLOWER
        size0  = get_image_size(images)
        images = [torchvision.transforms.ToTensor()(i) for i in images]
        images = torch.stack(images)
        images = images.to(self.device)
        if self.shape:
            images = resize(images, self.shape)
            # images = [resize(i, self.shape) for i in images]
        size1  = get_image_size(images)
        """
        return images, size0, size1

    def postprocess(
        self, results: Tensor, size0: Int2T, size1: Int2T
    ) -> np.ndarray:
        """Postprocessing results.

        Args:
            results (Tensor):
                Output images.
            size0 (Int2T):
                The original images' sizes.
            size1 (Int2T):
                The resized images' sizes.
                
        Returns:
            results (np.ndarray):
                Post-processed output images as [B, H, W, C].
        """
        # For multi-stages models, we only get the last pred
        if isinstance(results, (list, tuple)):
            results = results[-1]
        
        results = to_image(results, denormalize=True)  # List of 4D-array
        
        if size0 != size1:
            results = resize(
                results, size0, interpolation=InterpolationMode.CUBIC
            )

        return results

    def run_routine_end(self):
        """When run routine ends we release data loader and writers."""
        self.model.train()
        
        if self.verbose:
            cv2.destroyAllWindows()

    # MARK: Visualize

    def show_results(self, results: np.ndarray):
        """Show results.
        
        Args:
            results (np.ndarray):
                Post-processed output images as [B, H, W, C].
        """
        for i in results:
            cv2.imshow("results", i)
            cv2.waitKey(1)


# MARK: - MultiThreadInference

class MultiThreadInference(Inference):
    """Multi-Thread Inference class defines the prediction loop for image data:
    images, folders, video, etc.
    
    Attributes:
        default_root_dir (str):
            Root dir to save predicted data.
        output_dir (str):
            Output directory to save predicted images.
        model (nn.Module):
            Model to run.
        data (str):
            Data source. Can be a path or pattern to image/video/directory.
        data_loader (Any):
            Data loader object.
        shape (tuple, optional):
            Input and output shape of the image as [H, W, C]. If `None`,
            use the input image shape.
        batch_size (int):
            Batch size. Default: `1`.
        device (int, str, optional):
            Will be mapped to either gpus, tpu_cores, num_processes or ipus,
            based on the accelerator type.
        queue_size (int):
        
        verbose (bool):
            Verbosity mode. Default: `False`.
        save_image (bool):
            Save predicted images. Default: `False`.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        default_root_dir: str,
        version         : Union[int, str, None] = None,
        shape           : Optional[Int3T]       = None,
        batch_size      : int                   = 1,
        device          : Union[int, str, None] = 0,
        queue_size      : int                   = 10,
        verbose         : bool                  = True,
        save_image      : bool                  = False,
        *args, **kwargs
    ):
        super().__init__(
            default_root_dir = default_root_dir,
            version          = version,
            shape            = shape,
            batch_size       = batch_size,
            device           = device,
            queue_size       = queue_size,
            verbose          = verbose,
            save_image       = save_image,
            *args, **kwargs
        )
        self.pbar          = None
        self.task          = None
        self.queue_size    = queue_size

        # NOTE: Queue
        self.frames_queue  = Queue(maxsize=self.queue_size)
        self.input_queue   = Queue(maxsize=self.queue_size)
        self.pred_queue    = Queue(maxsize=self.queue_size)
        self.results_queue = Queue(maxsize=self.queue_size)
        
    # MARK: Run
    
    def run(self, model: Any, data: str):
        """Main prediction loop.
        
        Args:
            model (nn.Module):
                Model to run.
            data (str):
                Data source. Can be a path or pattern to image/video/directory.
        """
        self.model = model
        self.data  = data
        
        self.run_routine_start()
        
        # NOTE: Thread for data reader
        thread_data_reader = threading.Thread(target=self.run_data_reader)
        thread_data_reader.start()

        # NOTE: Thread for pre-process
        thread_preprocess = threading.Thread(target=self.run_preprocess)
        thread_preprocess.start()

        # NOTE: Thread for model
        thread_model = threading.Thread(target=self.run_model)
        thread_model.start()

        # NOTE: Thread for post-process
        thread_postprocess = threading.Thread(target=self.run_postprocess)
        thread_postprocess.start()

        # NOTE: Thread for result writer
        thread_result_writer = threading.Thread(target=self.run_result_writer)
        thread_result_writer.start()
        
        # NOTE: Joins threads when all terminate
        thread_data_reader.join()
        thread_preprocess.join()
        thread_model.join()
        thread_postprocess.join()
        thread_result_writer.join()
                    
        self.run_routine_end()
        
    def run_routine_start(self):
        """When run routine starts we build the `output_dir` on the fly."""
        create_dirs(paths=[self.output_dir])
        self.init_data_loader()
        self.init_data_writer()
        self.validate_attributes()

        self.model.to(self.device)
        self.model.eval()
        
        self.pbar = progress_bar()
        with self.pbar:
            self.task = self.pbar.add_task("Inferring", total=len(self.data_loader))
            
        if self.verbose:
            cv2.namedWindow(
                "results", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO
            )
    
    def run_data_reader(self):
        """Run data reader thread and push images and frame_indexes to queue.
        """
        for images, indexes, files, rel_paths in self.data_loader:
            if len(indexes) > 0:
                # NOTE: Push frame index and images to queue
                self.frames_queue.put([images, indexes, files, rel_paths])

        # NOTE: Push None to queue to act as a stopping condition for next
        # thread
        self.frames_queue.put([None, None, None, None])
    
    def run_preprocess(self):
        """Run pre-processing thread and push input to queue."""
        while True:
            # NOTE: Get frame indexes and images from queue
            (images, indexes, files, rel_paths) = self.frames_queue.get()
            if indexes is None:
                break
                
            # NOTE: Pre-processing images
            images, size0, size1 = self.preprocess(images)
            
            # NOTE: Push input to queue
            self.input_queue.put([images, indexes, files, rel_paths, size0, size1])

        # NOTE: Push None to queue to act as a stopping condition for next
        # thread
        self.input_queue.put([None, None, None, None, None, None])
        
    def run_model(self):
        """Run model thread and push pred to queue."""
        while True:
            # NOTE: Get input from queue
            (input, indexes, files, rel_paths, size0, size1) = self.input_queue.get()
            if indexes is None:
                break
    
            # NOTE: Detect batch of inputs
            preds = self.model.forward(input)

            # NOTE: Push predictions to queue
            self.pred_queue.put([preds, indexes, files, rel_paths, size0, size1])

        # NOTE: Push None to queue to act as a stopping condition for next
        # thread
        self.pred_queue.put([None, None, None, None, None, None])
    
    def run_postprocess(self):
        """Run post-processing thread and push results to queue."""
        while True:
            # NOTE: Get predictions from queue
            (preds, indexes, files, rel_paths, size0, size1) = self.pred_queue.get()
            if indexes is None:
                break
                
            # NOTE: Post-processing images
            results = self.postprocess(preds, size0, size1)
            
            # NOTE: Push results to queue
            self.results_queue.put([results, indexes, files, rel_paths])
            
            with self.pbar:
                self.pbar.update(self.task, advance=1)
            
        # NOTE: Push None to queue to act as a stopping condition for next
        # thread
        self.results_queue.put([None, None, None, None])
    
    def run_result_writer(self):
        """Run result writing thread."""
        while True:
            # NOTE: Get predictions from queue
            (results, indexes, files, rel_paths) = self.results_queue.get()
            if indexes is None:
                break

            if self.verbose:
                self.show_results(results=results)
            if self.save_image:
                self.image_writer.write_batch(
                    images=results, image_files=rel_paths
                )
    
    def run_routine_end(self):
        """When run routine ends we release data loader and writers."""
        self.model.train()

        if self.verbose:
            cv2.destroyAllWindows()

    # MARK: Visualize

    def show_results(self, results: np.ndarray):
        """Show results.
        
        Args:
            results (np.ndarray):
                Post-processed output images as [B, H, W, C].
        """
        for i in results:
            cv2.imshow("results", i)
            cv2.waitKey(1)
