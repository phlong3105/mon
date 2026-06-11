"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import random

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from PIL import Image

from ...core import register
from .._misc import convert_to_tv_tensor


@register()
class Mosaic(T.Transform):
    """
    Applies Mosaic augmentation to a batch of images. Combines four randomly selected images
    into a single composite image with randomized transformations.
    """

    def __init__(self, output_size=320, max_size=None, rotation_range=0, translation_range=(0.1, 0.1),
                 scaling_range=(0.5, 1.5), fill_value=114, max_cached_images=50,
                 random_pop=True) -> None:
        """
        Args:
            output_size (int): Target size for resizing individual images.
            rotation_range (float): Range of rotation in degrees for affine transformation.
            translation_range (tuple): Range of translation for affine transformation.
            scaling_range (tuple): Range of scaling factors for affine transformation.
            probability (float): Probability of applying the Mosaic augmentation.
            fill_value (int): Fill value for padding or affine transformations.
            max_cached_images (int): The maximum length of the cache.
            random_pop (bool): Whether to randomly pop a result from the cache.
        """
        super().__init__()
        self.resize = T.Resize(size=output_size, max_size=max_size)
        self.affine_transform = T.RandomAffine(degrees=rotation_range, translate=translation_range,
                                               scale=scaling_range, fill=fill_value)
        self.mosaic_cache = []
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop

    def load_samples_from_cache(self, image, target, cache):
        image, target = self.resize(image, target)
        cache.append(dict(img=image, labels=target))

        if len(cache) > self.max_cached_images:
            if self.random_pop:
                index = random.randint(0, len(cache) - 2)  # do not remove last image
            else:
                index = 0
            cache.pop(index)
        sample_indices = random.choices(range(len(cache)), k=3)
        mosaic_samples = [dict(img=cache[idx]["img"].copy(), labels=self._clone(cache[idx]["labels"])) for idx in
                          sample_indices]  # sample 3 images
        mosaic_samples = [dict(img=image.copy(), labels=self._clone(target))] + mosaic_samples

        get_size_func = F.get_size if hasattr(F, "get_size") else F.get_spatial_size
        sizes = [get_size_func(mosaic_samples[idx]["img"]) for idx in range(4)]
        max_height = max(size[0] for size in sizes)
        max_width = max(size[1] for size in sizes)

        return mosaic_samples, max_height, max_width

    def create_mosaic_from_cache(self, mosaic_samples, max_height, max_width):
        # Create canvas for image and masks
        merged_image = Image.new(mode=mosaic_samples[0]["img"].mode, size=(max_width * 2, max_height * 2), color=0)
        
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
        offsets = torch.tensor([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]).repeat(1, 2)

        mosaic_target = []
        all_masks = []
        has_masks = 'masks' in mosaic_samples[0]["labels"]

        for i, sample in enumerate(mosaic_samples):
            img = sample["img"]
            target = sample["labels"]

            merged_image.paste(img, placement_offsets[i])
            target['boxes'] = target['boxes'] + offsets[i]
            
            if has_masks:
                curr_m = target['masks'] 
                # Create a full-size zero mask for this quadrant's objects
                full_size_m = torch.zeros((curr_m.shape[0], max_height * 2, max_width * 2), 
                                         dtype=curr_m.dtype, device=curr_m.device)
                y_off, x_off = placement_offsets[i][1], placement_offsets[i][0]
                h_c, w_c = curr_m.shape[1], curr_m.shape[2]
                full_size_m[:, y_off : y_off + h_c, x_off : x_off + w_c] = curr_m
                all_masks.append(full_size_m)
                # Remove small masks to avoid cat mismatch later
                del target['masks']

            mosaic_target.append(target)

        # Merge targets (boxes, labels, etc.)
        merged_target = {}
        for key in mosaic_target[0]:
            merged_target[key] = torch.cat([target[key] for target in mosaic_target])

        # Concatenate full-size masks along the object dimension
        if has_masks:
            merged_target['masks'] = torch.cat(all_masks, dim=0)

        return merged_image, merged_target

    @staticmethod
    def _clone(tensor_dict):
        return {key: value.clone() for (key, value) in tensor_dict.items()}

    def forward(self, *inputs):
        """
        Args:
            inputs (tuple): Input tuple containing (image, target, dataset).

        Returns:
            tuple: Augmented (image, target, dataset).
        """
        if len(inputs) == 1:
            inputs = inputs[0]
        image, target = inputs

        # Skip mosaic augmentation with probability 1 - self.probability
        # if self.probability < 1.0 and random.random() > self.probability:
        #     return image, target, dataset

        # Prepare mosaic components
        
        mosaic_samples, max_height, max_width = self.load_samples_from_cache(image, target, self.mosaic_cache)
        mosaic_image, mosaic_target = self.create_mosaic_from_cache(mosaic_samples, max_height, max_width)
            
        # Clamp boxes and convert target formats
        if 'boxes' in mosaic_target:
            mosaic_target['boxes'] = convert_to_tv_tensor(mosaic_target['boxes'], 'boxes', box_format='xyxy',
                                                          spatial_size=mosaic_image.size[::-1])
        if 'masks' in mosaic_target:
            mosaic_target['masks'] = convert_to_tv_tensor(mosaic_target['masks'], 'masks')

        # Apply affine transformations
        mosaic_image, mosaic_target = self.affine_transform(mosaic_image, mosaic_target)

        return mosaic_image, mosaic_target
