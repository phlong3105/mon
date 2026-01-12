#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""WiderFace dataset.

This module provides the WiderFace dataset for face detection.
"""

from __future__ import annotations

__all__ = [
    "WiderFace",
    "WiderFaceVal",
]


from ...api import *


@DATASETS.register()
class WiderFace(ImageDataset, RegistrableMixin):
    """WiderFace dataset."""
    
    _name      : str         = "widerface"
    _tasks     : list[Task]  = [Task.DETECT]
    _subset    : str         = None
    _splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(
            name    = "image",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
            primary = True,
        ),
    }
    _classlist : ClassList   = ClassList([
        {"name": "face", "id": 0, "color": [81, 120, 228]},
    ])


@DATASETS.register()
class WiderFaceVal(WiderFace, RegistrableMixin):
    """WiderFace-Val subset."""
    
    _name: str = "widerfaceval"
    
    # --- Data Loading ---
    def _load_primary_data(self) -> list[Image]:
        """Load primary modality data files in the dataset.
        
        Returns:
            A list of primary modality data files.
        """
        patterns = [self._root / "val" / "image"]
        
        images   = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(data=path, root=pattern))

        return images
