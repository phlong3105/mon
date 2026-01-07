#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NightCity dataset.

This module implements the NightCity dataset for nighttime scene parsing.

References:
    - Paper: "Night-time Scene Parsing with a Large Real Dataset".
	- Data: https://dmcv.sjtu.edu.cn/people/phd/tanxin/NightCity/index.html
"""

__all__ = [
	"NightCity",
]

from mon.core import rich
from ....api import *


@DATASETS.register(name="nightcity")
class NightCity(ImageDataset):
    """NightCity dataset."""
    
    _subset    : str         = "nightcity"
    _tasks     : list[Task]  = [Task.NTE, Task.LLE, Task.SEGMENT]
    _splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",    type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName,  type="image", module=DefaultDepthMap, train=True, test=True),
        "mask" : Modality(name="labelIds", type="image", module=SemanticMask,    train=True, test=False),
    }
    _classlist : ClassList   = ClassList([
        {"name": "unlabeled"           , "id": 0 , "train_id": 255, "category": "void"        , "category_id": 0, "ignore_in_eval": True , "color": [0  , 0  ,   0]},
        {"name": "ego vehicle"         , "id": 1 , "train_id": 255, "category": "void"        , "category_id": 0, "ignore_in_eval": True , "color": [0  , 0  ,   0]},
        {"name": "rectification border", "id": 2 , "train_id": 255, "category": "void"        , "category_id": 0, "ignore_in_eval": True , "color": [0  , 0  ,   0]},
        {"name": "out of roi"          , "id": 3 , "train_id": 255, "category": "void"        , "category_id": 0, "ignore_in_eval": True , "color": [0  , 0  ,   0]},
        {"name": "static"              , "id": 4 , "train_id": 255, "category": "void"        , "category_id": 0, "ignore_in_eval": True , "color": [0  , 0  ,   0]},
        {"name": "dynamic"             , "id": 5 , "train_id": 255, "category": "void"        , "category_id": 0, "ignore_in_eval": True , "color": [111, 74 ,   0]},
        {"name": "ground"              , "id": 6 , "train_id": 255, "category": "void"        , "category_id": 0, "ignore_in_eval": True , "color": [81 , 0  ,  81]},
        {"name": "road"                , "id": 7 , "train_id": 0  , "category": "flat"        , "category_id": 1, "ignore_in_eval": False, "color": [128, 64 , 128]},
        {"name": "sidewalk"            , "id": 8 , "train_id": 1  , "category": "flat"        , "category_id": 1, "ignore_in_eval": False, "color": [244, 35 , 232]},
        {"name": "parking"             , "id": 9 , "train_id": 255, "category": "flat"        , "category_id": 1, "ignore_in_eval": True , "color": [250, 170, 160]},
        {"name": "rail track"          , "id": 10, "train_id": 255, "category": "flat"        , "category_id": 1, "ignore_in_eval": True , "color": [230, 150, 140]},
        {"name": "building"            , "id": 11, "train_id": 2  , "category": "construction", "category_id": 2, "ignore_in_eval": False, "color": [70 , 70 ,  70]},
        {"name": "wall"                , "id": 12, "train_id": 3  , "category": "construction", "category_id": 2, "ignore_in_eval": False, "color": [102, 102, 156]},
        {"name": "fence"               , "id": 13, "train_id": 4  , "category": "construction", "category_id": 2, "ignore_in_eval": False, "color": [190, 153, 153]},
        {"name": "guard rail"          , "id": 14, "train_id": 255, "category": "construction", "category_id": 2, "ignore_in_eval": True , "color": [180, 165, 180]},
        {"name": "bridge"              , "id": 15, "train_id": 255, "category": "construction", "category_id": 2, "ignore_in_eval": True , "color": [150, 100, 100]},
        {"name": "tunnel"              , "id": 16, "train_id": 255, "category": "construction", "category_id": 2, "ignore_in_eval": True , "color": [150, 120,  90]},
        {"name": "pole"                , "id": 17, "train_id": 5  , "category": "object"      , "category_id": 3, "ignore_in_eval": False, "color": [153, 153, 153]},
        {"name": "polegroup"           , "id": 18, "train_id": 255, "category": "object"      , "category_id": 3, "ignore_in_eval": True , "color": [153, 153, 153]},
        {"name": "traffic light"       , "id": 19, "train_id": 6  , "category": "object"      , "category_id": 3, "ignore_in_eval": False, "color": [250, 170,  30]},
        {"name": "traffic sign"        , "id": 20, "train_id": 7  , "category": "object"      , "category_id": 3, "ignore_in_eval": False, "color": [220, 220,   0]},
        {"name": "vegetation"          , "id": 21, "train_id": 8  , "category": "nature"      , "category_id": 4, "ignore_in_eval": False, "color": [107, 142,  35]},
        {"name": "terrain"             , "id": 22, "train_id": 9  , "category": "nature"      , "category_id": 4, "ignore_in_eval": False, "color": [152, 251, 152]},
        {"name": "sky"                 , "id": 23, "train_id": 10 , "category": "sky"         , "category_id": 5, "ignore_in_eval": False, "color": [70 , 130, 180]},
        {"name": "person"              , "id": 24, "train_id": 11 , "category": "human"       , "category_id": 6, "ignore_in_eval": False, "color": [220, 20 ,  60]},
        {"name": "rider"               , "id": 25, "train_id": 12 , "category": "human"       , "category_id": 6, "ignore_in_eval": False, "color": [255, 0  ,   0]},
        {"name": "car"                 , "id": 26, "train_id": 13 , "category": "vehicle"     , "category_id": 7, "ignore_in_eval": False, "color": [0  , 0  , 142]},
        {"name": "truck"               , "id": 27, "train_id": 14 , "category": "vehicle"     , "category_id": 7, "ignore_in_eval": False, "color": [0  , 0  ,  70]},
        {"name": "bus"                 , "id": 28, "train_id": 15 , "category": "vehicle"     , "category_id": 7, "ignore_in_eval": False, "color": [0  , 60 , 100]},
        {"name": "caravan"             , "id": 29, "train_id": 255, "category": "vehicle"     , "category_id": 7, "ignore_in_eval": True , "color": [0  , 0  ,  90]},
        {"name": "trailer"             , "id": 30, "train_id": 255, "category": "vehicle"     , "category_id": 7, "ignore_in_eval": True , "color": [0  , 0  , 110]},
        {"name": "train"               , "id": 31, "train_id": 16 , "category": "vehicle"     , "category_id": 7, "ignore_in_eval": False, "color": [0  , 80 , 100]},
        {"name": "motorcycle"          , "id": 32, "train_id": 17 , "category": "vehicle"     , "category_id": 7, "ignore_in_eval": False, "color": [0  , 0  , 230]},
        {"name": "bicycle"             , "id": 33, "train_id": 18 , "category": "vehicle"     , "category_id": 7, "ignore_in_eval": False, "color": [119, 11 ,  32]},
        {"name": "license plate"       , "id": -1, "train_id": -1 , "category": "vehicle"     , "category_id": 7, "ignore_in_eval": True , "color": [0  , 0  , 142]},
    ])
    
    # --- Data Loading ---
    def _load_primary_data(self) -> list[Image]:
        """Load primary modality data files in the dataset.
        
        Returns:
            A list of Image instances for the primary modality.
        """
        if self.split == Split.TEST:
            patterns = [self.root / "val" / "image"]
        else:
            patterns = [self.root / self.split_str / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(data=path, root=pattern))

        return images
