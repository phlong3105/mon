#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NightCity Datasets.

This module provides NightCity datasets.
"""

from __future__ import annotations

__all__ = [
    "NightCity",
]

from mon.core import Class, ClassList, DATASETS, Split, Task
from mon.dataset.base import (
    DatasetRegisterMixin,
    DepthModality,
    ImageDataset,
    ImageModality,
    ModalityList,
)


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register(name="nightcity")
class NightCity(ImageDataset, DatasetRegisterMixin):
    """NightCity dataset."""

    name: str = "nightcity"
    tasks: list[Task] = [Task.LLE, Task.SEGMENT]
    dirname: str = "nightcity"
    subdir: str = "nightcity"
    splits: list[Split] = [Split.TRAIN, Split.VAL]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="images"),
        DepthModality(name="depth", dirname="depth"),
    ])
    classes: ClassList = ClassList([
        Class(name="unlabeled"           , id=0 , train_id=255, category="void"        , category_id=0, ignore_in_eval=True , color=(0  , 0  , 0)),
        Class(name="ego vehicle"         , id=1 , train_id=255, category="void"        , category_id=0, ignore_in_eval=True , color=(0  , 0  , 0)),
        Class(name="rectification border", id=2 , train_id=255, category="void"        , category_id=0, ignore_in_eval=True , color=(0  , 0  , 0)),
        Class(name="out of roi"          , id=3 , train_id=255, category="void"        , category_id=0, ignore_in_eval=True , color=(0  , 0  , 0)),
        Class(name="static"              , id=4 , train_id=255, category="void"        , category_id=0, ignore_in_eval=True , color=(0  , 0  , 0)),
        Class(name="dynamic"             , id=5 , train_id=255, category="void"        , category_id=0, ignore_in_eval=True , color=(111, 74 , 0)),
        Class(name="ground"              , id=6 , train_id=255, category="void"        , category_id=0, ignore_in_eval=True , color=(81 , 0  , 81)),
        Class(name="road"                , id=7 , train_id=0  , category="flat"        , category_id=1, ignore_in_eval=False, color=(128, 64 , 128)),
        Class(name="sidewalk"            , id=8 , train_id=1  , category="flat"        , category_id=1, ignore_in_eval=False, color=(244, 35 , 232)),
        Class(name="parking"             , id=9 , train_id=255, category="flat"        , category_id=1, ignore_in_eval=True , color=(250, 170, 160)),
        Class(name="rail track"          , id=10, train_id=255, category="flat"        , category_id=1, ignore_in_eval=True , color=(230, 150, 140)),
        Class(name="building"            , id=11, train_id=2  , category="construction", category_id=2, ignore_in_eval=False, color=(70 , 70 , 70)),
        Class(name="wall"                , id=12, train_id=3  , category="construction", category_id=2, ignore_in_eval=False, color=(102, 102, 156)),
        Class(name="fence"               , id=13, train_id=4  , category="construction", category_id=2, ignore_in_eval=False, color=(190, 153, 153)),
        Class(name="guard rail"          , id=14, train_id=255, category="construction", category_id=2, ignore_in_eval=True , color=(180, 165, 180)),
        Class(name="bridge"              , id=15, train_id=255, category="construction", category_id=2, ignore_in_eval=True , color=(150, 100, 100)),
        Class(name="tunnel"              , id=16, train_id=255, category="construction", category_id=2, ignore_in_eval=True , color=(150, 120, 90)),
        Class(name="pole"                , id=17, train_id=5  , category="object"      , category_id=3, ignore_in_eval=False, color=(153, 153, 153)),
        Class(name="polegroup"           , id=18, train_id=255, category="object"      , category_id=3, ignore_in_eval=True , color=(153, 153, 153)),
        Class(name="traffic light"       , id=19, train_id=6  , category="object"      , category_id=3, ignore_in_eval=False, color=(250, 170, 30)),
        Class(name="traffic sign"        , id=20, train_id=7  , category="object"      , category_id=3, ignore_in_eval=False, color=(220, 220, 0)),
        Class(name="vegetation"          , id=21, train_id=8  , category="nature"      , category_id=4, ignore_in_eval=False, color=(107, 142, 35)),
        Class(name="terrain"             , id=22, train_id=9  , category="nature"      , category_id=4, ignore_in_eval=False, color=(152, 251, 152)),
        Class(name="sky"                 , id=23, train_id=10 , category="sky"         , category_id=5, ignore_in_eval=False, color=(70 , 130, 180)),
        Class(name="person"              , id=24, train_id=11 , category="human"       , category_id=6, ignore_in_eval=False, color=(220, 20 , 60)),
        Class(name="rider"               , id=25, train_id=12 , category="human"       , category_id=6, ignore_in_eval=False, color=(255, 0  , 0)),
        Class(name="car"                 , id=26, train_id=13 , category="vehicle"     , category_id=7, ignore_in_eval=False, color=(0  , 0  , 142)),
        Class(name="truck"               , id=27, train_id=14 , category="vehicle"     , category_id=7, ignore_in_eval=False, color=(0  , 0  , 70)),
        Class(name="bus"                 , id=28, train_id=15 , category="vehicle"     , category_id=7, ignore_in_eval=False, color=(0  , 60 , 100)),
        Class(name="caravan"             , id=29, train_id=255, category="vehicle"     , category_id=7, ignore_in_eval=True , color=(0  , 0  , 90)),
        Class(name="trailer"             , id=30, train_id=255, category="vehicle"     , category_id=7, ignore_in_eval=True , color=(0  , 0  , 110)),
        Class(name="train"               , id=31, train_id=16 , category="vehicle"     , category_id=7, ignore_in_eval=False, color=(0  , 80 , 100)),
        Class(name="motorcycle"          , id=32, train_id=17 , category="vehicle"     , category_id=7, ignore_in_eval=False, color=(0  , 0  , 230)),
        Class(name="bicycle"             , id=33, train_id=18 , category="vehicle"     , category_id=7, ignore_in_eval=False, color=(119, 11 , 32)),
        Class(name="license plate"       , id=-1, train_id=-1 , category="vehicle"     , category_id=7, ignore_in_eval=True , color=(0  , 0  , 142))
    ])

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
