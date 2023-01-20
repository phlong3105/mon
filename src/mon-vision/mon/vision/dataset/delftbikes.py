#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DelftBikes datasets and datamodules.
"""

from __future__ import annotations

import argparse
import json

from one.constants import *
from one.core import *
from one.data import ClassLabels
from one.data import ClassLabels_
from one.data import DataModule
from one.data import Image
from one.data import load_from_file
from one.data import YOLODetectionDataset
from one.vision.shape import box_xyxy_to_cxcywh_norm
from one.vision.transformation import Resize


# H1: - Functional -------------------------------------------------------------

def generate_train_val(
    root       : Path_,
    json_file  : Path_,
    val_size   : int,
    yolo_labels: bool = True,
    *args, **kwargs
):
    root       = Path(root)
    json_file  = Path(json_file)
    src_path   = root / "trainval" / "images"
    train_path = root / "train"    / "images"
    val_path   = root / "val"      / "images"
    img_list   = os.listdir(src_path)
    val_list   = img_list[:val_size]
    train_list = img_list[val_size:]
    json_data  = load_from_file(open(root / "trainval" / json_file))
    
    # Val set generation
    create_dirs(paths=[val_path])
    for img in val_list:
        shutil.copy(os.path.join(src_path, img), val_path)
    
    valset_list = os.listdir(val_path)
    if len(valset_list) == 1000:
        print("Val images are successfully generated.")
    
    val_dict = {}
    for im_name in valset_list:
        val_dict[im_name] = json_data[im_name]
    
    with open(os.path.join(root, "val", "val_annotations.json"), "w") as outfile:
        json.dump(val_dict, outfile)
    
    if yolo_labels:
        generate_yolo_labels(
            root      = os.path.join(root, "val"),
            json_file = "val_annotations.json"
        )
    print("Val labels are successfully generated.")
    
    # Train set generation
    create_dirs(paths=[train_path])
    for img in train_list:
        shutil.copy(os.path.join(src_path, img), train_path)
        
    trainset_list = os.listdir(train_path)
    if len(trainset_list) == 7000:
        print("Train images are successfully generated.")
        
    train_dict = {}
    for im_name in train_list:
        train_dict[im_name] = json_data[im_name]
    
    with open(os.path.join(root, "train", "train_annotations.json"), "w") as outfile:
        json.dump(train_dict, outfile)
    
    if yolo_labels:
        generate_yolo_labels(
            root      = os.path.join(root, "train"),
            json_file = "train_annotations.json"
        )
    print("Train labels are successfully generated.")


def generate_yolo_labels(
    root: Path_, json_file: Path_, *args, **kwargs
):
    root        = Path(root)
    json_file   = Path(json_file)
    # images    = list(sorted(os.listdir(os.path.join(root, "images"))))
    json_data   = load_from_file(root / json_file)
    labels_path = root / "yolo_labels"
    
    create_dirs(paths=[labels_path])

    with progress_bar () as pbar:
        for k, v in pbar.track(
            json_data.items(), description=f"[red]Processing labels"
        ):
            image_path = root / "images" / k
            image      = v["image"]
            channels   = image["channels"]
            height     = image["height"]
            width      = image["width"]
    
            ids   = []
            boxes = []
            names = []
            confs = []
            for idx, i in enumerate(v["parts"], 0):
                label = v["parts"][i]
                if label["object_state"] != "absent":
                    loc = label["absolute_bounding_box"]
                    x1  = loc["left"]
                    x2  = loc["left"] + loc["width"]
                    y1  = loc["top"]
                    y2  = loc["top"] + loc["height"]
                    boxes.append(torch.FloatTensor([x1, y1, x2, y2]))
                    ids.append(idx)
                    names.append(label["part_name"])
                    confs.append(label["trust"])
            
            boxes = torch.stack(boxes, 0)
            boxes = box_xyxy_to_cxcywh_norm(boxes, height, width)
            
            yolo_file = labels_path / k.replace(".jpg", ".txt")
            with open(yolo_file, mode="w") as f:
                for i, b in enumerate(boxes):
                    b = b.numpy()
                    f.write(f"{ids[i]} {b[0]} {b[1]} {b[2]} {b[3]} {confs[i]}\n")


def split_4subsets(root: Path_, *args, **kwargs ):
    """
    Split the original train-val set into 4-fold subsets of 6000 train and
    2000 val.
    """
    root         = Path(root)
    trainval_dir = root / "trainval"
    kfold_dir    = root / "kfold"
    subsets      = ["set1", "set2", "set3", "set4"]
    
    images = sorted(list(trainval_dir.rglob("images/*.jpg")))
    labels = sorted(list(trainval_dir.rglob("yolo_labels/*.txt")))
    data   = [[i, l] for i, l in zip(images, labels)]
    random.shuffle(data)
    data_kfold = [
        data[0:2000],
        data[2000:4000],
        data[4000:6000],
        data[6000:8000]
    ]
    
    with progress_bar() as pbar:
        for i in pbar.track(range(4), description=f"Splitting subsets"):
            new_images_dir = kfold_dir / subsets[i] / "images"
            new_labels_dir = kfold_dir / subsets[i] / "yolo_labels"
            create_dirs([new_images_dir, new_labels_dir])
            for data in data_kfold[i]:
                shutil.copy2(data[0], new_images_dir)
                shutil.copy2(data[1], new_labels_dir)
                

# H1: - Dataset ----------------------------------------------------------------

delftbikes_classlabels = [
    {"id": 0 , "supercategory": "bike", "name": "back_hand_break", "color": [  0,   0,   0]},
    {"id": 1 , "supercategory": "bike", "name": "back_handle"    , "color": [ 81, 120, 228]},
    {"id": 2 , "supercategory": "bike", "name": "back_light"     , "color": [138, 183,  33]},
    {"id": 3 , "supercategory": "bike", "name": "back_mudguard"  , "color": [ 49,   3, 150]},
    {"id": 4 , "supercategory": "bike", "name": "back_pedal"     , "color": [122,  35,   2]},
    {"id": 5 , "supercategory": "bike", "name": "back_reflector" , "color": [165, 168, 193]},
    {"id": 6 , "supercategory": "bike", "name": "back_wheel"     , "color": [140,  24, 143]},
    {"id": 7 , "supercategory": "bike", "name": "bell"           , "color": [179, 165, 212]},
    {"id": 8 , "supercategory": "bike", "name": "chain"          , "color": [ 72, 153, 152]},
    {"id": 9 , "supercategory": "bike", "name": "dress_guard"    , "color": [ 19,  64,  83]},
    {"id": 10, "supercategory": "bike", "name": "dynamo"         , "color": [122,  40,  57]},
    {"id": 11, "supercategory": "bike", "name": "front_handbreak", "color": [219,  42, 205]},
    {"id": 12, "supercategory": "bike", "name": "front_handle"   , "color": [ 15,  90, 125]},
    {"id": 13, "supercategory": "bike", "name": "front_light"    , "color": [187,  80,  10]},
    {"id": 14, "supercategory": "bike", "name": "front_mudguard" , "color": [ 76, 226, 142]},
    {"id": 15, "supercategory": "bike", "name": "front_pedal"    , "color": [ 24,  56,  34]},
    {"id": 16, "supercategory": "bike", "name": "front_wheel"    , "color": [ 41, 174, 251]},
    {"id": 17, "supercategory": "bike", "name": "gear_case"      , "color": [ 21,   8, 251]},
    {"id": 18, "supercategory": "bike", "name": "kickstand"      , "color": [106, 128, 177]},
    {"id": 19, "supercategory": "bike", "name": "lock"           , "color": [147,  90, 131]},
    {"id": 20, "supercategory": "bike", "name": "saddle"         , "color": [ 65, 159, 189]},
    {"id": 21, "supercategory": "bike", "name": "steer"          , "color": [129,  70,  30]},
]


@DATASETS.register(name="delftbikes-yolo")
class DelftBikesYOLO(YOLODetectionDataset):
    """
    
    Args:
        name (str): Dataset's name.
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        classlabels (ClassLabels_ | None): ClassLabels object. Defaults to
            None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """

    def __init__(
        self,
        root            : Path_,
        split           : str,
        shape           : Ints,
        classlabels     : ClassLabels_ | None = None,
        transform       : Transforms_  | None = None,
        target_transform: Transforms_  | None = None,
        transforms      : Transforms_  | None = None,
        cache_images    : bool                = False,
        cache_data      : bool                = False,
        backend         : VisionBackend_      = VISION_BACKEND,
        verbose         : bool                = True,
        *args, **kwargs
    ):
        path = os.path.join(root, f"classlabels.json")
        if classlabels is None:
            if os.path.isfile(path):
                classlabels = ClassLabels.from_file(path)
            else:
                classlabels = ClassLabels.from_list(delftbikes_classlabels)
        
        super().__init__(
            name             = "delftbikes-yolo",
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
    
    def list_images(self):
        """
        List image files.
        """
        if self.split not in ["train", "val"]:
            console.log(
                f"{self.__class__.classname} dataset only supports `split`: "
                f"`train` or `val`. Get: {self.split}."
            )
        
        self.images: list[Image] = []
        with progress_bar() as pbar:
            pattern = self.root / self.split
            for path in pbar.track(
                list(pattern.rglob("images/*.jpg")),
                description=f"Listing {self.__class__.classname} "
                            f"{self.split} images"
            ):
                self.images.append(Image(path=path, backend=self.backend))

    def annotation_files(self) -> Paths_:
        """
        Returns the path to json annotation files.
        """
        files: list[Path] = []
        for img in self.images:
            path = str(img.path)
            path = path.replace("images", "yolo_labels")
            path = path.replace(".jpg", ".txt")
            files.append(Path(path))
        return files
    
    def filter(self):
        """
        Filter unwanted samples.
        """
        pass


# H1: - Datamodule -------------------------------------------------------------

@DATAMODULES.register(name="delftbikes-yolo")
class DelftBikesYOLODataModule(DataModule):
    """
    DelftBikes-YOLO DataModule.
    """
    
    def __init__(
        self,
        name            : str                = "delftbikes-yolo",
        root            : Path_              = DATA_DIR / "delftbikes",
        shape           : Ints               = (3, 512, 512),
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        batch_size      : int                = 1,
        devices         : Devices            = 0,
        shuffle         : bool               = True,
        collate_fn      : Callable    | None = None,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            shape            = shape,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            batch_size       = batch_size,
            devices          = devices,
            shuffle          = shuffle,
            collate_fn       = collate_fn,
            verbose          = verbose,
            *args, **kwargs
        )
        
    def prepare_data(self, *args, **kwargs):
        """
        Use this method to do things that might write to disk or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.load_classlabels()
    
    def setup(self, phase: ModelPhase_ | None = None):
        """
        There are also data operations you might want to perform on every GPU.

        Todos:
            - Count number of classes.
            - Build classlabels vocabulary.
            - Perform train/val/test splits.
            - Apply transforms (defined explicitly in your datamodule or
              assigned in init).
            - Define collate_fn for you custom dataset.

        Args:
            phase (ModelPhase_ | None):
                Stage to use: [None, ModelPhase.TRAINING, ModelPhase.TESTING].
                Set to None to setup all train, val, and test data.
                Defaults to None.
        """
        console.log(f"Setup [red]{DelftBikesYOLO.classname}[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            self.train = DelftBikesYOLO(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.val = DelftBikesYOLO(
                root             = self.root,
                split            = "val",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.train, "classlabels", None)
            self.collate_fn  = getattr(self.train, "collate_fn",  None)
            
        # Assign test datasets for use in dataloader(s)
        if phase in [None, ModelPhase.TESTING]:
            self.test = DelftBikesYOLO(
                root             = self.root,
                split            = "val",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.load_classlabels()

        self.summarize()
        
    def load_classlabels(self):
        """
        Load ClassLabels.
        """
        self.classlabels = ClassLabels.from_value(delftbikes_classlabels)


# H1: - Test -------------------------------------------------------------------

def test_delftbikes_yolo():
    cfg = {
        "name": "delftbikes_yolo",
            # Dataset's name.
        "root": DATA_DIR / "delftbikes",
            # Root directory of dataset.
        "shape": [3, 512, 512],
            # Image shape as [C, H, W], [H, W], or [S, S].
        "transform": [
            Resize(size=[3, 512, 512]),
        ],
            # Functions/transforms that takes in an input sample and returns a
            # transformed version.
        "target_transform": None,
            # Functions/transforms that takes in a target and returns a
            # transformed version.
        "transforms": None,
            # Functions/transforms that takes in an input and a target and
            # returns the transformed versions of both.
        "cache_data": True,
            # If True, cache data to disk for faster loading next time.
            # Defaults to False.
        "cache_images": False,
            # If True, cache images into memory for faster training (WARNING:
            # large datasets may exceed system RAM). Defaults to False.
        "backend": VISION_BACKEND,
            # Vision backend to process image. Defaults to VISION_BACKEND.
        "batch_size": 8,
            # Number of samples in one forward & backward pass. Defaults to 1.
        "devices" : 0,
            # The devices to use. Defaults to 0.
        "shuffle": True,
            # If True, reshuffle the data at every training epoch.
            # Defaults to True.
        "verbose": True,
            # Verbosity. Defaults to True.
    }
    dm  = DelftBikesYOLODataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    
    data_iter           = iter(dm.train_dataloader)
    start = time.time()
    input, target, meta = next(data_iter)
    end = time.time()
    console.log(end - start)
    # print(target)
    # print(target[0].shape)
    # imshow(winname="image", image=input)
    # plt.show(block=True)


# H1: - Main -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task"       , type=str , default="split_4subsets"        , help="The task to run")
    parser.add_argument("--root"       , type=str , default=DATA_DIR / "delftbikes" , help="Root of the dataset")
    parser.add_argument("--json-file"  , type=str , default="train_annotations.json", help="JSON file")
    parser.add_argument("--val_size"   , type=int , default=1000                    , help="Size of the validation set to split")
    parser.add_argument("--yolo_labels", type=bool, default=True                    , help="Generate YOLO label when splitting train-val set")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.task == "generate-train-val":
        generate_train_val(**args.__dict__)
    elif args.task == "generate-yolo-labels":
        generate_yolo_labels(**args.__dict__)
    elif args.task == "split-4subsets":
        split_4subsets(**args.__dict__)
    elif args.task == "test-delftbikes-yolo":
        test_delftbikes_yolo()
