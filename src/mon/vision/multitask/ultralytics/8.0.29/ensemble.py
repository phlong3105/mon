#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the ensemble scripts for YOLOv8."""

from __future__ import annotations

import sys
from collections import OrderedDict

import click

import mon
from ultralytics import YOLO, yolo  # noqa
from ultralytics.nn.tasks import (
    ClassificationModel,
    DetectionModel,
    SegmentationModel,
    attempt_load_one_weight,
    attempt_load_weights,
    guess_model_task, nn
)
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.engine.exporter import Exporter
from ultralytics.yolo.utils import (
    callbacks, DEFAULT_CFG, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, is_git_dir,
    LOGGER, RANK, ROOT, yaml_load,
)
from ultralytics.yolo.utils.checks import (
    check_imgsz,
    check_yaml,
)
from ultralytics.yolo.utils.torch_utils import smart_inference_mode

_current_dir = mon.Path(__file__).absolute().parent


# region YOLOEnsemble

class YOLOEnsemble:
    
    TASK_MAP = {
        "classify": [
            ClassificationModel,
            yolo.v8.classify.ClassificationTrainer,
            yolo.v8.classify.ClassificationValidator,
            yolo.v8.classify.ClassificationPredictor
        ],
        "detect": [
            DetectionModel,
            yolo.v8.detect.DetectionTrainer,
            yolo.v8.detect.DetectionValidator,
            yolo.v8.detect.DetectionPredictor
        ],
        "segment": [
            SegmentationModel,
            yolo.v8.segment.SegmentationTrainer,
            yolo.v8.segment.SegmentationValidator,
            yolo.v8.segment.SegmentationPredictor
        ]
    }
    
    def __init__(self, model="yolov8n.pt", task=None, session=None):
        self._reset_callbacks()
        self.predictor = None     # reuse predictor
        self.model     = None     # model object
        self.trainer   = None     # trainer object
        self.task      = None     # task type
        self.ckpt      = None     # if loaded from *.pt
        self.cfg       = None     # if loaded from *.yaml
        self.ckpt_path = None
        self.overrides = {}       # overrides for trainer object
        self.metrics   = None     # validation/training metrics
        self.session   = session  # HUB session

    def __call__(self, source=None, stream: bool = False, **kwargs):
        return self.predict(source, stream, **kwargs)

    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(
            f"'{name}' object has no attribute '{attr}'. "
            f"See valid attributes below.\n{self.__doc__}"
        )

    def new(self, cfg: str, task=None, verbose: bool = True):
        self.cfg   = check_yaml(cfg)  # check YAML
        cfg_dict   = yaml_load(self.cfg, append_filename=True)  # model dict
        # 3 task 'segment' 'classify' 'detect'
        # self.task = 'detect'
        self.task  = task or guess_model_task(cfg_dict)
        self.model = self.TASK_MAP[self.task][0](cfg_dict, verbose=verbose and RANK == -1)  # build model
        self.overrides["model"] = self.cfg

        # Below added to allow export from yamls
        args            = {**DEFAULT_CFG_DICT, **self.overrides}  # combine model and default args, preferring model args
        self.model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
        self.model.task = self.task
    
    def load(self, weights: str, task=None):
        if isinstance(weights, list):
            # self.model, self.ckpt = attempt_load_weights(weights)
            # self.task = self.model.args['task']
            # self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            # self.ckpt_path = self.model.pt_path
            self.model, self.ckpt = weights, None
            self.task      = task or guess_model_task(weights)
            self.ckpt_path = weights
        elif isinstance(weights, str):
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task      = self.model.args["task"]
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        self.overrides["model"] = weights

    def _check_is_pytorch_model(self):
        if not isinstance(self.model, nn.Module):
            raise TypeError(
                f"model='{self.model}' must be a *.pt PyTorch model, but is a different type. "
                f'PyTorch models can be used to train, val, predict and export, i.e. '
                f"'yolo export model=yolov8n.pt', but exported formats like ONNX, TensorRT etc. only "
                f"support 'predict' and 'val' modes, i.e. 'yolo predict model=yolov8n.onnx'."
            )

    def _check_pip_update(self):
        """
        if ONLINE and is_pip_package():
            check_pip_update()
        """
        return

    def reset(self):
        self._check_is_pytorch_model()
        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True

    def info(self, verbose: bool = False):
        self._check_is_pytorch_model()
        self.model.info(verbose=verbose)

    def fuse(self):
        self._check_is_pytorch_model()
        self.model.fuse()

    @smart_inference_mode()
    def predict(self, source=None, stream: bool = False, **kwargs):
        if source is None:
            source = ROOT / "assets" if is_git_dir() else "https://ultralytics.com/images/bus.jpg"
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")
        is_cli = (sys.argv[0].endswith("yolo") or sys.argv[0].endswith("ultralytics")) \
                 and ("predict" in sys.argv or "mode=predict" in sys.argv)

        overrides = self.overrides.copy()
        overrides["conf"] = 0.25
        overrides.update(kwargs)  # prefer kwargs
        overrides["mode"] = kwargs.get("mode", "predict")
        assert overrides["mode"] in ["track", "predict"]
        overrides["save"] = kwargs.get("save", False)  # not save files by default
        if not self.predictor:
            self.task      = overrides.get("task") or self.task
            self.predictor = self.TASK_MAP[self.task][3](overrides=overrides)
            self.predictor.setup_model(model=self.model)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, overrides)
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)
    
    @smart_inference_mode()
    def val(self, data=None, **kwargs):
        overrides = self.overrides.copy()
        overrides["rect"] = True  # rect batches as default
        overrides.update(kwargs)
        overrides["mode"] = "val"
        args      = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.data = data or args.data
        if 'task' in overrides:
            self.task = args.task
        else:
            args.task = self.task
        if args.imgsz == DEFAULT_CFG.imgsz and not isinstance(self.model, (str, mon.Path)):
            args.imgsz = self.model.args["imgsz"]  # use trained imgsz unless custom value is passed
        args.imgsz = check_imgsz(args.imgsz, max_dim=1)

        validator = self.TASK_MAP[self.task][2](args=args)
        validator(model=self.model)
        self.metrics = validator.metrics

        return validator.metrics
    
    def export(self, **kwargs):
        self._check_is_pytorch_model()
        overrides = self.overrides.copy()
        overrides.update(kwargs)
        args      = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.task = self.task
        if args.imgsz == DEFAULT_CFG.imgsz:
            args.imgsz = self.model.args["imgsz"]  # use trained imgsz unless custom value is passed
        if args.batch == DEFAULT_CFG.batch:
            args.batch = 1  # default to 1 if not modified
        return Exporter(overrides=args)(model=self.model)

    def train(self, **kwargs):
        self._check_is_pytorch_model()
        self._check_pip_update()
        overrides = self.overrides.copy()
        overrides.update(kwargs)
        if kwargs.get("cfg"):
            LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
            overrides = yaml_load(check_yaml(kwargs["cfg"]))
        overrides['mode'] = "train"
        if not overrides.get("data"):
            raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
        if overrides.get("resume"):
            overrides["resume"] = self.ckpt_path

        self.task    = overrides.get("task") or self.task
        self.trainer = self.TASK_MAP[self.task][1](overrides=overrides)
        if not overrides.get("resume"):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model         = self.trainer.model
        self.trainer.hub_session = self.session  # attach optional HUB session
        self.trainer.train()
        # update model and cfg after training
        if RANK in {0, -1}:
            self.model, _  = attempt_load_one_weight(str(self.trainer.best))
            self.overrides = self.model.args
            self.metrics   = getattr(self.trainer.validator, "metrics", None)  # TODO: no metrics returned by DDP

    def to(self, device):
        self._check_is_pytorch_model()
        self.model.to(device)

    @property
    def names(self):
        return self.model.names if hasattr(self.model, "names") else None

    @property
    def device(self):
        return next(self.model.parameters()).device if isinstance(self.model, nn.Module) else None

    @property
    def transforms(self):
        return self.model.transforms if hasattr(self.model, "transforms") else None

    @staticmethod
    def add_callback(event: str, func):
        callbacks.default_callbacks[event].append(func)

    @staticmethod
    def _reset_ckpt_args(args):
        include = {"imgsz", "data", "task", "single_cls"}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}

    @staticmethod
    def _reset_callbacks():
        for event in callbacks.default_callbacks.keys():
            callbacks.default_callbacks[event] = [callbacks.default_callbacks[event][0]]


def adjust_state_dict(state_dict: OrderedDict) -> OrderedDict:
    od = OrderedDict()
    for key, value in state_dict.items():
        new_key     = key.replace("module.", "")
        od[new_key] = value
    return od


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad  = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4]     /= gain
    # clip_coords(coords, img0_shape)
    return coords

# endregion


# region Function

@click.command()
@click.option("--task",           default="detect", help="Inference task, i.e. detect, segment, or classify.")
@click.option("--model",          default=["weight/yolov8n.pt"], multiple=True, help="Path to model file, i.e. yolov8n.pt, yolov8n.yaml.")
@click.option("--data",           default="data/visdrone-a2i2-of.yaml", type=click.Path(exists=True), help="Path to data file, i.e. i.e. coco128.yaml.")
@click.option("--project",        default="run/train", type=click.Path(exists=False), help="Project name.")
@click.option("--name",           default="exp", help="Experiment name.")
@click.option("--source",         default=mon.DATA_DIR, type=click.Path(exists=True), help="Source directory for images or videos.")
@click.option("--imgsz",          default=1280, type=int,   help="Size of input images as integer or w,h.")
@click.option("--conf",           default=0.25, type=float, help="Object confidence threshold for detection.")
@click.option("--iou",            default=0.7,  type=float, help="Intersection over union (IoU) threshold for NMS.")
@click.option("--max-det",        default=300,  type=int,   help="Maximum number of detections per image.")
@click.option("--augment",        is_flag=True, help="Apply image augmentation to prediction sources.")
@click.option("--agnostic-nms",   is_flag=True, help="class-agnostic NMS.")
@click.option("--device",         default="cpu", help="Device to run on, i.e. cuda device=0/1/2/3 or device=cpu.")
@click.option("--stream",         is_flag=True)
@click.option("--exist-ok",       is_flag=True, help="Whether to overwrite existing experiment.")
@click.option("--show",           is_flag=True, help="Show results if possible.")
@click.option("--visualize",      is_flag=True, help="Visualize model features.")
@click.option("--save",           is_flag=True, help="Save train checkpoints and predict results.")
@click.option("--save-txt",       is_flag=True, help="Save results as .txt file.")
@click.option("--save-conf",      is_flag=True, help="Save results with confidence scores.")
@click.option("--save-mask",      is_flag=True, help="Save binary segmentation mask.")
@click.option("--save-crop",      is_flag=True, help="Save cropped images with results.")
@click.option("--hide-labels",    is_flag=True, help="Hide labels.")
@click.option("--hide-conf",      is_flag=True, help="Hide confidence scores.")
@click.option("--vid-stride",     default=1, type=int, help="Video frame-rate stride.")
@click.option("--overlap-mask",   is_flag=True)
@click.option("--line-thickness", default=4, type=int, help="Bounding box thickness (pixels).")
@click.option("--retina-masks",   is_flag=True, help="Use high-resolution segmentation masks.")
@click.option("--classes",        type=int, help="Filter results by class, i.e. class=0, or class=[0,2,3].")
@click.option("--box",            is_flag=True, help="Show boxes in segmentation predictions.")
def predict(
    task, model, data, project, name, source, imgsz, conf, iou, max_det,
    augment, agnostic_nms, device, stream, exist_ok, show, visualize, save,
    save_txt, save_conf, save_mask, save_crop, hide_labels, hide_conf,
    vid_stride, overlap_mask, line_thickness, retina_masks, classes, box,
):
    model  = list(model)
    models = YOLOEnsemble(task=task)
    models.load(weights=model, task=task)
    
    # Predict with the model
    args = {
        "task"          : task,
        "mode"          : "predict",
        "data"          : data,
        "project"       : project,
        "name"          : name,
        "source"        : source,
        "imgsz"         : imgsz,
        "conf"          : conf,
        "iou"           : iou,
        "max_det"       : max_det,
        "augment"       : augment,
        "agnostic_nms"  : agnostic_nms,
        "device"        : device,
        # "stream"        : stream,
        "exist_ok"      : exist_ok,
        "show"          : show,
        "visualize"     : visualize,
        "save"          : save,
        "save_txt"      : save_txt,
        "save_conf"     : save_conf,
        "save_mask"     : save_mask,
        "save_crop"     : save_crop,
        "hide_labels"   : hide_labels,
        "hide_conf"     : hide_conf,
        "vid_stride"    : vid_stride,
        "overlap_mask"  : overlap_mask,
        "line_thickness": line_thickness,
        "retina_masks"  : retina_masks,
        "classes"       : classes,
        "box"           : box,
    }
    results = models(stream=stream, **args)

# endregion


# region Main

if __name__ == "__main__":
    predict()

# endregion
