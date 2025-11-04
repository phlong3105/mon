#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "DEIM: DETR with Improved Matching for Fast
Convergence," CVPR 2025.

References:
    - https://github.com/ShihuaHuang95/DEIM
"""

import os
import sys

import box
import tensorrt as trt
import torch

import albumentations as A
import box
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import mon
from mon import console, metrics, Path, tfms, optims

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from engine.core import YAMLConfig

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Export -----
class Model(torch.nn.Module):

    def __init__(self, cfg: list, export_postprocessor: bool = True):
        super().__init__()
        self.models = torch.nn.ModuleList([c.model.deploy() for c in cfg])
        if export_postprocessor:
            self.postprocessor = cfg[0].postprocessor.deploy()
        else:
            self.postprocessor = None

    def forward(self, images, orig_target_sizes):
        outputs = {
            "pred_logits": [],
            "pred_boxes" : [],
        }
        for model in self.models:
            y = model(images)
            outputs["pred_logits"].append(y["pred_logits"])
            outputs["pred_boxes"].append(y["pred_boxes"])

        outputs["pred_logits"] = torch.stack(outputs["pred_logits"]).mean(0)    # Mean ensemble
        outputs["pred_boxes"]  = torch.stack(outputs["pred_boxes"]).mean(0)     # Mean ensemble
        if self.postprocessor is not None:
            outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs


@torch.no_grad()
def export_onnx(model: Model, path: Path, args: dict | box.Box) -> Path:
    imgsz = args.imgsz[0] if isinstance(args.imgsz, list | tuple) else args.imgsz
    data  = torch.rand(32, 3, imgsz, imgsz)
    size  = torch.tensor([[imgsz, imgsz]])
    _     = model(data, size)
    dynamic_axes = {
        "images"           : {0: "N"},
        "orig_target_sizes": {0: "N"}
    }

    if args.get("export_postprocessor", True):
        output_names = ["labels", "boxes", "scores"]
    else:
        output_names = ["outputs"]

    torch.onnx.export(
        model,
        (data, size),
        path,
        input_names         = ["images", "orig_target_sizes"],
        output_names        = output_names,
        dynamic_axes        = dynamic_axes,
        opset_version       = args.opset,
        verbose             = False,
        do_constant_folding = True,
    )

    check = True
    if check:
        import onnx
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        mon.log("Check export onnx model done...")

    if args.simplify:
        import onnx
        import onnxsim
        dynamic = True
        # input_shapes = {'images': [1, 3, 640, 640], 'orig_target_sizes': [1, 2]} if dynamic else None
        input_shapes = {"images": data.shape, "orig_target_sizes": size.shape} if dynamic else None
        onnx_model_simplify, check = onnxsim.simplify(path, test_input_shapes=input_shapes)
        onnx.save(onnx_model_simplify, path)
        print(f"Simplify onnx model {check}...")


@torch.no_grad()
def export_trt(onnx_path: Path, path: Path, args: dict | box.Box) -> Path:
    if not onnx_path.is_onnx_file(exist=True):
        raise FileNotFoundError(f"Invalid ONNX model: {onnx_path}.")

    # Setup
    logger        = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO)
    builder       = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network       = builder.create_network(network_flags)
    parser        = trt.OnnxParser(network, logger)

    # Load ONNX model
    mon.log(f"Loading ONNX model from: {onnx_path}.")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                mon.error_console.log(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX file!")

    # Create builder config
    config = builder.create_builder_config()
    memory_pool_limit = 8 << 30  # 1 << 30  1GB
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, memory_pool_limit)

    if args.use_fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            mon.log("Apply FP16 optimization.")
        else:
            mon.log("Apply FP32 optimization.")

    # Create optimization profile
    imgsz   = args.imgsz[0] if isinstance(args.imgsz, list | tuple) else args.imgsz
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "images",
        min=(args.min_batch_size, 3, imgsz, imgsz),
        opt=(args.opt_batch_size, 3, imgsz, imgsz),
        max=(args.max_batch_size, 3, imgsz, imgsz)
    )
    profile.set_shape("orig_target_sizes", min=(1, 2), opt=(1, 2), max=(1, 2))
    config.add_optimization_profile(profile)

    # Customize layer precision
    if args.opset == 16:
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            # Heuristic: match common LayerNorm-related names
            if any(kw in layer.name.lower() for kw in ["layernorm", "norm", "rms"]):
                mon.log(f"Apply FP32 on LayerNorm-related layer: {layer.name}.")
                layer.precision = trt.DataType.FLOAT
                layer.set_output_type(0, trt.DataType.FLOAT)
    elif args.opset == 17:
        force_fp32_types = [
            trt.LayerType.REDUCE,
            trt.LayerType.ELEMENTWISE,
            trt.LayerType.UNARY,
            trt.LayerType.NORMALIZATION,
        ]
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type in force_fp32_types:
                layer.precision = trt.DataType.FLOAT
                layer.set_output_type(0, trt.DataType.FLOAT)

    # Debug
    # for i in range(network.num_layers):
    #     layer = network.get_layer(i)
    #     mon.log(f"Layer {i}: {layer.name} | Type: {layer.type}")

    mon.log("Building TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build the engine.")

    mon.log(f"Saving engine to {path}")
    with open(path, "wb") as f:
        f.write(serialized_engine)
    mon.log("Engine export complete.")


@torch.no_grad()
def export(args: dict | box.Box) -> str:
    # Start
    mon.rt.print_run_summary(args)

    # Device
    device = mon.create_device(args.device)

    # Seed
    mon.set_random_seed(args.seed)

    # Pretrained
    pretrained = None
    if args.weights and isinstance(args.weights, list | tuple):
        pretrained = args.weights
    if pretrained and isinstance(pretrained, list | tuple):
        mon.log(f"Pretrained: {pretrained}.")
    else:
        raise ValueError(f"Invalid weights file: {pretrained}.")

    # Model
    if not isinstance(args.cfg, list | tuple):
        raise ValueError(f"Invalid cfg: {args.cfg}.")

    cfgs = []
    for i, cfg in enumerate(args.cfg):
        cfg_path     = root_dir / "option" / cfg
        updated_cfg  = args.updated_cfg[i]
        updated_cfg |= {"resume": str(pretrained[i])} if pretrained[i] else {}
        updated_cfg |= {
            "device": device,
            "seed"  : args.seed,
        }
        cfg = YAMLConfig(cfg_path=str(cfg_path), root=str(args.root), **updated_cfg)

        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        if pretrained[i]:
            checkpoint = torch.load(pretrained[i], map_location="cpu")
            if "ema" in checkpoint:
                state = checkpoint["ema"]["module"]
            else:
                state = checkpoint["model"]
        else:
            raise AttributeError("Only support resume to load model.state_dict by now.")

        # Load train mode state and convert to deploy mode
        cfg.model.load_state_dict(state)
        cfgs.append(cfg)

    model = Model(cfgs, export_postprocessor=args.export_postprocessor)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Export ONNX model (always export ONNX first)
    save_dir  = pretrained.parent if args.save_nearby  else args.save_dir
    file_stem = args.fullname     if args.use_fullname else pretrained.stem
    onnx_file = save_dir / f"{file_stem}.onnx"
    export_onnx(model, onnx_file, args)
    mon.log(f"Exported ONNX model to: {onnx_file}.")

    # Export TensorRT engine
    if args.format in ["engine", "trt"]:
        engine_file = save_dir / f"{file_stem}.engine"
        export_trt(onnx_file, engine_file, args)


# ----- Main -----
def main() -> str:
    args = mon.rt.parse_predict_args(model_root=root_dir)
    export(args)


if __name__ == "__main__":
    main()
