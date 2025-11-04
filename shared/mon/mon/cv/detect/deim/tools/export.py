#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DEIM model for object detection.

References:
    - Paper: "DEIM: DETR with Improved Matching for Fast convergence," CVPR 2025.
    - Code: https://github.com/ShihuaHuang95/DEIM
"""

import os
import sys

import box
import tensorrt as trt
import torch

import mon

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from engine.core import YAMLConfig

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Utils -----
class Model(torch.nn.Module):

    def __init__(self, cfg, export_postprocessor: bool = True):
        super().__init__()
        self.model = cfg.model.deploy()
        if export_postprocessor:
            self.postprocessor = cfg.postprocessor.deploy()
        else:
            self.postprocessor = None

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        if self.postprocessor is not None:
            outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs


class ModelEnsemble(torch.nn.Module):

    def __init__(self, cfg: list, export_postprocessor: bool = True):
        super().__init__()
        if not isinstance(cfg, list | tuple):
            raise TypeError(f"``cfg`` must be a list or tuple of configurations, got {type(cfg)}.")

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


# ----- Export -----
@torch.no_grad()
def export_onnx(model: Model, path: Path, args: dict | box.Box) -> Path:
    opset    = args.opset
    simplify = args.simplify
    imgsz    = args.imgsz[0] if isinstance(args.imgsz, list | tuple) else args.imgsz
    data     = torch.rand(32, 3, imgsz, imgsz)
    size     = torch.tensor([[imgsz, imgsz]])
    _        = model(data, size)
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
        opset_version       = opset,
        verbose             = False,
        do_constant_folding = True,
    )

    check = True
    if check:
        import onnx
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        mon.log("Check export onnx model done...")

    if simplify:
        import onnx
        import onnxsim
        dynamic = True
        # input_shapes = {'images': [1, 3, 640, 640], 'orig_target_sizes': [1, 2]} if dynamic else None
        input_shapes = {"images": data.shape, "orig_target_sizes": size.shape} if dynamic else None
        onnx_model_simplify, check = onnxsim.simplify(path, test_input_shapes=input_shapes)
        onnx.save(onnx_model_simplify, path)
        print(f"Simplify onnx model {check}...")


@torch.no_grad()
def export_trt(onnx_path: Path, engine_path: Path, args: dict | box.Box) -> Path:
    onnx_path   = Path(onnx_path)
    engine_path = Path(engine_path)
    imgsz       = args.imgsz[0] if isinstance(args.imgsz, list | tuple) else args.imgsz
    opset       = args.opset
    trt_p       = args.trt_precision

    if not onnx_path.is_onnx_file(exist=True):
        raise FileNotFoundError(f"Invalid ONNX file: {onnx_path}.")

    if trt_p not in mon.TRTPrecision:
        raise ValueError(f"``fp`` must be one of {mon.TRTPrecision.values()}, got {trt_p}.")

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
    config.builder_optimization_level = 5   # Maximum optimization level

    if trt_p in ["fp16n32", "fp16", "fp8n32", "fp8", "int8n32", "int8"]:
        # if dla_core not in [None, -1]:  # For Jetson devices
        config.DLA_core = 0
        config.default_device_type = trt.DeviceType.DLA
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        print("Apply DLA core.")

    if trt_p in ["fp16n32", "fp16"]:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("Apply FP16 optimization.")
        else:
            print("Apply FP32 optimization.")
    elif trt_p in ["int8n32", "int8"]:
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("Apply INT8 optimization.")
        else:
            print("Apply FP32 optimization.")

    # Create optimization profile
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "images",
        min=(args.min_batch_size, 3, imgsz, imgsz),
        opt=(args.opt_batch_size, 3, imgsz, imgsz),
        max=(args.max_batch_size, 3, imgsz, imgsz)
    )
    profile.set_shape("orig_target_sizes", min=(1, 2), opt=(1, 2), max=(1, 2))
    config.add_optimization_profile(profile)

    # Retain FP32 for specific layers
    if opset == 16:
        if trt_p in ["fp16n32", "fp8n32", "int8n32"]:
            layer_names = ["layernorm", "norm", "rms"]
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                # Heuristic: match common LayerNorm-related names
                if any(kw in layer.name.lower() for kw in layer_names):
                    print(f"Apply FP32 on LayerNorm-related layer: {layer.name}.")
                    layer.precision = trt.DataType.FLOAT
                    layer.set_output_type(0, trt.DataType.FLOAT)

    mon.log("Building TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build the engine.")

    mon.log(f"Saving engine to {engine_path}")
    with open(engine_path, "wb") as f:
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
    pretrained = args.resume
    if args.weights and (isinstance(args.weights, list | tuple) or args.weights.is_weights_file(exist=True)):
        pretrained = args.weights
    if pretrained and (isinstance(pretrained, list | tuple) or pretrained.is_weights_file(exist=True)):
        mon.log(f"Pretrained: {pretrained}.")
    else:
        raise ValueError(f"Invalid weights file: {pretrained}.")
    if not isinstance(pretrained, list | tuple):
        pretrained = [pretrained]

    # Model
    cfg         = args.cfg
    updated_cfg = args.updated_cfg
    if not isinstance(args.cfg, list | tuple):
        cfg         = [args.cfg]
        updated_cfg = [args.updated_cfg]
    if len(cfg) != len(pretrained):
        raise ValueError(f"Number of configurations ({len(cfg)}) does not match number of pretrained weights ({len(pretrained)}).")

    for i in range(len(cfg)):
        cfg_path        = root_dir / "option" / cfg[i]
        updated_cfg[i] |= {"resume": str(pretrained[i])} if pretrained[i] else {}
        updated_cfg[i] |= {
            "device": device,
            "seed"  : args.seed,
        }
        cfg[i] = YAMLConfig(cfg_path=str(cfg_path), root=str(args.root), **updated_cfg[i])

        if "HGNetv2" in cfg[i].yaml_cfg:
            cfg[i].yaml_cfg["HGNetv2"]["pretrained"] = False

        if pretrained[i]:
            checkpoint = torch.load(pretrained[i], map_location="cpu")
            if "ema" in checkpoint:
                state = checkpoint["ema"]["module"]
            else:
                state = checkpoint["model"]
        else:
            raise AttributeError("Only support resume to load model.state_dict by now.")

        # Load train mode state and convert to deploy mode
        cfg[i].model.load_state_dict(state)

    if len(cfg) == 1:  # Single model
        model = Model(cfg[0], export_postprocessor=args.export_postprocessor)
    else:
        model = ModelEnsemble(cfg, export_postprocessor=args.export_postprocessor)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Export ONNX model (always export ONNX first)
    # save_dir  = pretrained.parent if args.save_nearby  else args.save_dir
    # file_stem = args.fullname     if args.use_fullname else pretrained.stem
    save_dir  = args.save_dir
    file_stem = args.fullname
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
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", "-c", type=str, default="options/deim_dfine/dfine_hgnetv2_l_coco80.yml")
    # parser.add_argument("--resume", "-r", type=str)
    # parser.add_argument("--check",        action="store_true", default=True)
    # parser.add_argument("--simplify",     action="store_true", default=True)
    # args = parser.parse_args()
    # main(args)
    main()
