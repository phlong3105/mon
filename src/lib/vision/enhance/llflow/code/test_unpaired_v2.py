#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/wyf0912/LLFlow

from __future__ import annotations

import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
from natsort import natsort

import mon
import options.options as option
from models import create_model

console = mon.console

def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))
 

def predict(model, lr):
    model.feed_data({"LQ": t(lr)}, need_GT=False)
    model.test()
    visuals = model.get_current_visuals(need_GT=False)
    return visuals.get("rlt", visuals.get("NORMAL"))


def t(array):
    return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255


def rgb(t):
    return (np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(np.uint8)


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img[:, :, [2, 1, 0]])


def imCropCenter(img, size):
    h, w, c = img.shape
    h_start = max(h // 2 - size // 2, 0)
    h_end   = min(h_start + size, h)
    w_start = max(w // 2 - size // 2, 0)
    w_end   = min(w_start + size, w)
    return img[h_start:h_end, w_start:w_end]


def impad(img, top=0, bottom=0, left=0, right=0, color=255):
    return np.pad(img, [(top, bottom), (left, right), (0, 0)], "reflect")


def hiseq_color_cv2_img(img):
    (b, g, r) = cv2.split(img)
    bH        = cv2.equalizeHist(b)
    gH        = cv2.equalizeHist(g)
    rH        = cv2.equalizeHist(r)
    result    = cv2.merge((bH, gH, rH))
    return result


def auto_padding(img, times=16):
    # img: numpy image with shape H*W*C
    h, w, _ = img.shape
    h1, w1  = (times - h % times) // 2, (times - w % times) // 2
    h2, w2  = (times - h % times) - h1, (times - w % times) - w1
    img     = cv2.copyMakeBorder(img, h1, h2, w1, w2, cv2.BORDER_REFLECT)
    return img, [h1, h2, w1, w2]


def format_measurements(meas):
    s_out = []
    for k, v in meas.items():
        v = f"{v:0.2f}" if isinstance(v, float) else v
        s_out.append(f"{k}: {v}")
    str_out = ", ".join(s_out)
    return str_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       type=str, default="data/test_data/")
    parser.add_argument("--weights",    type=str, default=mon.ZOO_DIR/"vision"/"enhance"/"llflow"/"llflow-lol-smallnet.pth")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default=mon.RUN_DIR/"predict/llflow")
    parser.add_argument("--opt",        type=str, default="./confs/LOL_smallNet.yml")
    parser.add_argument("--name", "-n", type=str, default="unpaired")
    args = parser.parse_args()
    
    args.data       = mon.Path(args.data)
    args.output_dir = mon.Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    console.log(f"Data: {args.data}")
    
    conf_path      = args.opt
    conf           = conf_path.split("/")[-1].replace(".yml", "")
    opt            = option.parse(conf_path, is_train=False)
    opt["gpu_ids"] = None
    opt            = option.dict_to_nonedict(opt)
    model          = create_model(opt)
    # model_path     = opt_get(opt, ["model_path"], None)
    model_path     = str(args.weights)
    model.load_network(load_path=model_path, network=model.netG)
    model.netG     = model.netG.cuda()

    # Measure efficiency score
    flops, params, avg_time = model.measure_efficiency_score(image_size=args.image_size)
    console.log(f"FLOPs  = {flops:.4f}")
    console.log(f"Params = {params:.4f}")
    console.log(f"Time   = {avg_time:.4f}")

    #
    with torch.no_grad():
        image_paths = list(args.data.rglob("*"))
        image_paths = [path for path in image_paths if path.is_image_file()]
        sum_time    = 0
        with mon.get_progress_bar() as pbar:
            for _, image_path in pbar.track(
                sequence    = enumerate(image_paths),
                total       = len(image_paths),
                description = f"[bright_yellow] Inferring"
            ):
                # print(image_path)
                lr        = imread(str(image_path))
                raw_shape = lr.shape
                
                start_time = time.time()
                lr, padding_params = auto_padding(lr)
                his = hiseq_color_cv2_img(lr)
                if opt.get("histeq_as_input", False):
                    lr = his
                lr_t = t(lr)
                if opt["datasets"]["train"].get("log_low", False):
                    lr_t = torch.log(torch.clamp(lr_t + 1e-3, min=1e-3))
                if opt.get("concat_histeq", False):
                    his  = t(his)
                    lr_t = torch.cat([lr_t, his], dim=1)
                heat = opt["heat"]
                
                with torch.cuda.amp.autocast():
                    sr_t = model.get_sr(lq=lr_t.cuda(), heat=None)
                sr = rgb(
                    torch.clamp(sr_t, 0, 1)[:, :,
                    padding_params[0]:sr_t.shape[2] - padding_params[1],
                    padding_params[2]:sr_t.shape[3] - padding_params[3]]
                )
                # assert raw_shape == sr.shape
                run_time  = (time.time() - start_time)
                sum_time += run_time
                
                result_path = args.output_dir / image_path.name
                imwrite(str(result_path), sr)
        avg_time = float(sum_time / len(image_paths))
        console.log(f"Average time: {avg_time}")


if __name__ == "__main__":
    main()
