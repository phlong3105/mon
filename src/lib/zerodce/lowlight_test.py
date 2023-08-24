#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.optim
import torchvision
from PIL import Image

import model
import mon


def predict(image_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    data_lowlight = Image.open(image_path)
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    DCE_net  = model.enhance_net_nopool().cuda()
    DCE_net.load_state_dict(torch.load(config.weights))
    start    = time.time()
    _, enhanced_image, _ = DCE_net(data_lowlight)
    end_time = (time.time() - start)
    print(end_time)
    '''
    image_path  = image_path.replace("test_data", "result")
    result_path = image_path
    if not os.path.exists(image_path.replace("/" + image_path.split("/")[-1], '')):
        os.makedirs(image_path.replace("/" + image_path.split("/")[-1], ''))
    torchvision.utils.save_image(enhanced_image, result_path)
    '''
    return enhanced_image, end_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       type=str, default="data/test_data/")
    parser.add_argument("--weights",    type=str, default="weights/Epoch99.pth")
    parser.add_argument("--output-dir", type=str, default="predict/")
    config = parser.parse_args()
    
    config.output_dir = mon.Path(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test_images
    with torch.no_grad():
        config.data = mon.Path(config.data)
        image_paths = list(config.data.rglob("*"))
        image_paths = [path for path in image_paths if path.is_image_file()]
        sum_time    = 0
        num_images  = 0
        for image_path in image_paths:
            print(image_path)
            enhanced_image, end_time = predict(image_path)
            image_path   = mon.Path(image_path)
            result_path  = config.output_dir / image_path.name
            torchvision.utils.save_image(enhanced_image, str(result_path))
            sum_time    += end_time
            num_images  += 1
        avg_time = float(sum_time / num_images)
        print(f"Average time: {avg_time}")
