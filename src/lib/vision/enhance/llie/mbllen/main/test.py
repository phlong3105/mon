#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/Lvfeifan/MBLLEN
# pip install keras-flops

from __future__ import annotations

import argparse
import time

import cv2
import keras
import keras.backend as K
import numpy as np
from keras_flops import get_flops

import Network
import mon
import utls
from mon import ZOO_DIR, RUN_DIR

console = mon.console


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        type=str, default="./input", help="test image folder")
    parser.add_argument("--model",       type=str, default=ZOO_DIR / "vision/enhance/llie//mbllen/mbllen-syn_img_lowlight_withnoise", help="model name")
    parser.add_argument("--weights",     type=str, default="./models")
    parser.add_argument("--image-size",  type=int, default=512)
    parser.add_argument("--com",         type=int, default=1,  help="Output with/without origional image and mid-result")
    parser.add_argument("--highpercent", type=int, default=95, help="Should be in [85,100], linear amplification")
    parser.add_argument("--lowpercent",  type=int, default=5,  help="Should be in [0,15], rescale the range [p%,1] to [0, 1]")
    parser.add_argument("--gamma",       type=int, default=8,  help="Should be in [6,10], increase the saturability")
    parser.add_argument("--maxrange",    type=int, default=8,  help="Linear amplification range")
    parser.add_argument("--output-dir",  type=str, default=RUN_DIR / "predict/vision/enhance/llie/mbllen")
    args = parser.parse_args()
    
    args.data       = mon.Path(args.data)
    args.output_dir = mon.Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    console.log(f"Data: {args.data}")
    
    # Measure efficiency score
    model  = Network.build_mbllen((args.image_size, args.image_size, 3))
    flops  = get_flops(model, batch_size=1)
    params = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    console.log(f"FLOPs  = {flops:.4f}")
    console.log(f"Params = {params:.4f}")
    
    # Load model
    model_name = args.model
    mbllen     = Network.build_mbllen((None, None, 3))
    mbllen.load_weights(args.weights + "/" + model_name + ".h5")
    opt        = keras.optimizers.Adam(lr=2 * 1e-04, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    mbllen.compile(loss="mse", optimizer=opt)
    
    #
    flag        = args.com
    lowpercent  = args.lowpercent
    highpercent = args.highpercent
    maxrange    = args.maxrange / 10.0
    hsvgamma    = args.gamma    / 10.0
    image_paths = list(args.data.rglob("*"))
    image_paths = [path for path in image_paths if path.is_image_file()]
    sum_time    = 0
    with mon.get_progress_bar() as pbar:
        for _, image_path in pbar.track(
            sequence    = enumerate(image_paths),
            total       = len(image_paths),
            description = f"[bright_yellow] Inferring"
        ):
            # console.log(image_path)
            img_A       = utls.imread_color(str(image_path))
            img_A       = img_A[np.newaxis, :]
            start_time  = time.time()
            out_pred    = mbllen.predict(img_A)
            run_time    = (time.time() - start_time)
    
            fake_B      = out_pred[0, :, :, :3]
            fake_B_o    = fake_B
            gray_fake_B = fake_B[:, :, 0] * 0.299 + fake_B[:, :, 1] * 0.587 + fake_B[:, :, 1] * 0.114
            percent_max = sum(sum(gray_fake_B >= maxrange)) / sum(sum(gray_fake_B <= 1.0))
            max_value   = np.percentile(gray_fake_B[:], highpercent)
            if percent_max < (100 - highpercent) / 100.:
                scale  = maxrange / max_value
                fake_B = fake_B * scale
                fake_B = np.minimum(fake_B, 1.0)
            gray_fake_B = fake_B[:,:,0]*0.299 + fake_B[:,:,1]*0.587 + fake_B[:,:,1]*0.114
            sub_value  = np.percentile(gray_fake_B[:], lowpercent)
            fake_B     = (fake_B - sub_value)*(1./(1-sub_value))
            imgHSV     = cv2.cvtColor(fake_B, cv2.COLOR_RGB2HSV)
            H, S, V    = cv2.split(imgHSV)
            S          = np.power(S, hsvgamma)
            imgHSV     = cv2.merge([H, S, V])
            fake_B     = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2RGB)
            fake_B     = np.minimum(fake_B, 1.0)
            if flag:
                outputs = np.concatenate([img_A[0,:,:,:], fake_B_o, fake_B], axis=1)
            else:
                outputs = fake_B
            outputs = np.minimum(outputs, 1.0)
            outputs = np.maximum(outputs, 0.0)
        
            result_path = args.output_dir / image_path.name
            utls.imwrite(str(result_path), outputs)
            sum_time      += run_time
    avg_time = float(sum_time / len(image_paths))
    console.log(f"Average time: {avg_time}")
    

if __name__ == "__main__":
    test()
