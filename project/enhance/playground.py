#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit testing for in-develop components."""

from __future__ import annotations

import cv2
import numpy as np
import torch

import mon

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Function

def run():
    path       = mon.Path("./data/10.jpg")
    image      = cv2.imread(str(path))

    dcp1       = mon.get_dark_channel_prior(image, 15)
    dcp2       = mon.get_dark_channel_prior_02(image, 15)
    cv2.imshow("DCP 01", dcp1)
    cv2.imshow("DCP 02", dcp2)

    prior0     = mon.get_guided_brightness_enhancement_map_prior(image, 2, None)
    prior      = np.where(prior0 > 0.2, 255, 0).astype(np.uint8)
    dark       = cv2.bitwise_and(image, image, mask=prior)
    bright     = cv2.bitwise_and(image, image, mask=(255 - prior))
    contrast   = 3.5  # Contrast control (0-127)
    brightness = 5.0  # Brightness control (0-100)
    e_dark     = cv2.addWeighted(dark, contrast, dark, 0, brightness)
    enhance    = cv2.addWeighted(e_dark, 1, bright, 1, 0)

    mon.detect_blur_spot(image, verbose=True)
    mon.detect_bright_spot(image, verbose=True)

    contours, hierarchy = cv2.findContours(255 - prior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        c      = max(contours, key=cv2.contourArea)
        mask   = np.full(image.shape, 255, image.dtype)
        cv2.drawContours(mask, c, -1, 255, 3)
        image  = cv2.bitwise_and(image, mask, mask=prior)
        # image  = np.where(image > 0, image, 255).astype(np.uint8)
    
    cv2.imwrite(f"data/{path.stem}.png",         image)
    cv2.imwrite(f"data/{path.stem}-prior0.png",  255 * prior0[..., ::-1])
    cv2.imwrite(f"data/{path.stem}-prior.png",   prior[..., ::-1])
    cv2.imwrite(f"data/{path.stem}-bright.png",  bright)
    cv2.imwrite(f"data/{path.stem}-dark.png",    dark)
    cv2.imwrite(f"data/{path.stem}-edark.png",   e_dark)
    cv2.imwrite(f"data/{path.stem}-enhance.png", enhance)
    
    cv2.imshow("Image",   image)
    cv2.imshow("Prior 0", prior0)
    cv2.imshow("Prior",   prior)
    cv2.imshow("Bright",  bright)
    cv2.imshow("Dark",    dark)
    cv2.imshow("EDark",   e_dark)
    cv2.imshow("Enhance", enhance)
    cv2.waitKey(0)
    

def run_zsn2n():
    path   = mon.Path("./data/00691.png")
    image  = cv2.imread(str(path))
    device = torch.device("cuda:0")
    net = mon.ZSN2N(
        channels     = 3,
        num_channels = 64,
    ).to(device)
    denoised = net.fit_one(
        image,
        max_epochs = 5000,
        lr         = 0.0001,
    )
    denoised = mon.to_image_nparray(denoised, False, True)
    cv2.imshow("Image",    image)
    cv2.imshow("Denoised", denoised)
    cv2.waitKey(0)


def run_rrdnet():
    path    = mon.Path("./data/Madison.png")
    image   = cv2.imread(str(path))
    #
    device  = torch.device("cuda:0")
    net     = mon.RRDNet().to(device)
    zerodce = mon.ZeroDCE(
        num_iters = 4,
        weights   = _current_dir / "run/train/zerodce_sice_mix/weights/last.pt"
    ).to(device)
    zsn2n   = mon.ZSN2N(
        channels     = 3,
        num_channels = 64,
    ).to(device)
    #
    with torch.no_grad():
        input            = mon.to_image_tensor(image, False, True)
        input            = input.to(device)
        lightup, enhance = zerodce(input)
    #
    pred                  = net.fit_one(enhance)
    illumination          = pred[0]
    adjusted_illumination = pred[1]
    reflectance           = pred[2]
    noise                 = pred[3]
    relight               = pred[4]
    #
    relight2 = relight.clone().detach().requires_grad_(True)
    denoised = zsn2n.fit_one(relight2)
    #
    illumination          = torch.concat([illumination, illumination, illumination], dim=1)
    adjusted_illumination = torch.concat([adjusted_illumination, adjusted_illumination, adjusted_illumination], dim=1)
    #
    lightup = list(torch.split(lightup, 3, dim=1))
    for i, l in enumerate(lightup):
        lightup[i] = mon.to_image_nparray(lightup[i], False, True)
    enhance               = mon.to_image_nparray(enhance              , False, True)
    illumination          = mon.to_image_nparray(illumination         , False, True)
    adjusted_illumination = mon.to_image_nparray(adjusted_illumination, False, True)
    reflectance           = mon.to_image_nparray(reflectance          , False, True)
    noise                 = mon.to_image_nparray(noise                , False, True)
    relight               = mon.to_image_nparray(relight              , False, True)
    denoised              = mon.to_image_nparray(denoised             , False, True)
    cv2.imshow("Image"                , image)
    # cv2.imshow("Light-up 0"           , lightup[0])
    # cv2.imshow("Light-up 1"           , lightup[1])
    # cv2.imshow("Light-up 2"           , lightup[2])
    # cv2.imshow("Light-up 3"           , lightup[3])
    # cv2.imshow("Light-up 4"           , lightup[4])
    # cv2.imshow("Light-up 5"           , lightup[5])
    # cv2.imshow("Light-up 6"           , lightup[6])
    cv2.imshow("Light-up 7"           , lightup[7])
    cv2.imshow("Enhance"              , enhance)
    cv2.imshow("Illumination"         , illumination)
    cv2.imshow("Adjusted Illumination", adjusted_illumination)
    cv2.imshow("Reflectance"          , reflectance)
    cv2.imshow("Noise"                , noise)
    cv2.imshow("Relight"              , relight)
    cv2.imshow("Denoised"             , denoised)
    cv2.waitKey(0)


def run_cerdnet():
    path    = mon.Path("./data/02.jpg")
    image   = cv2.imread(str(path))
    #
    device  = torch.device("cuda:0")
    net     = mon.CERDNet().to(device)
    #
    pred             = net.fit_one(image)
    lightup          = pred[0]
    lightup_image    = pred[1]
    illumination     = pred[2]
    illumination_hat = pred[3]
    reflectance      = pred[4]
    noise            = pred[5]
    relight          = pred[6]
    denoised         = pred[7]
    #
    illumination     = torch.concat([illumination,     illumination,     illumination],     dim=1)
    illumination_hat = torch.concat([illumination_hat, illumination_hat, illumination_hat], dim=1)
    #
    lightup          = mon.to_image_nparray(lightup         , False, True)
    lightup_image    = mon.to_image_nparray(lightup_image   , False, True)
    illumination     = mon.to_image_nparray(illumination    , False, True)
    illumination_hat = mon.to_image_nparray(illumination_hat, False, True)
    reflectance      = mon.to_image_nparray(reflectance     , False, True)
    noise            = mon.to_image_nparray(noise           , False, True)
    relight          = mon.to_image_nparray(relight         , False, True)
    denoised         = mon.to_image_nparray(denoised        , False, True)
    cv2.imshow("Image"                , image)
    cv2.imshow("Light-up"             , lightup)
    cv2.imshow("Light-up Image"       , lightup_image)
    cv2.imshow("Illumination"         , illumination)
    cv2.imshow("Adjusted Illumination", illumination_hat)
    cv2.imshow("Reflectance"          , reflectance)
    cv2.imshow("Noise"                , noise)
    cv2.imshow("Relight"              , relight)
    cv2.imshow("Denoised"             , denoised)
    cv2.waitKey(0)
    
# endregion


# region Main

if __name__ == "__main__":
    run_cerdnet()

# endregion
