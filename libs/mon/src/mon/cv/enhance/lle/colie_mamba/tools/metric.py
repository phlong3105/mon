#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import math
import os

import cv2
import numpy as np


def main():
    # Configurations

    # GT - Ground-truth;
    # Gen: Generated / Restored / Recovered images
    #folder_GT = 'D:\Program Files (x86)\PycharmProjects\InvDN\codes\data\CBSD68\original_png'
    #folder_Gen = 'D:\Program Files (x86)\PycharmProjects\InvDN\experiments\Test_InvDN\CBSD68_15_test'
    folder_GT = '/home/lijin/PycharmProjects/colie/input/dataset/LOLdataset/eval15/high'
    folder_Gen = '/home/lijin/PycharmProjects/colie/output/dataset/LOLdataset/eval15/mamba_high_v2'

    crop_border = 4
    suffix = ''  # suffix for Gen images
    test_Y = False  # True: test Y channel only; False: test RGB channels

    PSNR_all = []
    SSIM_all = []
    img_list = sorted(glob.glob(folder_GT + '/*'))

    if test_Y:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    for i, img_path in enumerate(img_list):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        im_GT = cv2.imread(img_path) / 255.
        im_Gen = cv2.imread(os.path.join(folder_Gen, f"{base_name}{suffix}.png")) / 255.

        # # 调整生成图像尺寸以匹配GT
        # if im_GT.shape[:2] != im_Gen.shape[:2]:
        #     im_Gen = cv2.resize(im_Gen, (im_GT.shape[1], im_GT.shape[0]), interpolation=cv2.INTER_LINEAR)
        #     print(f"⚠️ 已调整 {base_name} 尺寸至 {im_GT.shape[:2]}")

        if test_Y and im_GT.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
            im_GT_in = bgr2ycbcr(im_GT)
            im_Gen_in = bgr2ycbcr(im_Gen)
        else:
            im_GT_in = im_GT
            im_Gen_in = im_Gen

        # crop borders
        if im_GT_in.ndim == 3:
            cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border, :]
            cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border, :]
        elif im_GT_in.ndim == 2:
            cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border]
            cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border]
        else:
            raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT_in.ndim))

        # calculate PSNR and SSIM
        PSNR = calculate_psnr(cropped_GT * 255, cropped_Gen * 255)

        SSIM = calculate_ssim(cropped_GT * 255, cropped_Gen * 255)
        print('{:3d} - {:25}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}'.format(
            i + 1, base_name, PSNR, SSIM))
        PSNR_all.append(PSNR)
        SSIM_all.append(SSIM)
    print('Average: PSNR: {:.6f} dB, SSIM: {:.6f}'.format(
        sum(PSNR_all) / len(PSNR_all),
        sum(SSIM_all) / len(SSIM_all)))


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def bgr2ycbcr(image, only_y: bool = True):
    """Same as matlab rgb2ycbcr.
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    dtype = image.dtype
    image.astype(np.float32)
   
    if dtype != np.uint8:
        image *= 255.0
    
    # Convert
    if only_y:
        rlt = np.dot(image, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(image,[
            [ 24.966,   112.0, -18.214],
            [128.553, -74.203, -93.786],
            [ 65.481, -37.797,   112.0]
        ]) / 255.0 + [16, 128, 128]
    
    if dtype == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.0
        
    return rlt.astype(dtype)


if __name__ == "__main__":
    main()
