import math
import os

import cv2
import numpy as np
from bm3d import bm3d_rgb
from utils.experiment_funcs import get_experiment_noise


def get_newest_model(path):
    key_files = [(float(file.replace('model_', '').replace('.pt', '')), file) for file in os.listdir(path) if 'model_' in file]
    key_files = sorted(key_files, key=lambda x: x[0], reverse=True)
    paths = [os.path.join(path, basename[1]) for basename in key_files]
    for path in paths:
        if '.pt' in path:
            return path
    print('Could not find any model')
    return None


def make_im(pred, X_batch, target):
    b, im, c, w, h = X_batch.size()
    image = np.zeros((3, h, (w + 5) * (im + 2)))
    image[:, :, :w] = target[0, :, :, :].detach().cpu().numpy()
    image[:, :, (w+5):(2*w + 5)] = pred[0, :, :, :].detach().cpu().numpy()
    for i in range(im):
        image[:, :, (2 + i) * w + (2 + i) * 5:(3 + i) * w + (2 + i) * 5] = X_batch[0, i, :, :, :].detach().cpu().numpy()
    return image.astype(np.uint8)


def load_namespace(file):
    file = open(file)
    namespace = file.read()
    file.close()
    namespace_dict = {}
    namespace = namespace.split('Namespace')[1].replace("(", "").replace(")", "").replace(" ", "").replace('"','').replace("'","").split(',')
    for attr in namespace:
        att, val = attr.split("=")
        namespace_dict[att] = val
    return namespace_dict


## DENOISING #############################################
def denoiseCBM3D(I):
    # _,psd,_ = get_experiment_noise('g4', 0.005, 0, I.shape) # <--- GOLDEN
    # _,psd,_ = get_experiment_noise('g4', 0.001, 0, I.shape) # <--- GOLDEN
    _,psd,_ = get_experiment_noise('g4', 0.0005, 0, I.shape) # <--- GOLDEN
    Iqsef = bm3d_rgb(I, psd, 'refilter','YCbCr')
    # Iqsef = bm3d_rgb(I, psd, 'deb','YCbCr')
    return Iqsef


## metric ####################
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
    
    
def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)
