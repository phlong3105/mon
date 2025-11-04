import os
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d

from network.conv_node import NODE
from misc import *
from losses import *

MODEL_NAME = "default_noise"

os.environ["CUDA_VISIBLE_DEVICES"] = '5'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
# CLODE model
model = NODE(device, (3, 128, 128), 32, augment_dim=0, time_dependent=True, adjoint=True)
model.eval()
model.to(device)

# model_path = "/home/soom/CLODE/pth/lowlight.pth"
model_path = Path(__file__).parent / "fsp" / "checkpoints" / MODEL_NAME / "best_psnr.pth"
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)

# - saving trajectory
trajectory = []
ode_func = model.odefunc.forward
def get_trajectory(t, x):
    t_val = t.item() if isinstance(t, torch.Tensor) else float(t)
    t_denoised = torch.clamp(x[:, :3, :, :] - model.odefunc.denoise(x[:, :3, :, :]), 0, 1)
    t_denoised = t_denoised.detach().cpu()
    trajectory.append((t_val, t_denoised))
    
    return ode_func(t, x)

model.odefunc.forward = get_trajectory

# Image dataset
def load_image(filepath):
    img = cv2.imread(str(filepath))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (np.asarray(img)/255.0)
    img = torch.from_numpy(img).float()
    img = img.permute(2, 0, 1).unsqueeze(0)
    
    return img.to(device)

IMAGE_PATH = Path("/home/soom/data/LOL/eval15")

filenames = sorted(os.listdir(IMAGE_PATH / 'low'))
data = np.zeros((len(filenames), 1000, 4))

"""
Main:
    PSNR trajectory 뽑고 interpolate해서 저장

"""
x_dense = np.linspace(2, 5, 1000)

for i in tqdm(range(len(filenames))):
    lq_img = load_image(IMAGE_PATH / 'low' / filenames[i])
    gt_img = load_image(IMAGE_PATH / 'high' / filenames[i])
    trajectory = []

    T = 5
    with torch.no_grad():
        integration_time = torch.tensor([0, T]).float().to(device)
        pred = model(lq_img, integration_time, inference=True)['output']
    trajectory.sort(key=lambda x: x[0])
    t_vals, denoised_imgs = zip(*trajectory)
    t_vals, denoised_imgs = np.array(t_vals), np.array(denoised_imgs)
    
    # Remove duplicates
    _, unique_indices = np.unique(t_vals, return_index=True)
    t_vals, denoised_imgs = t_vals[unique_indices], denoised_imgs[unique_indices]
    
    datum = np.zeros((len(t_vals), 3))

    lum = denoised_imgs.squeeze(1).max(axis=1)
    lum_mean = lum.mean(axis=(1,2))
    datum[:, 0] = lum_mean
    
    for _t, pred in enumerate(denoised_imgs):
        pred = torch.from_numpy(pred).float()

        metrics = [
            calculate_psnr(pred[0], gt_img[0]),
            calculate_ssim(pred[0], gt_img[0]),
        ]
        datum[_t, 1:] = metrics

    # Interpolate scores
    interp_func = interp1d(t_vals, datum, kind='cubic', axis=0)
    y_dense = interp_func(x_dense)
    data[i, :, 0] = x_dense
    data[i, :, 1:] = y_dense

np.save(f'data/psnr_ssim_{MODEL_NAME}_lol_eval.npy', data)

# psnr 기준으로 가장 높은 인덱스들
max_psnr_indices = np.argmax(data[:, :, 2], axis=1)
max_psnrs = data[np.arange(len(data)), max_psnr_indices, 2]
max_ssims = data[np.arange(len(data)), max_psnr_indices, 3]
print("psnr: ", max_psnrs.mean())
print("ssim: ", max_ssims.mean())