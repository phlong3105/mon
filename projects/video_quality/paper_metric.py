import os.path
import subprocess

import matplotlib.pyplot as plt
import numpy
import scipy.misc

import niqe
import psnr
import reco
import ssim
import vifp
import mse
from one.constants import DATA_DIR
from one.constants import PROJECTS_DIR
from one.core import console
from one.core import is_image_file
from one.core import progress_bar

pred_dir   = PROJECTS_DIR / "train" / "infer" / "sice" / ""
target_dir = DATA_DIR / "sice" / "part2_900x1200_low" / "high"
pred_files = list(pred_dir.rglob("*/*"))

# Metrics
quality_values = []
size_values    = []

mse_values     = []
niqe_values    = []
psnr_values    = []
reco_values    = []
ssim_values    = []
vifp_values    = []

with progress_bar() as pbar:
    for i, pred_file in pbar.track(
        enumerate(pred_files),
        total=len(pred_files),
        description=f"[bright_yellow] Processing"
    ):
        if not is_image_file(pred_file):
            continue
        name        = str(pred_file.parent.name)
        target_file = target_dir / f"{name}.jpg"
        pred        = scipy.misc.imread(str(pred_file),   flatten=True).astype(numpy.float32)
        target      = scipy.misc.imread(str(target_file), flatten=True).astype(numpy.float32)
        file_size   = os.path.getsize(pred_file)
        
        quality_values.append(i)
        mse_values.append(mse.mse(target, pred))
        niqe_values.append(niqe.niqe(pred / 255.0))
        psnr_values.append(psnr.psnr(target, pred))
        reco_values.append(reco.reco(target / 255.0, pred / 255.0))
        size_values.append(int(file_size / 1024))
        ssim_values.append(ssim.ssim_exact(target / 255.0, pred / 255.0))
        vifp_values.append(vifp.vifp_mscale(target, pred))

mse  = sum(mse_values)  / len(mse_values)
niqe = sum(niqe_values) / len(niqe_values)
psnr = sum(psnr_values) / len(psnr_values)
reco = sum(reco_values) / len(reco_values)
ssim = sum(ssim_values) / len(ssim_values)
vifp = sum(vifp_values) / len(vifp_values)

console.log(f"mse : {mse:.9f}")
console.log(f"niqe: {niqe:.9f}")
console.log(f"psnr: {psnr:.9f}")
console.log(f"reco: {reco:.9f}")
console.log(f"ssim: {ssim:.9f}")
console.log(f"vifp: {vifp:.9f}")

"""
plt.figure(figsize=(8, 8))
plt.plot(quality_values, vifp_values, label='VIFP')
plt.plot(quality_values, ssim_values, label='SSIM')
# plt.plot(niqe_values, label='NIQE')
plt.plot(quality_values, reco_values, label='RECO')
plt.plot(quality_values, numpy.asarray(psnr_values)/100.0, label='PSNR/100')
plt.legend(loc='lower right')
plt.xlabel('JPEG Quality')
plt.ylabel('Metric')
plt.savefig('jpg_demo_quality.png')

plt.figure(figsize=(8, 8))
plt.plot(size_values, vifp_values, label='VIFP')
plt.plot(size_values, ssim_values, label='SSIM')
# plt.plot(size_values, label='NIQE')
plt.plot(size_values, reco_values, label='RECO')
plt.plot(size_values, numpy.asarray(psnr_values)/100.0, label='PSNR/100')
plt.legend(loc='lower right')
plt.xlabel('JPEG File Size, KB')
plt.ylabel('Metric')
plt.savefig('jpg_demo_size.png')
"""
