import os
import time

import torch
from thop import clever_format, profile

from model import lle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

width  = 640
height = 480


def compute_FLOPs_and_model_size(model, width, height):
    input = torch.randn(1, 3, width, height).cuda()
    macs, params = profile(model, inputs=(input,), verbose=False)
    return macs, params


@torch.no_grad()
def compute_fps_and_inference_time(model, shape, epoch=100, warmup=10, device=None):
    total_time = 0.0

    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model.eval()  # Switch to evaluation mode

    # Warm-up iterations
    for _ in range(warmup):
        data = torch.randn(shape).to(device)
        model(data)
    
    # Actual timing iterations
    for _ in range(epoch):
        data = torch.randn(shape).to(device)

        start = time.time()
        outputs = model(data)
        torch.cuda.synchronize()  # Ensure CUDA has finished all tasks
        end = time.time()

        total_time += (end - start)

    avg_inference_time = total_time / epoch
    fps = epoch / total_time

    return fps, avg_inference_time


def test_model_flops(width, height):
    model = lle.MobileIES(channels=12) 
    model.cuda()

    FLOPs, params = compute_FLOPs_and_model_size(model, width, height)

    model_size = params * 4.0 / 1024 / 1024
    flops, params = clever_format([FLOPs, params], "%.3f")

    print('Number of parameters: {}'.format(params))
    print('Size of model: {:.2f} MB'.format(model_size))
    print('Computational complexity: {} FLOPs'.format(flops))


def test_fps_and_inference_time(width, height):
    model = lle.MobileIES(channels=12)  
    model.cuda()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    fps, avg_inference_time = compute_fps_and_inference_time(model, (1, 3, width, height), device=device)
    print('device: {} - fps: {:.3f}, average inference time per frame: {:.6f} seconds'.format(device.type, fps, avg_inference_time))

if __name__ == "__main__":
    test_model_flops(width, height)
    test_fps_and_inference_time(width, height)
