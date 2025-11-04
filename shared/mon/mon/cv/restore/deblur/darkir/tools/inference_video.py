'''
This script works as an inference video recorder.
'''
import os
import numpy as np
import cv2 as cv
from options.options import parse
import argparse

parser = argparse.ArgumentParser(description="Script for video inference")
parser.add_argument('-p', '--config', type=str, default='./options/inference_video/Baseline.yml', help = 'Config file of video inference')
args = parser.parse_args()


path_options = args.config
opt = parse(path_options)
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

# PyTorch library
import torch
import torch.optim
import torch.multiprocessing as mp
from tqdm import tqdm
from torchvision.transforms import Resize

from data.dataset_reader.datapipeline import *
from archs import *
from utils.test_utils import *

from ptflops import get_model_complexity_info

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

#define some transforms
pil_to_tensor = transforms.ToTensor()
tensor_to_pil = transforms.ToPILImage()

resize = opt['Resize']

def array_to_tensor(frame):
    '''
    Transform from numpy array [H,W,C] to torch tensor [B,C,H,W]
    '''
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    tensor_frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() 
    return tensor_frame 

def tensor_to_array(tensor):
    '''
    Transform from torch tensor [B,C,H,W] to numpy array [H,W,C].
    '''
    array = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    frame = (array * 255).astype(np.uint8)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # flip red and blue channels
    return frame

def normalize_tensor(tensor):
    '''
    Normalize tensor to the range [0,1]
    '''
    max_value = torch.max(tensor)
    min_value = torch.min(tensor)
    output = (tensor - min_value)/(max_value)
    return output

def save_tensor(tensor, path):
    '''
    Save tensor as PIL image.
    '''
    tensor = tensor.squeeze(0)
    # tensor = normalize_tensor(tensor)
    print(tensor.shape, tensor.dtype, torch.max(tensor), torch.min(tensor))
    img = tensor_to_pil(tensor)
    img.save(path)

def pad_tensor(tensor, multiple = 8):
    '''
    Pad the tensor to be multiple of some number (its size).
    '''
    multiple = multiple
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value = 0)
    return tensor

def load_model(model, path_weights):
    '''
    Load the weights of the model.
    '''
    map_location = 'cpu'
    checkpoints = torch.load(path_weights, map_location=map_location, weights_only=False)
   
    weights = checkpoints['params']
    weights = {'module.' + key: value for key, value in weights.items()}

    macs, params = get_model_complexity_info(model, (3, 256, 256), print_per_layer_stat=False, verbose=False)
    print('Complexity information of the model: ', macs, params)
    model.load_state_dict(weights)
    print('Loaded weights correctly')
    return model

def apply_model(model, tensor, resize = False):
    '''
    Apply the inference over each specific frame. If resize = True, resizes before inference.
    '''
    _, _, H, W = tensor.shape
    if resize:
        new_size = [720, 1080]
        downsample = Resize(new_size)
    else:
        downsample = torch.nn.Identity()
    tensor = downsample(tensor)
    tensor = pad_tensor(tensor)

    with torch.no_grad():
        output = model(tensor, side_loss=False)
    if resize:
        upsample = Resize((H, W))
    else: upsample = torch.nn.Identity()

    output = upsample(output)
    output = torch.clamp(output, 0., 1.)
    output = output[:,:, :H, :W]
    return output

def inference_video(rank, world_size):
    '''
    Inferences the video frames and constructs a new video. The result video is a composition of the original and the process ones.
    '''
    import argparse

    parser = argparse.ArgumentParser(description="Video inference script")
    parser.add_argument('-i', '--inp_path', type=str, default=None, 
                    help="File path to video")
    args = parser.parse_args()

    setup(rank, world_size=world_size) # setup the torch.distributor

    # Open the video file
    cap = cv.VideoCapture(args.inp_path)
    
    # Get video properties
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')

    output_path = os.path.join('./videos/results', os.path.basename(args.inp_path))

    out = cv.VideoWriter(output_path, fourcc, fps, (int(frame_width * 2), frame_height))

    # Instantiate model and load weights
    model, _, _ = create_model(opt['network'], rank=rank)
    model = load_model(model, path_weights = opt['save']['path'])

    model.eval()

    if rank==0:
        pbar = tqdm(total = int(cap.get(cv.CAP_PROP_FRAME_COUNT)))

    while cap.isOpened():
        ret, frame = cap.read()
        old_frame = np.copy(frame)
        if not ret: break

        tensor = array_to_tensor(frame)
        tensor = normalize_tensor(tensor)

        output = apply_model(model, tensor, resize = resize)

        frame = tensor_to_array(output)
        combined = np.hstack((old_frame, frame))

        out.write(combined)
        if rank==0: pbar.update(1)

    cap.release()
    out.release()
    print('Finished inference!')
    if rank == 0:
        pbar.close()   
    cleanup()


def main():
    world_size = 1
    print('Used GPUS:', world_size)
    mp.spawn(inference_video, args =(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()