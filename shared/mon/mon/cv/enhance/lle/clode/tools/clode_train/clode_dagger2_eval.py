import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import pyiqa

from network.conv_node import NODE
from misc import *

os.environ["CUDA_VISIBLE_DEVICES"] = '5'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
# CLODE model
model = NODE(device, (3, 128, 128), 32, augment_dim=0, time_dependent=True, adjoint=True)
model.eval()
model.to(device)

MODEL_NAME = "default_noise"
# model_path = "/home/soom/CLODE/pth/lowlight.pth"
model_path = Path(__file__).parent / "fsp" / "checkpoints" / MODEL_NAME / "best_psnr.pth"
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)

data = np.load(f'data/psnr_ssim_{MODEL_NAME}_lol_eval.npy', allow_pickle=True)
t_val = np.linspace(2, 5, 1000)
max_psnr_indices = np.argmax(data[:, :, 2], axis=1)

# Image dataset
def load_image(filepath):
    img = cv2.imread(str(filepath))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (np.asarray(img)/255.0)
    img = torch.from_numpy(img).float()
    img = img.permute(2, 0, 1).unsqueeze(0)
    
    return img.to(device)

IMAGE_PATH = Path('/home/soom/data/LOL/eval15')
filenames = sorted(os.listdir(IMAGE_PATH / 'low'))
num_images = len(filenames)

# Non-reference metrics
niqe_metric = pyiqa.create_metric('niqe', device=device)
brisque_metric = pyiqa.create_metric('brisque', device=device)
pi_metric = pyiqa.create_metric('pi', device=device)
entropy_metric = pyiqa.create_metric('entropy', device=device)

total_psnr = 0
total_ssim = 0
total_niqe = 0
total_brisque = 0
total_pi = 0
total_entropy = 0

for i in tqdm(range(num_images)):
    lq_img = load_image(IMAGE_PATH / 'low' / filenames[i])
    gt_img = load_image(IMAGE_PATH / 'high' / filenames[i])
    with torch.no_grad():
        integration_time = torch.tensor([0, t_val[max_psnr_indices[i]]]).float().to(device)
        pred = model(lq_img, integration_time, inference=True)['output']
        
    total_psnr += calculate_psnr(pred[0], gt_img[0])
    total_ssim += calculate_ssim(pred[0], gt_img[0])
    total_niqe += niqe_metric(pred).item()
    total_brisque += brisque_metric(pred).item()
    total_pi += pi_metric(pred).item()
    total_entropy += entropy_metric(pred).item()

print("PSNR:", total_psnr / num_images)
print("SSIM:", total_ssim / num_images)
print("NIQE:", total_niqe / num_images)
print("BRISQUE:", total_brisque / num_images)
print("PI:", total_pi / num_images)
print("Entropy:", total_entropy / num_images)