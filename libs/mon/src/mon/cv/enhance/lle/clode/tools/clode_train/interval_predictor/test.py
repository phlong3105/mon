from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm
import sys
import os

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
from data import *
from model import *
from network.conv_node import NODE
from utils import plot_regression_results
from misc import calculate_psnr, calculate_ssim

RUN_NAME = sys.argv[1]
args = OmegaConf.load(Path(__file__).parent / 'model' / RUN_NAME / 'config.yaml')
args.merge_with_cli()

models = {
    'v1': Regressor_v1,
    'v2': Regressor_v2,
    'v3': Regressor_v3,
    'v4': Regressor_v4,
    'v5': Regressor_v5,
    'v6': Regressor_v6,
    'v7': Regressor_v7,
    'v8': Regressor_v8,
    'v9': Regressor_v9,
    'v10': Regressor_v10,
}

datasets = {
    'v1': [TrainDataset_v1, EvalDataset_v1],
    'v2': [TrainDataset_v1, EvalDataset_v1],
    'v3': [TrainDataset_v1, EvalDataset_v1],
    'v4': [TrainDataset_v2, EvalDataset_v2],
    'v5': [TrainDataset_v2, EvalDataset_v2],
    'v6': [TrainDataset_v2, EvalDataset_v2],
    'v7': [TrainDataset_v3, EvalDataset_v3],
    'v8': [TrainDataset_v3, EvalDataset_v3],
    'v9': [TrainDataset_v2, EvalDataset_v2],
    'v10': [TrainDataset_v2, EvalDataset_v2],
}

CLODE_model_path = Path(__file__).parent / '..' / 'pth'
CLODE_model_file = 'sice.pth' if args.data == 'SICE' else 'lowlight.pth'
model_path = Path(__file__).parent / 'model' / RUN_NAME
model_file = next(model_path.glob('*.pth'))

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = NODE(device, (3, 400, 600), 32, augment_dim=0, time_dependent=True, adjoint=True)
model.eval()
model.to(device)
model.load_state_dict(torch.load(CLODE_model_path / CLODE_model_file, weights_only=True), strict=False)

regressor = models[args.model](hidden_dim=args.hidden_dim).to(device)
regressor.load_state_dict(torch.load(model_file, weights_only=True, map_location=device))
regressor.eval()


# Plot regression performance on [train, val] dataset
train_data = 'LOL' if args.data in ['LOL', 'LSRW'] else 'SICE'
train_set = datasets[args.model][0](train_data, split='train', val_size=0.2, device=device)
val_set = datasets[args.model][0](train_data, split='val', val_size=0.2, device=device)

with torch.no_grad():
    y_pred = regressor(train_set.X).cpu().numpy()
    y_true = train_set.y.cpu().numpy()
    
    plot_regression_results(
        y_true, 
        y_pred, 
        'Regression Results (for Train set): True vs Predicted T', 
        model_path / 'train_results.png'
    )

    y_pred = regressor(val_set.X).cpu().numpy()
    y_true = val_set.y.cpu().numpy()
    
    plot_regression_results(
        y_true, 
        y_pred, 
        'Regression Results (for Test set): True vs Predicted T', 
        model_path / 'test_results.png'
    )
    
# Caculate PSNR for test dataset
test_dataset = datasets[args.model][1](args.data, device)
psnrs = []
ssims = []

for datum in tqdm(test_dataset):
    x, lq, gt = datum
    
    with torch.no_grad():
        # (Ours) Regression for T
        pred_T = regressor(x).item()
        # CLODE
        integration_time = torch.tensor([0, pred_T]).float().cuda()
        pred = model(lq, integration_time, inference=True)['output'][0]
        
        _psnr = calculate_psnr(pred, gt)
        _ssim = calculate_ssim(pred, gt)
    psnrs.append(_psnr)
    ssims.append(_ssim)

print(f"PSNR: {np.mean(psnrs):.4f} dB")
print(f"SSIM: {np.mean(ssims):.4f}")