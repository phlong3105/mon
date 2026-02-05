import torch
import numpy as np

from tqdm import tqdm
import os
from pathlib import Path
import wandb

from losses import *
from misc import *

MODEL_SAVE_PATH = Path(__file__).parent / "checkpoints"
IMAGE_PATH      = Path("/home/soom/data/LOL/eval15")
filenames       = sorted(os.listdir(IMAGE_PATH / 'low'))


def exp_value(t):
    t = np.clip(t, 0, 3)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    norm = (sigmoid(t - 1.5) - sigmoid(-1.5)) / (sigmoid(1.5) - sigmoid(-1.5))
    return 0.6 * norm


class Trainer():
    '''
    1. NODE-SR처럼 CLODE의 final state T를 랜덤하게 바꾸면서 학습하는 Trainer 클래스
    '''
    
    def __init__(self, model, optimizer, device, config):
        self.model = model
        self.optimizer = optimizer
        self.device = device

        # Loss functions
        self.L_tv = L_TV().to(device)
        self.L_color = L_color().to(device)
        self.L_spa = L_spa().to(device)
        self.L_exp = L_exp(16).to(device)
        self.L_exp_val = L_exp_value(16).to(device)
        # Loss weights
        self.exp_w, self.col_w, self.spa_w, self.tv_w = 10, 20, 1, 200

        self.train_epoch_loss = 0
        self.best_loss = np.inf
        self.best_psnr = 0

        # Log
        wandb.init(project="clode_fsp", name=config.run, config=config)
        self.ckpt_dir = MODEL_SAVE_PATH / config.run
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def train(self, data_loader):
        self.train_epoch_loss = self._train_epoch(data_loader)
        
        # Save model
        if self.train_epoch_loss < self.best_loss:
            self.best_loss = self.train_epoch_loss
            torch.save(self.model.state_dict(), self.ckpt_dir / "best.pth")
    
    def validation(self):        
        _psnr, _ssim = 0, 0
        self.model.eval()

        # In training phese, fix the final state to 3 for validation
        eval_time = torch.tensor([0, 3]).float().to(self.device)

        for filename in filenames:
            with torch.no_grad():
                lq = image_tensor(IMAGE_PATH / 'low' / filename).unsqueeze(0).to(self.device)
                gt = image_tensor(IMAGE_PATH / 'high' / filename).unsqueeze(0).to(self.device)
                pred = self.model(lq, eval_time, inference=True)['output']
                
                _psnr += calculate_psnr(pred, gt)
                _ssim += calculate_ssim(pred, gt)

        wandb.log({
            "val/psnr": _psnr / len(filenames),
            "val/ssim": _ssim / len(filenames),
        })

        # Save model
        if _psnr > self.best_psnr:
            self.best_psnr = _psnr
            torch.save(self.model.state_dict(), self.ckpt_dir / "best_psnr.pth")
    
    def _train_epoch(self, data_loader):
        epoch_losses = np.zeros(5)  # spa, exp, col, param, noise
        self.model.train()
        
        for x_batch, _ in tqdm(data_loader):
            x_batch = x_batch.to(self.device)

            t_end = np.random.normal(loc=3.0, scale=1.0)
            eval_time = torch.tensor([0, t_end]).float().to(self.device)    
       
            pred = self.model(x_batch, eval_time)
            pred_img = pred['output']
            A_map = pred['curve_map']
            noise_map = pred['noise_map']
                                    
            loss_param =  self.tv_w * torch.mean(A_map)
            loss_col = self.col_w * torch.mean(self.L_color(pred_img))
            loss_spa = self.spa_w * torch.mean(self.L_spa(pred_img, x_batch))
            loss_exp = self.exp_w * torch.mean(self.L_exp_val(pred_img, exp_value(t_end)))
            loss_noise = torch.mean(noise_map)
            
            curve_adjust_loss = loss_spa + loss_exp + loss_col + loss_param + loss_noise
            self.optimizer.zero_grad()
            curve_adjust_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()
            
            epoch_losses += [loss_spa.item(), loss_exp.item(), loss_col.item(), loss_param.item(), loss_noise.item()]
        epoch_losses /= len(data_loader)
        wandb.log({
            "train/loss_spa": epoch_losses[0],
            "train/loss_exp": epoch_losses[1],
            "train/loss_col": epoch_losses[2],
            "train/loss_param": epoch_losses[3],
            "train/loss_noise": epoch_losses[4],
            "train/total_loss": np.sum(epoch_losses),
        })
                
        return np.sum(epoch_losses)
