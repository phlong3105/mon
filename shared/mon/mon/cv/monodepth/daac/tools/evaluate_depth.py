import argparse
import logging
import os
import pprint

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from dataset.semi_depth_synthetic_uncertainty_unsupervised import SemiDataset
from matplotlib import cm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from depth_anything.dpt import DepthAnything_AC
from util.dist_helper import setup_distributed
from util.utils import init_log


class RunningAverage:

    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


def denormalize(x):
    """Reverses the imagenet normalization applied to the input.

    Args:
        x (torch.Tensor - shape(N,3,H,W)): input tensor

    Returns:
        torch.Tensor - shape(N,3,H,W): Denormalized input
    """
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    return x * std + mean


class RunningAverageDict:
    """A dictionary of running averages."""

    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if new_dict is None:
            return

        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        if self._dict is None:
            return {}
        return {key: value.get_value() for key, value in self._dict.items()}


def compute_errors(gt, pred):
    """Compute metrics for 'pred' compared to 'gt'

    Args:
        gt (numpy.ndarray): Ground truth values
        pred (numpy.ndarray): Predicted values

        gt.shape should be equal to pred.shape

    Returns:
        dict: Dictionary containing the following metrics:
            'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
            'a2': Delta2 accuracy: Fraction of pixels that are within a scale factor of 1.25^2
            'a3': Delta3 accuracy: Fraction of pixels that are within a scale factor of 1.25^3
            'abs_rel': Absolute relative error
            'rmse': Root mean squared error
            'log_10': Absolute log10 error
            'sq_rel': Squared relative error
            'rmse_log': Root mean squared error on the log scale
            'silog': Scale invariant log error
    """
    eps = 0
    thresh = np.maximum((gt / (pred+eps)), (pred / (gt+eps)))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / (gt+eps))
    sq_rel = np.mean(((gt - pred) ** 2) / (gt))

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)


def compute_metrics(gt, pred, mask, dataset='kitti', min_depth_eval=0.1, max_depth_eval=80):
    """
    Compute depth estimation metrics for predicted depth maps.
    
    This function processes predictions and ground truth depth maps by:
    1. Applying depth value clamping and handling invalid values
    2. Applying dataset-specific cropping (Garg crop or Eigen crop)
    3. Using valid mask to filter pixels for evaluation
    4. Computing standard depth estimation metrics
    
    Applies cropping and masking as necessary or specified via arguments.
    Refer to compute_errors for more details on individual metrics.
    
    Args:
        gt (torch.Tensor): Ground truth depth map
        pred (torch.Tensor): Predicted depth map
        mask (torch.Tensor): Valid pixel mask
        dataset (str): Dataset name for applying specific crops (default: 'kitti')
        min_depth_eval (float): Minimum depth for evaluation (default: 0.1)
        max_depth_eval (float): Maximum depth for evaluation (default: 80)
        
    Returns:
        dict: Dictionary containing computed depth metrics
    """

    assert gt.shape[-2:] == pred.shape[-2:]

    pred = pred.squeeze().cpu().numpy()
    pred[pred < min_depth_eval] = min_depth_eval
    pred[pred > max_depth_eval] = max_depth_eval
    pred[np.isinf(pred)] = max_depth_eval
    pred[np.isnan(pred)] = min_depth_eval
   

    gt_depth = gt.squeeze().cpu().numpy()
    valid_mask = mask.squeeze().cpu().numpy()
    
    valid_mask = np.logical_and(valid_mask, mask.squeeze().cpu().numpy())
   
    garg_crop = True
    eigen_crop = False
    if garg_crop or eigen_crop:
        gt_height, gt_width = gt_depth.shape
        eval_mask = np.zeros(valid_mask.shape)

        if garg_crop:
            eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                      int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

        elif eigen_crop:
            if dataset == 'kitti':
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                          int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            else:
                # assert gt_depth.shape == (480, 640), "Error: Eigen crop is currently only valid for (480, 640) images"
                eval_mask[45:471, 41:601] = 1
        else:
            eval_mask = np.ones(valid_mask.shape)
    return compute_errors(gt_depth[valid_mask], pred[valid_mask])


class Disparity2Depth:

    def __init__(self, threshold=1.25, depth_cap=10):
        self.__threshold = threshold
        self.__depth_cap = depth_cap

    def compute_scale_and_shift(self, prediction, target, mask):
        """
        Compute optimal scale and shift parameters to align prediction with target.
        
        This method solves a least squares problem to find the best linear transformation
        (scale and shift) that aligns the predicted disparity with the target disparity.
        
        Args:
            prediction (torch.Tensor): Predicted disparity values
            target (torch.Tensor): Target disparity values  
            mask (torch.Tensor): Valid pixel mask (1 for valid, 0 for invalid)
            
        Returns:
            tuple: (scale, shift) - optimal scale and shift parameters
        """
        # system matrix: A = [[a_00, a_01], [a_10, a_11]]
        a_00 = torch.sum(mask * prediction * prediction, (1, 2))
        a_01 = torch.sum(mask * prediction, (1, 2))
        a_11 = torch.sum(mask, (1, 2))

        # right hand side: b = [b_0, b_1]
        b_0 = torch.sum(mask * prediction * target, (1, 2))
        b_1 = torch.sum(mask * target, (1, 2))

        # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
        x_0 = torch.zeros_like(b_0)
        x_1 = torch.zeros_like(b_1)

        det = a_00 * a_11 - a_01 * a_01
        # A needs to be a positive definite matrix.
        valid = det > 0

        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

        return x_0, x_1

    def __call__(self, prediction, target, mask):
        """
        Transform predicted disparity to aligned depth using optimal scale and shift.
        
        This method converts disparity predictions to depth by:
        1. Converting target depth to disparity (1/depth)
        2. Computing optimal scale and shift to align prediction with target disparity
        3. Applying the transformation and converting back to depth
        4. Applying depth cap to ensure reasonable depth values
        
        Args:
            prediction (torch.Tensor): Predicted disparity values
            target (torch.Tensor): Target depth values
            mask (torch.Tensor): Valid pixel mask (1 for valid, 0 for invalid)
            
        Returns:
            torch.Tensor: Aligned depth predictions
        """
        # transform predicted disparity to aligned depth
        target_disparity = torch.zeros_like(target)
        target_disparity[mask == 1] = 1.0 / target[mask == 1]
        scale, shift = self.compute_scale_and_shift(prediction, target_disparity, mask)
        prediction_aligned = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
        disparity_cap = 1.0 / self.__depth_cap
        prediction_aligned[prediction_aligned < disparity_cap] = disparity_cap

        prediciton_depth = 1.0 / prediction_aligned
        return prediciton_depth


def evaluate(model, loader, cfg, dataset='kitti', depth_min=0.1,depth_max=80, use_tqdm=True,vis=False):
    """
    Evaluate depth estimation model on a dataset using standard depth metrics.
    
    This function performs comprehensive evaluation of a depth estimation model by:
    1. Running inference on all samples in the data loader
    2. Aligning predictions with ground truth using disparity-to-depth conversion
    3. Computing standard depth estimation metrics (RMSE, absolute relative error, etc.)
    4. Optionally saving visualization results
    5. Aggregating results across multiple GPUs in distributed setting
    
    Args:
        model: Depth estimation model to evaluate
        loader: DataLoader containing evaluation samples
        cfg: Configuration dictionary
        dataset (str): Dataset name for evaluation (default: 'kitti')
        depth_min (float): Minimum depth value for evaluation (default: 0.1)
        depth_max (float): Maximum depth value for evaluation (default: 80)
        use_tqdm (bool): Whether to show progress bar (default: True)
        vis (bool): Whether to save visualization results (default: False)
        
    Returns:
        dict: Dictionary containing evaluation metrics (a1, a2, a3, abs_rel, rmse, etc.)
    """
    return_dict = {}
    model.eval()
    metrics  = RunningAverageDict()
    metric   = Disparity2Depth(depth_cap=depth_max)
    rank     = int(os.environ['LOCAL_RANK'])
    gpu_num  = dist.get_world_size()

    use_tqdm = use_tqdm
    if rank == 0:
        if use_tqdm:
            tbar = tqdm(total=len(loader))

    with torch.no_grad():
        for sample in loader:
            dist.barrier()
            if not sample['has_valid_depth']:
                continue
            img     = sample['image'].cuda()
            depth   = sample['depth'].cuda()
            mask    = sample['mask'].cuda()
            h, w    = sample['ori_size']
            h_d,h_w = depth.shape[-2:]
            
            prediction = model(img)
            prediction = prediction['out']
            prediction = F.interpolate(prediction[None], (h_d, h_w), mode='bilinear', align_corners=False)[:, 0,:,:]
            prediction = metric(prediction, depth, mask)

            spectral_cmap = cm.get_cmap('Spectral_r')
            
            if vis:
                mask_np  = mask.squeeze().cpu().numpy()
                depth_np = depth.squeeze().cpu().numpy()

                masked_depth = depth_np.copy()
                masked_depth[mask_np == 0] = 0 

                if masked_depth.max() > masked_depth.min():
                    depth_vis = (masked_depth - masked_depth.min()) / (masked_depth.max() - masked_depth.min()) * 255
                else:
                    depth_vis = masked_depth * 0  
                depth_vis = depth_vis.astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
            
                pred_np = prediction.squeeze().cpu().numpy()
                pred_norm = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min())

                pred_color = (spectral_cmap(pred_norm) * 255).astype(np.uint8)
                pred_vis = cv2.cvtColor(pred_color, cv2.COLOR_RGBA2BGR)

                img_vis = denormalize(img)  
                img_vis = img_vis.squeeze().cpu().numpy().transpose(1,2,0)
                img_vis = (img_vis * 255).astype(np.uint8)  
                img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)  

                mask_vis = mask.squeeze().cpu().numpy() * 255
                mask_vis = mask_vis.astype(np.uint8)

                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                save_dir = 'vis_results'
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, f'{timestamp}_img.png'),   img_vis)
                cv2.imwrite(os.path.join(save_dir, f'{timestamp}_depth.png'), depth_vis)
                cv2.imwrite(os.path.join(save_dir, f'{timestamp}_pred.png'),  pred_vis)
                cv2.imwrite(os.path.join(save_dir, f'{timestamp}_mask.png'),  mask_vis)

            res = compute_metrics(depth, prediction, mask, dataset, depth_min, depth_max)
            res_all = [dict() for i in range(gpu_num)]
            for k in res.keys():
                v = res[k]
                v = torch.from_numpy(np.array(v)).unsqueeze(0).cuda()
                v_gather = torch.zeros([gpu_num, 1], dtype=v.dtype).cuda()
                dist.all_gather_into_tensor(v_gather, v)
                v = v_gather
                for gpu in range(gpu_num):
                    res_all[gpu][k] = v[gpu][0].cpu().numpy()
            for res_now in res_all:
                metrics.update(res_now)

            temp = {k: round(v,6) for k, v in metrics.get_value().items()}

            if rank == 0:
                if use_tqdm:
                    tbar.set_description(' Metrics: {}'.format(temp))
                    tbar.update(1)

    if rank == 0:
        if use_tqdm:
            tbar.close()
    metrics = {k: v for k, v in metrics.get_value().items()}
    return metrics


def evaluate_DA2K(model, loader, depth_max=80, use_tqdm=True):
    """
    Evaluate depth estimation model on DA2K dataset using relative depth accuracy.
    
    DA2K (Depth Anything 2K) evaluation focuses on relative depth ordering accuracy
    by comparing depth values at two specific points and determining which is closer.
    This evaluation method tests the model's ability to understand relative depth
    relationships rather than absolute depth values.
    
    Args:
        model: Depth estimation model to evaluate
        loader: DataLoader containing DA2K evaluation samples with point pairs
        depth_max (float): Maximum depth value for capping (default: 80)
        use_tqdm (bool): Whether to show progress bar (default: True)
        
    Returns:
        dict: Dictionary containing accuracy metric for relative depth ordering
    """
    model.eval()
    
    metric = Disparity2Depth(depth_cap=depth_max)
    rank = int(os.environ['LOCAL_RANK'])
    gpu_num = dist.get_world_size()
    use_tqdm = use_tqdm
    if rank == 0:
        if use_tqdm:
            tbar = tqdm(total=len(loader))
    with torch.no_grad():
        correct_num = 0
        all_num = 0
        for sample in loader:
            dist.barrier()
            if sample['has_valid_depth'] == False:
                continue
            img = sample['image'].cuda()
            h_d, h_w = sample['ori_size']
            points = sample['points']

            prediction = model(img)
            prediction = prediction['out']
            prediction = F.interpolate(prediction[None], (h_d, h_w), mode='bilinear', align_corners=False)[:, 0,:,:]
            prediction = 1/(prediction+1e-6)

            batch_correct = 0
            batch_total = 0
            for b in range(len(points)):
                
                h1, w1, h2, w2, point_type = points[b]
        
                depth1 = prediction[0,int(h1), int(w1)].item()
                depth2 = prediction[0,int(h2), int(w2)].item()

                pred_closer = "point1" if depth1 < depth2 else "point2"
                if str(pred_closer) == str(point_type[0]):
                    batch_correct += 1
                    
                batch_total += 1
            
            correct_tensor = torch.tensor([batch_correct], dtype=torch.float32).cuda()
            total_tensor = torch.tensor([batch_total], dtype=torch.float32).cuda()
            
            correct_gather = torch.zeros([gpu_num, 1], dtype=torch.float32).cuda()
            total_gather = torch.zeros([gpu_num, 1], dtype=torch.float32).cuda()
            
            dist.all_gather_into_tensor(correct_gather, correct_tensor.unsqueeze(0))
            dist.all_gather_into_tensor(total_gather, total_tensor.unsqueeze(0))
            
            for i in range(gpu_num):
                correct_num += correct_gather[i][0].item()
                all_num += total_gather[i][0].item()
            
            accuracy = correct_num / max(all_num, 1)
            temp = {'Accuracy': round(accuracy, 5)}

            if rank == 0:
                if use_tqdm:
                    tbar.set_description(' Metrics: {}'.format(temp))
                    tbar.update(1)

    if rank == 0:
        if use_tqdm:
            tbar.close()
    
    metrics = {'Accuracy': round(correct_num / max(all_num, 1), 3)}

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--port', default=None, type=int)
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['kitti', 'nyu', 'sintel', 'DIODE', 'ETH3D', 'robotcar', 'nuscene', 
                               'foggy', 'cloudy', 'rainy', 'kitti_c_fog', 'kitti_c_snow', 
                               'kitti_c_dark', 'kitti_c_motion', 'kitti_c_gaussian', 'DA2K',
                               'DA2K_dark', 'DA2K_snow', 'DA2K_fog', 'DA2K_blur'],
                       help='Dataset to evaluate')
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    logger = init_log('global', logging.INFO)
    if logger is not None:
        logger.propagate = False
        logger.info('{}\n'.format(pprint.pformat(cfg)))

    rank, word_size = setup_distributed(port=args.port)
    local_rank      = int(os.environ["LOCAL_RANK"])

    model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024],'version':'v2'},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768],'version':'v2' },
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384],'version':'v2'}
        }
    model = DepthAnything_AC(model_configs['vits'])   
    model.load_state_dict(torch.load(f'./checkpoints/depth_anything_AC_vits.pth'),strict=False)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    eval_mode     = 'original'
    dataset_name  = args.dataset
    
    dataset_key   = f'dataset_val_{dataset_name}' if dataset_name != 'kitti' else 'dataset_val'
    data_root_key =   f'data_root_{dataset_name}' if dataset_name != 'kitti' else 'data_root_val'
    crop_size_key =   f'crop_size_{dataset_name}' if dataset_name != 'kitti' else 'crop_size'
    
    if dataset_key not in cfg or data_root_key not in cfg:
        if logger is not None:
            logger.error(f'Dataset configuration for {dataset_name} not found in config file')
            logger.error(f'Missing keys: {dataset_key} or {data_root_key}')
        return

    valset     = SemiDataset(cfg[dataset_key], cfg[data_root_key], 'val', cfg[crop_size_key], argu_mode=cfg['argu_mode'], cfg=cfg)
    valsampler = DistributedSampler(valset)
    valloader  = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=False, sampler=valsampler)
    
    if dataset_name.startswith('DA2K'):
        result = evaluate_DA2K(model, valloader)
        if logger is not None:
            logger.info(f'***** Evaluation {dataset_name} {eval_mode} ***** >>>> Metrics: {result} \n')
    else:
        depth_min_key = f'depth_min_val_{dataset_name}' if dataset_name != 'kitti' else 'depth_min_val'
        depth_cap_key = f'depth_cap_val_{dataset_name}' if dataset_name != 'kitti' else 'depth_cap_val'
        depth_min = cfg.get(depth_min_key, 0.1)
        depth_max = cfg.get(depth_cap_key,  80)
        result    = evaluate(model, valloader, cfg, dataset=dataset_name, depth_min=depth_min, depth_max=depth_max)
        if logger is not None:
            logger.info(f'***** Evaluation {dataset_name} {eval_mode} ***** >>>> Metrics: {result} \n')


if __name__ == '__main__':
    main()
