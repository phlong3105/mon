#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

import logging
import os
import sys
import traceback

import mon
from saicinpainting.evaluation.data import InpaintingImageLabelDataset
from saicinpainting.evaluation.refinement import refine_predict
from saicinpainting.evaluation.utils import move_to_device

os.environ['OMP_NUM_THREADS']        = '1'
os.environ['OPENBLAS_NUM_THREADS']   = '1'
os.environ['MKL_NUM_THREADS']        = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS']    = '1'
os.environ['CUDA_LAUNCH_BLOCKING']   = '1'

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)


@hydra.main(config_path='../configs/prediction', config_name='default.yaml')
def main(predict_config: OmegaConf):
    try:
        register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        device = torch.device(predict_config.device)
        
        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        out_ext = predict_config.get('out_ext', '.jpg')
        
        checkpoint_path = os.path.join(predict_config.model.path, 'models', predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)

        #
        output_dir = predict_config.get("output_dir", None)
        if output_dir is not None:
            output_dir = mon.Path(output_dir)
        else:
            file       = mon.Path(predict_config.video_file)
            output_dir = file.parent / f"{file.stem}-inpainting"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        #
        dataset = InpaintingImageLabelDataset(
            image_dir=predict_config.image_dir,
            label_dir=predict_config.label_dir,
            **predict_config.dataset
        )
        
        #
        for img_i in tqdm.trange(len(dataset)):
            batch = default_collate([dataset[img_i]])
            if predict_config.get('refine', False):
                assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
                # image unpadding is taken care of in the refiner, so that output image
                # is same size as the input image
                cur_res = refine_predict(batch, model, **predict_config.refiner)
                cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()
            else:
                with torch.no_grad():
                    batch = move_to_device(batch, device)
                    batch['mask'] = (batch['mask'] > 0) * 1
                    batch = model(batch)                    
                    cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
                    unpad_to_size = batch.get('unpad_to_size', None)
                    if unpad_to_size is not None:
                        orig_height, orig_width = unpad_to_size
                        cur_res = cur_res[:orig_height, :orig_width]

            cur_res      = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res      = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            cur_res_name = output_dir / f"{img_i:06}{out_ext}"
            cv2.imwrite(str(cur_res_name), cur_res)
            if predict_config.get('verbose', False):
                cv2.imshow("Inpainting", cur_res)

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()
