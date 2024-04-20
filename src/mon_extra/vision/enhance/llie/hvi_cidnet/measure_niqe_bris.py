import argparse
import glob

import imquality.brisque as brisque
from PIL import Image
from tqdm import tqdm

from loss.niqe_utils import *

eval_parser = argparse.ArgumentParser(description="Eval")
eval_parser.add_argument("--dicm", action="store_true", help="output DICM dataset")
eval_parser.add_argument("--lime", action="store_true", help="output LIME dataset")
eval_parser.add_argument("--mef",  action="store_true", help="output MEF dataset")
eval_parser.add_argument("--npe",  action="store_true", help="output NPE dataset")
eval_parser.add_argument("--vv",   action="store_true", help="output VV dataset")
ep = eval_parser.parse_args()


def metrics(im_dir):
    avg_niqe = 0
    n = 0
    avg_brisque = 0
        
    for item in tqdm(sorted(glob.glob(im_dir))):
        n += 1
        
        im1 = Image.open(item).convert("RGB")
        score_brisque = brisque.score(im1) 
        im1 = np.array(im1)
        score_niqe = calculate_niqe(im1)
        
        
        avg_brisque += score_brisque
        avg_niqe += score_niqe

        torch.cuda.empty_cache()
    
    avg_brisque = avg_brisque / n
    avg_niqe = avg_niqe / n
    return avg_niqe, avg_brisque


if __name__ == "__main__":
    if ep.dicm:
        im_dir = "./output/DICM/*.jpg"
    elif ep.lime:
        im_dir = "./output/LIME/*.bmp"
    elif ep.mef:
        im_dir = "./output/MEF/*.png"
    elif ep.npe:
        im_dir = "./output/NPE/*.jpg"
    elif ep.vv:
        im_dir = "./output/VV/*.jpg"
        
    avg_niqe, avg_brisque = metrics(im_dir)
    print(avg_niqe)
    print(avg_brisque)
