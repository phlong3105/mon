import os
import torch
import numpy as np
from PIL import Image

import argparse
from torchvision import transforms
import datetime
import math
from model import NestedUNet



def main(checkpoint, imgs_path, result_path):

    ori_dirs = []
    for image in os.listdir(imgs_path):
        ori_dirs.append(os.path.join(imgs_path, image))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NestedUNet()

    model = model.cuda()
    model.load_state_dict(torch.load(checkpoint))

    model.eval()

    testtransform = transforms.Compose([
                transforms.Resize([384, 384]),                  
                transforms.ToTensor(),
            ])
    unloader = transforms.ToPILImage()

    starttime = datetime.datetime.now()
    for imgdir in ori_dirs:
        img_name = (imgdir.split('/')[-1]).split('.')[0]
        img = Image.open(imgdir)
        inp = testtransform(img).unsqueeze(0)
        inp = inp.to(device)
        out = model(inp)

        corrected = unloader(out.cpu().squeeze(0))
        corrected.save("./output"+'/{}corrected.png'.format(img_name))
    endtime = datetime.datetime.now()
    print(endtime-starttime)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./checkpoint/model_39.pth')
    parser.add_argument('--images', default='./testimage')
    parser.add_argument('--result', default='./output/')
    args = parser.parse_args()
    checkpoint = args.checkpoint
    imgs = args.images
    result_path = args.result
    main(checkpoint=checkpoint, imgs_path=imgs, result_path=result_path)