import argparse
import glob
import logging
import os
import subprocess
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils
from PIL import Image

import utils
from dataset import ImageLowSemDataset, ImageLowSemDataset_Val
from model import *

'''
python train_loss.py --arc WithoutCalNet --batch_size 10
'''
# 该脚本命令行参数 可选项
parser = argparse.ArgumentParser("enlighten-anything")
parser.add_argument("--batch_size", type=int,   default=10,              help="batch size")
parser.add_argument("--cuda",       type=bool,  default=True,            help="Use CUDA to train model")
parser.add_argument("--gpu",        type=str,   default="0",             help="gpu device id")
parser.add_argument("--seed",       type=int,   default=2,               help="random seed")
parser.add_argument("--epochs",     type=int,   default=1000,            help="epochs")
parser.add_argument("--lr",         type=float, default=0.0003,          help="learning rate")
parser.add_argument("--stage",      type=int,   default=3,               help="epochs")
parser.add_argument("--save",       type=str,   default="exp/LOL-tgrs/", help="location of the data corpus")
parser.add_argument("--pretrain",   type=str,   default="weights/pretrained_SCI/difficult.pt",          help="pretrained weights directory")
parser.add_argument("--arch",       type=str,   choices=["WithCalNet", "WithoutCalNet"], required=True, help="with/without Calibrate Net")
parser.add_argument("--frozen",     type=str,   default=None, choices=["CalEnl", "Cal", "Enl"],         help="froze the original weights")
parser.add_argument("--train_dir",  type=str,   default="./data/LOL/train480/low", help="training data directory")
parser.add_argument("--val_dir",    type=str,   default="./data/LOL/test15/low",   help="training data directory")
parser.add_argument("--comment",    type=str,   default=None,                      help="comment")
args = parser.parse_args()

# 根据命令行参数进行设置
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

snapshot_dir = args.save + '/' + 'Train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(snapshot_dir, scripts_to_save=glob.glob('*.py'))
model_path = snapshot_dir + '/model_epochs/'
os.makedirs(model_path, exist_ok=True)
image_path = snapshot_dir + '/image_epochs/'
os.makedirs(image_path, exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(snapshot_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


def model_init(model):
    if args.pretrain is None:
        # model.enhance.in_conv.apply(model.weights_init)
        # model.enhance.conv.apply(model.weights_init)
        # model.enhance.out_conv.apply(model.weights_init)
        # model.calibrate.in_conv.apply(model.weights_init)
        # model.calibrate.convs.apply(model.weights_init)
        # model.calibrate.out_conv.apply(model.weights_init)
        # model.enhance.apply(model.weights_init)
        # model.calibrate.apply(model.weights_init)
        model.apply(model.weights_init)
    else:
        pretrained_dict = torch.load(args.pretrain)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        if args.frozen is not None:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.enhance.fusion.parameters() if 'Enl' in args.frozen else model.enhance.parameters():
            # for param in model.enhance.parameters():
                param.requires_grad = True


class GradCAM:
    
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self.hook_layers()

    def hook_layers(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, sem, depth, target_output):
        self.model.zero_grad()
        output = self.model(input_image, sem, depth)

        target = target_output  # 使用目标输出计算梯度
        target.backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


def visualize_cam_on_image(image, cam, save_path):
    image = image.cpu().numpy().transpose(1, 2, 0)
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))  # 确保 cam 尺寸与 image 一致
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_on_image = heatmap + np.float32(image)
    cam_on_image = cam_on_image / np.max(cam_on_image)

    plt.imshow(np.uint8(255 * cam_on_image))
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()


def main():
    logging.info("train file name = %s", os.path.split(__file__))
    logging.info("args = %s", args)
    
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    else:
        logging.info('gpu device = %s' % args.gpu)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    
    model = Network_woCalibrate()
    model_init(model)
    
    model = model.cuda()
    MB = utils.count_parameters_in_MB(model)
    logging.info("model size = %f", MB)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                  lr=args.lr*100, betas=(0.9, 0.999), weight_decay=3e-4)

    TrainDataset = ImageLowSemDataset(img_dir=args.train_dir, sem_dir=os.path.join(os.path.split(args.train_dir)[0], 'low_semantic'), depth_dir=os.path.join(os.path.split(args.train_dir)[0], 'low_depth'))
    ValDataset = ImageLowSemDataset_Val(img_dir=args.val_dir, sem_dir=os.path.join(os.path.split(args.val_dir)[0], 'low_semantic'), depth_dir=os.path.join(os.path.split(args.val_dir)[0], 'low_depth'))
    
    train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )
    
    val_queue = torch.utils.data.DataLoader(
        ValDataset, batch_size=1, shuffle=False, pin_memory=True
    )

    # 初始化 Grad-CAM
    grad_cam = GradCAM(model.enhance, model.enhance.out_conv[1])

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_idx, (in_, sem_, depth_, imgname_, semname_, depthname_) in enumerate(train_queue):
            in_ = in_.cuda()
            sem_ = sem_.cuda()
            depth_ = depth_.cuda()
            loss = model._loss(in_, sem_, depth_)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(loss.item())
            logging.info('train: epoch %3d: batch %3d: loss %f', epoch, batch_idx, loss)

        logging.info('train: epoch %3d: average_loss %f', epoch, np.average(losses))
        logging.info('----------validation')
        utils.save(model, os.path.join(model_path, f'weights_{epoch}.pt'))

        model.eval()
        image_path_epoch = os.path.join(image_path, f'epoch_{epoch}')
        os.makedirs(image_path_epoch, exist_ok=True)
        
        if args.arch == 'WithCalNet':
            with torch.no_grad():
                for batch_idx, (in_, sem_, depth_, imgname_, semname_, depthname_) in enumerate(val_queue):
                    in_ = in_.cuda()
                    sem_ = sem_.cuda()
                    depth_ = depth_.cuda()
                    image_name = os.path.splitext(imgname_[0])[0]
                    illu_list, ref_list, input_list, atten = model(in_, sem_, depth_)
                    u_name = f'{image_name}_{epoch}.png' 
                    print('validation processing {}'.format(u_name))
                    u_path = os.path.join(image_path_epoch, u_name)
                    save_images(ref_list[0], u_path)
        elif args.arch == 'WithoutCalNet':
            with torch.no_grad():
                for batch_idx, (in_, sem_, depth_, imgname_, semname_, depthname_) in enumerate(val_queue):
                    in_ = in_.cuda()
                    sem_ = sem_.cuda()
                    depth_ = depth_.cuda()
                    image_name = os.path.splitext(imgname_[0])[0]
                    i, r, d = model(in_, sem_, depth_)
                    u_name = f'{image_name}.png'
                    print('validation processing {}'.format(u_name))
                    u_path = os.path.join(image_path_epoch, u_name)
                    save_images(r, u_path)

        # 使用 Grad-CAM 生成并保存可视化
        cam_save_dir = os.path.join(image_path_epoch, 'grad_cam')
        os.makedirs(cam_save_dir, exist_ok=True)
        for batch_idx, (in_, sem_, depth_, imgname_, semname_, depthname_) in enumerate(val_queue):
            in_ = in_.cuda()
            sem_ = sem_.cuda()
            depth_ = depth_.cuda()

            target_output = model.enhance(in_, sem_, depth_)  # 获取增强网络的输出
            cam = grad_cam.generate_cam(in_, sem_, depth_, target_output[0].mean())  # 使用输出均值计算梯度
            cam_save_path = os.path.join(cam_save_dir, f'{os.path.splitext(imgname_[0])[0]}_grad_cam.png')
            visualize_cam_on_image(in_[0], cam, cam_save_path)
        
        process = subprocess.Popen(
            ['python', 'evaluate.py', '--test_dir', image_path_epoch, '--test_gt_dir', './data/LOL/test15/high'],
            stdout=subprocess.PIPE
        )
        output, error = process.communicate()
        if output:
            logging.info(output.decode('utf-8'))
        if error:
            logging.error(error.decode('utf-8'))


if __name__ == '__main__':
    main()
