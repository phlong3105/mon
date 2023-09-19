from __future__ import print_function

import argparse
import os
from glob import glob

import tensorflow as tf

import mon
from model import lowlight_enhance
from utils import *

console = mon.console


def lowlight_train(args, lowlight_enhance):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    lr = args.start_lr * np.ones([args.epoch])
    lr[20:] = lr[0] / 10.0

    train_low_data = []
    train_high_data = []

    train_low_data_names = glob('./data/our485/low/*.png') + glob('./data/syn/low/*.png')
    train_low_data_names.sort()
    train_high_data_names = glob('./data/our485/high/*.png') + glob('./data/syn/high/*.png')
    train_high_data_names.sort()
    assert len(train_low_data_names) == len(train_high_data_names)
    print('[*] Number of training data: %d' % len(train_low_data_names))

    for idx in range(len(train_low_data_names)):
        low_im = load_images(train_low_data_names[idx])
        train_low_data.append(low_im)
        high_im = load_images(train_high_data_names[idx])
        train_high_data.append(high_im)

    eval_low_data = []
    eval_high_data = []

    eval_low_data_name = glob('./data/eval/low/*.*')

    for idx in range(len(eval_low_data_name)):
        eval_low_im = load_images(eval_low_data_name[idx])
        eval_low_data.append(eval_low_im)

    lowlight_enhance.train(train_low_data, train_high_data, eval_low_data, batch_size=args.batch_size, patch_size=args.patch_size, epoch=args.epoch, lr=lr, sample_dir=args.sample_dir, ckpt_dir=os.path.join(args.ckpt_dir, 'Decom'), eval_every_epoch=args.eval_every_epoch, train_phase="Decom")
    lowlight_enhance.train(train_low_data, train_high_data, eval_low_data, batch_size=args.batch_size, patch_size=args.patch_size, epoch=args.epoch, lr=lr, sample_dir=args.sample_dir, ckpt_dir=os.path.join(args.ckpt_dir, 'Relight'), eval_every_epoch=args.eval_every_epoch, train_phase="Relight")


def lowlight_test(args, model):
    if args.data is None:
        console.log("[!] please provide --data")
        exit(0)
    
    #
    image_paths = list(args.data.rglob("*"))
    image_paths = [path for path in image_paths if path.is_image_file()]
    sum_time    = 0
    with mon.get_progress_bar() as pbar:
        for _, image_path in pbar.track(
            sequence    = enumerate(image_paths),
            total       = len(image_paths),
            description = f"[bright_yellow] Inferring"
        ):
            # console.log(image_path)
            image = load_images(str(image_path))
            
    
    test_low_data_name = glob(os.path.join(args.test_dir) + "/*.*")
    test_low_data      = []
    test_high_data     = []
    for idx in range(len(test_low_data_name)):
        test_low_im = load_images(test_low_data_name[idx])
        test_low_data.append(test_low_im)
    
    model.test(test_low_data, test_high_data, test_low_data_name, save_dir=args.save_dir, decom_flag=args.decom)


def main(_):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data",             type=str,   default="./sample", help="Directory for evaluating outputs")
    parser.add_argument("--phase",            type=str,   default="train",    help="Train or test")
    parser.add_argument("--epochs",           type=int,   default=100,        help="Number of total epoches")
    parser.add_argument("--batch-size",       type=int,   default=16,         help="Number of samples in one batch")
    parser.add_argument("--patch-size",       type=int,   default=48,         help="Patch size")
    parser.add_argument("--lr",               type=float, default=0.001,      help="Initial learning rate for adam")
    parser.add_argument("--eval-every-epoch", type=int,   default=20,         help="Evaluating and saving checkpoints every #  epoch")
    parser.add_argument("--decom",            type=int,   default=0,          help="Decom flag, 0 for enhanced results only and 1 for decomposition results.")
    parser.add_argument("--use-gpu",          type=int,   default=1,          help="GPU flag, 1 for GPU and 0 for CPU")
    parser.add_argument("--gpu-idx",          type=str,   default="0",        help="GPU idx")
    parser.add_argument("--gpu-mem",          type=float, default=0.5,        help="0 to 1, gpu memory usage")
    parser.add_argument("--checkpoints-dir",  type=str,   default="./model",  help="Directory for checkpoints")
    parser.add_argument("--output-dir",       type=str,   default=mon.RUN_DIR/"predict/retinexnet")
    args = parser.parse_args()
    
    args.data            = mon.Path(args.data)
    args.checkpoints_dir = mon.Path(args.checkpoints_dir)
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir      = mon.Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    console.log(f"Data: {args.data}")
    
    #
    if args.use_gpu:
        console.log("[*] GPU\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = lowlight_enhance(sess)
            if args.phase == "train":
                lowlight_train(args, model)
            elif args.phase == "test":
                lowlight_test(args, model)
            else:
                console.log("[!] Unknown phase")
                exit(0)
    else:
        console.log("[*] CPU\n")
        with tf.Session() as sess:
            model = lowlight_enhance(sess)
            if args.phase == "train":
                lowlight_train(args, model)
            elif args.phase == "test":
                lowlight_test(args, model)
            else:
                console.log("[!] Unknown phase")
                exit(0)


if __name__ == "__main__":
    tf.app.run()
