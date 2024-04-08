import argparse
import os
from collections import OrderedDict

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from averageMeter import AverageMeter
from dataset import dataset
from loss import Loss
from model import NestedUNet


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--lr', default=0.00001, type=float)

    config = parser.parse_args()

    return config


def train(train_loader, model, criterion, optimizer):  
    avg_meters = AverageMeter()

    model.train()

    pbar = tqdm(total=len(train_loader))   
    for input, target in train_loader:
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()  
        optimizer.step() 

        avg_meters.update(loss.item(), input.size(0))
        
        pbar.set_postfix(loss=avg_meters.avg)
        pbar.update(1)
    pbar.close()
    return avg_meters.avg


def validate(val_loader, model, criterion):
    avg_meters = AverageMeter()

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target in val_loader:
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)

            avg_meters.update(loss.item(), input.size(0))

            pbar.set_postfix(loss=avg_meters.avg)
            pbar.update(1)
        pbar.close()

    return avg_meters.avg


def main():
    config = vars(parse_args())

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    writer = SummaryWriter(log_dir='runs/loss')

    criterion = Loss()
    criterion = criterion.cuda()

    cudnn.benchmark = True

    print("=> creating model")
    model = NestedUNet()
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    train_ori_fd = "/data/hope/VE-LOL-L-Syn/train_lowlight"
    train_ucc_fd = "/data/hope/VE-LOL-L-Syn/train"               
    train_ori_dirs = [os.path.join(train_ori_fd, f) for f in os.listdir(train_ori_fd)]
    train_ucc_dirs = [os.path.join(train_ucc_fd, f) for f in os.listdir(train_ucc_fd)]
    train_dataset = dataset(train_ori_dirs, train_ucc_dirs)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)

    val_ori_fd = "/data/hope/VE-LOL-L-Syn/test_lowlight"
    val_ucc_fd = "/data/hope/VE-LOL-L-Syn/test"
    val_ori_dirs = [os.path.join(val_ori_fd, f) for f in os.listdir(val_ori_fd)]
    val_ucc_dirs = [os.path.join(val_ucc_fd, f) for f in os.listdir(val_ucc_fd)]
    val_dataset = dataset(val_ori_dirs, val_ucc_dirs)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('val_loss', []),
    ])

    best_loss = 1000

    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))
        train_log = train(train_loader, model, criterion, optimizer)
        val_log = validate(val_loader, model, criterion)

        scheduler.step() 
        
        print('loss %.4f  - val_loss %.4f' % (train_log, val_log))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log)
        log['val_loss'].append(val_log)

        pd.DataFrame(log).to_csv('runs/log.csv')

        writer.add_scalars("loss",{"train":train_log,"test":val_log},epoch)
        writer.close()

        if val_log < best_loss or epoch>290:
            torch.save(model.state_dict(), 'checkpoint/model_%s.pth' %epoch)
            best_loss = val_log
            print("=> saved best model")
            trigger = 0

        torch.cuda.empty_cache()  


if __name__ == '__main__':
    main()
