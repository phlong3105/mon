import os
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomCrop, RandomHorizontalFlip, Resize, ToTensor


class dataset_single(data.Dataset):
    
    def __init__(self, opts, setname, input_dim):
        self.dataroot = opts.dataroot
        images = os.listdir(os.path.join(self.dataroot, opts.phase + setname))
        self.img = [os.path.join(self.dataroot, opts.phase + setname, x) for x in images]
        self.size = len(self.img)
        self.input_dim = input_dim
        
        # setup image transformation
        transforms = [Resize(opts.resize_size, Image.BICUBIC)]
        # transforms = [ToTensor()]
        # transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
        # transforms.append(CenterCrop(opts.crop_size))
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)
        print('%s: %d images' % (setname, self.size))
        return
    
    def __getitem__(self, index):
        data = self.load_img(self.img[index], self.input_dim)
        return data
    
    def load_img(self, img_name, input_dim):
        img = Image.open(img_name).convert('RGB')
        img = self.transforms(img)
        if input_dim == 1:
            img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = img.unsqueeze(0)
        return img
    
    def __len__(self):
        return self.size


class dataset_unpair(data.Dataset):
    
    def __init__(self, opts):
        self.dataroot = opts.dataroot
        
        # A
        images_A = os.listdir(os.path.join(self.dataroot, opts.phase + 'A'))
        self.A   = [os.path.join(self.dataroot, opts.phase + 'A', x) for x in images_A]
        
        # B
        images_B = os.listdir(os.path.join(self.dataroot, opts.phase + 'B'))
        self.B   = [os.path.join(self.dataroot, opts.phase + 'B', x) for x in images_B]
        
        self.A_size = len(self.A)
        self.B_size = len(self.B)
        self.dataset_size = max(self.A_size, self.B_size)
        self.input_dim_A = opts.input_dim_a
        self.input_dim_B = opts.input_dim_b
        
        # setup image transformation
        transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
        if opts.phase == 'train':
            transforms.append(RandomCrop(opts.crop_size))
        else:
            transforms.append(CenterCrop(opts.crop_size))
        if not opts.no_flip:
            transforms.append(RandomHorizontalFlip())
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)
        print('A: %d, B: %d images' % (self.A_size, self.B_size))
        return
    
    def __getitem__(self, index):
        if self.dataset_size == self.A_size:
            data_A = self.load_img(self.A[index], self.input_dim_A)
            data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)], self.input_dim_B)
        else:
            data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)], self.input_dim_A)
            data_B = self.load_img(self.B[index], self.input_dim_B)
        return data_A, data_B
    
    def load_img(self, img_name, input_dim):
        img = Image.open(img_name).convert('RGB')
        img = self.transforms(img)
        if input_dim == 1:
            img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = img.unsqueeze(0)
        return img
    
    def __len__(self):
        return self.dataset_size


class dataset_pair(data.Dataset):
    
    def __init__(self, opts):
        self.opt = opts
        self.dataroot = opts.dataroot
        # A
        images_A = os.listdir(os.path.join(self.dataroot, opts.phase + 'A'))
        self.A = [os.path.join(self.dataroot, opts.phase + 'A', x) for x in images_A]
        
        # B
        images_B = os.listdir(os.path.join(self.dataroot, opts.phase + 'B'))
        self.B = [os.path.join(self.dataroot, opts.phase + 'B', x) for x in images_B]
        
        # self.A_paths = sorted(self.A_paths)
        # self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A)
        self.B_size = len(self.B)
        
        transform_list = [Resize(opts.resize_size, Image.BICUBIC)]
        # transform_list = []
        transform_list.append(ToTensor())
        transform_list.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        # transform_list = [transforms.ToTensor()]
        
        self.transform = transforms.Compose(transform_list)
        # self.transform = get_transform(opt)
    
    def __getitem__(self, index):
        A_path = self.A[index]
        B_path = self.B[index]
        
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        
        A_img = self.transform(A_img)
        B_img = self.transform(B_img)
        
        w = A_img.size(2)
        h = A_img.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.resize_size - 1))
        h_offset = random.randint(0, max(0, h - self.opt.resize_size - 1))
        
        # A_img = A_img[:, h_offset:h_offset + self.opt.resize_size,
        #         w_offset:w_offset + self.opt.resize_size]
        # B_img = B_img[:, h_offset:h_offset + self.opt.resize_size,
        #         w_offset:w_offset + self.opt.resize_size]
        
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A_img = A_img.index_select(2, idx)
            B_img = B_img.index_select(2, idx)
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A_img.size(1) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A_img = A_img.index_select(1, idx)
            B_img = B_img.index_select(1, idx)
        
        return A_img, B_img
    
    def __len__(self):
        return self.A_size


class dataset_unaligned(data.Dataset):
    
    def __init__(self, opts):
        self.dataroot = opts.dataroot
        
        # A
        images_A = os.listdir(os.path.join(self.dataroot, opts.phase + 'A'))
        self.A = [os.path.join(self.dataroot, opts.phase + 'A', x) for x in images_A]
        
        # B
        images_B = os.listdir(os.path.join(self.dataroot, opts.phase + 'B'))
        self.B = [os.path.join(self.dataroot, opts.phase + 'B', x) for x in images_B]
        
        self.A_size = len(self.A)
        self.B_size = len(self.B)
        self.dataset_size = max(self.A_size, self.B_size)
        self.input_dim_A = opts.input_dim_a
        self.input_dim_B = opts.input_dim_b
        
        # setup image transformation
        transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
        if opts.phase == 'train':
            transforms.append(RandomCrop(opts.crop_size))
        else:
            transforms.append(CenterCrop(opts.crop_size))
        if not opts.no_flip:
            transforms.append(RandomHorizontalFlip())
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)
        print('A: %d, B: %d images' % (self.A_size, self.B_size))
        return
    
    def __getitem__(self, index):
        if self.dataset_size == self.A_size:
            data_A = self.load_img(self.A[index], self.input_dim_A)
            data_B = self.load_img(self.B[index], self.input_dim_B)
        else:
            data_A = self.load_img(self.A[index], self.input_dim_A)
            data_B = self.load_img(self.B[index], self.input_dim_B)
        return data_A, data_B
    
    def load_img(self, img_name, input_dim):
        img = Image.open(img_name).convert('RGB')
        img = self.transforms(img)
        if input_dim == 1:
            img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = img.unsqueeze(0)
        return img
    
    def __len__(self):
        return self.dataset_size

# class dataset_unaligned(data.Dataset):
#   def __init__(self, opts):
#     self.opt = opts
#     self.dataroot = opts.dataroot
#     # A
#     images_A = os.listdir(os.path.join(self.dataroot, opts.phase + 'A'))
#     self.A = [os.path.join(self.dataroot, opts.phase + 'A', x) for x in images_A]
#
#     # B
#     images_B = os.listdir(os.path.join(self.dataroot, opts.phase + 'B'))
#     self.B = [os.path.join(self.dataroot, opts.phase + 'B', x) for x in images_B]
#
#     # self.A_paths = sorted(self.A_paths)
#     # self.B_paths = sorted(self.B_paths)
#     self.A_size = len(self.A)
#     self.B_size = len(self.B)
#
#     transform_list = []
#
#     transform_list += [transforms.ToTensor(),
#                        transforms.Normalize((0.5, 0.5, 0.5),
#                                             (0.5, 0.5, 0.5))]
#     # transform_list = [transforms.ToTensor()]
#
#     self.transform = transforms.Compose(transform_list)
#
#   def __getitem__(self, index):
#     A_path = self.A[index]
#     B_path = self.B[index]
#
#     A_img = Image.open(A_path).convert('RGB')
#     B_img = Image.open(B_path).convert('RGB')
#
#     A_img = self.transform(A_img)
#     B_img = self.transform(B_img)
#     # A_path = self.A_paths[index % self.A_size]
#     # B_path = self.B_paths[index % self.B_size]
#     # A_size = A_img.size
#     # B_size = B_img.size
#     # A_size = A_size = (A_size[0]//16*16, A_size[1]//16*16)
#     # B_size = B_size = (B_size[0]//16*16, B_size[1]//16*16)
#     # A_img = A_img.resize(A_size, Image.BICUBIC)
#     # B_img = B_img.resize(B_size, Image.BICUBIC)
#     # A_gray = A_img.convert('LA')
#     # A_gray = 255.0-A_gray
#
#     # if self.opt.resize_or_crop == 'no':
#     #   r, g, b = A_img[0] + 1, A_img[1] + 1, A_img[2] + 1
#     #   A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
#     #   A_gray = torch.unsqueeze(A_gray, 0)
#     #   input_img = A_img
#     #   # A_gray = (1./A_gray)/255.
#     # else:
#     # w = A_img.size(2)
#     # h = A_img.size(1)
#
#       # A_gray = (1./A_gray)/255.
#     if (not self.opt.no_flip) and random.random() < 0.5:
#       idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
#       idx = torch.LongTensor(idx)
#       A_img = A_img.index_select(2, idx)
#       B_img = B_img.index_select(2, idx)
#     if (not self.opt.no_flip) and random.random() < 0.5:
#       idx = [i for i in range(A_img.size(1) - 1, -1, -1)]
#       idx = torch.LongTensor(idx)
#       A_img = A_img.index_select(1, idx)
#       B_img = B_img.index_select(1, idx)
#     # if self.opt.vary == 1 and (not self.opt.no_flip) and random.random() < 0.5:
#     #   times = random.randint(self.opt.low_times, self.opt.high_times) / 100.
#     #   input_img = (A_img + 1) / 2. / times
#     #   input_img = input_img * 2 - 1
#     # else:
#     #   input_img = A_img
#     # if self.opt.lighten:
#     #   B_img = (B_img + 1) / 2.
#     #   B_img = (B_img - torch.min(B_img)) / (torch.max(B_img) - torch.min(B_img))
#     #   B_img = B_img * 2. - 1
#
#     return A_img, B_img
#
#   def __len__(self):
#     return max(self.A_size, self.B_size)
#
#   def name(self):
#     return 'UnalignedDataset'
