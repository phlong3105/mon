import os
from copy import deepcopy

import h5py
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose

from .transform import *
from ..perturbation.perturbation import perturbation


def read_depth_sintel(filename):
    TAG_FLOAT = 202021.25
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    depth = np.clip(depth, None, 72)
    
    return depth


def read_depth_ETH3D(filename):
    HEIGHT, WIDTH = 4032, 6048
    with open(filename, "rb") as file:
        binary_data = file.read()
    depth_decoded = np.frombuffer(binary_data, dtype=np.float32).copy()
    depth_decoded = depth_decoded.reshape((HEIGHT, WIDTH))
    depth_decoded[depth_decoded == torch.inf] = 0.0
    return depth_decoded


class SemiDataset(Dataset):
    """
    A semi-supervised dataset class for depth estimation tasks.
    
    This dataset supports both labeled and unlabeled data for semi-supervised learning.
    It handles multiple depth datasets including KITTI, ETH3D, NYU, DIODE, and others.
    The dataset can operate in different modes: training with labeled data (train_l),
    training with unlabeled data (train_u), and validation (val).
    
    Attributes:
        name (str): Name of the dataset
        root (str): Root directory path for the dataset
        mode (str): Dataset mode ('train_l', 'train_u', or 'val')
        size (int): Target size for image resizing
        dataset (str): Dataset name (same as name)
        argu_mode (str): Argument mode for data augmentation
        cfg (dict): Configuration dictionary for dataset settings
        ids (list): List of sample IDs to load
    """

    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None,argu_mode="robo_depth",cfg=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.dataset = name
       
        self.argu_mode = argu_mode
        self.cfg = cfg

        if mode == 'train_l' or mode == 'train_u':
            if id_path is not None:
                with open(id_path, 'r') as f:
                    self.ids = f.read().splitlines()
                if nsample is not None:
                    self.ids *= math.ceil(nsample / len(self.ids))
                    random.shuffle(self.ids)
                    self.ids = self.ids[:nsample]
            else:
                self.ids = []
        else:
            with open('partitions/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        sample = {}
        if self.mode == 'val':
            transform = Compose([
                Resize(
                    width=self.size,
                    height=self.size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])
            file_name = id.split(' ')[0].split('/')[-1].split('.')[0]
            sample['file_name'] = file_name
            if self.dataset == 'kitti':
                raw_image = cv2.imread(os.path.join(self.root, id.split(' ')[0]))
                sample['path'] = os.path.join(self.root, id.split(' ')[0])
            elif self.dataset == 'ETH3D':
                sample['path'] = os.path.join(self.root, id.split(' ')[0])
                raw_image= cv2.imread(os.path.join(self.root, id.split(' ')[0]))
                raw_image = cv2.resize(raw_image, (6048, 4032), interpolation=cv2.INTER_CUBIC)
                
            else:
                raw_image = cv2.imread(os.path.join(self.root, id.split(' ')[0]))
                sample['path'] = os.path.join(self.root, id.split(' ')[0])
            sample['image'] = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
            is_labeled = True
            if is_labeled:
                if id.split(' ')[1].endswith('.h5'):
                    with h5py.File(os.path.join(self.root, id.split(' ')[1]), 'r') as h5_file:
                        depth_array = np.array(h5_file['depth'])
                        sample['depth'] = Image.fromarray(depth_array)
                elif self.dataset == 'kitti':
                    sample['depth'] = Image.fromarray(
                    np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))

                elif self.dataset in ['kitti_c_gaussian','kitti_c_motion','kitti_c_dark','kitti_c_snow','kitti_c_fog']:
                     sample['depth'] = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))

                elif self.dataset in ['robotcar', 'nuscene']:
                    sample['depth'] = Image.fromarray(np.load(os.path.join(self.root, id.split(' ')[1])))
                elif self.dataset in ['pixel_day','pixel_night','pixel_rain','pixel_fog','pixel_night_fog','pixel_night_rain']:
                    data=np.load(os.path.join(self.root, id.split(' ')[1]))
                    depth_np=data['arr_0']
                    sample['depth'] = Image.fromarray(depth_np)
                elif self.dataset == 'sintel':
                   
                    sample['depth']=Image.fromarray(read_depth_sintel(os.path.join(self.root, id.split(' ')[1])))
                    
                elif self.dataset =='DIODE':

                    depth_data=Image.fromarray(np.load(os.path.join(self.root, id.split(' ')[1])).reshape(768, 1024))
                    sample['depth']=depth_data
                
                elif self.dataset =="ETH3D":
                    sample['depth']=Image.fromarray(read_depth_ETH3D(os.path.join(self.root, id.split(' ')[1])))

                elif self.dataset in ['DA2K','DA2K_blur','DA2K_dark','DA2K_fog','DA2K_snow']:
                    depth_file = os.path.join(self.root, id.split(' ')[1])
                    with open(depth_file, 'r') as f:
                        points_data = f.read().strip().split('\n')
                        points = []
                        for line in points_data:
                            values = line.split()
                            if len(values) >= 5:
                                h1, w1, h2, w2, point_type = float(values[0]), float(values[1]), float(values[2]), float(values[3]), values[4]
                                points.append([h1, w1, h2, w2, point_type])
                   
                    sample['points']=points
                else:    
                    sample['depth'] = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))

                has_valid_depth = True
               
            else:
                sample['depth'] = False
                has_valid_depth = False               
            sample['has_valid_depth'] = has_valid_depth

            do_kb_crop = False
            if do_kb_crop:
                height = sample['image'].shape[0]
                width = sample['image'].shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                if has_valid_depth and sample['depth'] is not False:
                    sample['depth'] = sample['depth'].crop(
                        (left_margin, top_margin, left_margin + 1216, top_margin + 352))
                sample['image'] = sample['image'][top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
            h, w = sample['image'].shape[:2]
            sample['ori_size'] = (h, w)

            if self.dataset in ['DA2K','DA2K_blur','DA2K_dark','DA2K_fog','DA2K_snow']:
                return transform(sample)
            sample['depth'] = np.array(sample['depth']).astype('float32')
            
            sample= transform(sample)
            
        else:
            file_name = id.split(' ')[0].split('/')[-1].split('.')[0]
            sample['file_name'] = file_name
            sample['path'] = os.path.join(self.root, id.split(' ')[0])
            file_path = sample['path'].replace(sample['path'].split('/')[-1], '')
            sample['image'] = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
            sample['depth']=np.ones((sample['image'].height,sample['image'].width))
            
            
            sample['depth']=Image.fromarray(sample['depth'])
            
            do_kb_crop = False
            if do_kb_crop:
                height = sample['image'].height
                width = sample['image'].width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                sample['image'] = sample['image'].crop(
                    (left_margin, top_margin, left_margin + 1216, top_margin + 352))
                
        if self.mode == 'val':
            if has_valid_depth:
                sample = normalize_depth_only(sample,self.dataset)
                if self.dataset == 'nyu':
                    sample['depth'] = sample['depth'] / 1000.0
                    sample['mask'] = (sample['depth'] >= 1e-5) & (sample['depth'] <= 10)
                elif self.dataset == 'kitti':
                    sample['depth'] = sample['depth'] / 256.0
                    sample['mask'] = (sample['depth'] > 0)
                elif self.dataset in ['kitti_c_gaussian','kitti_c_motion','kitti_c_dark','kitti_c_snow','kitti_c_fog']:
                    sample['depth'] = sample['depth'] / 256.0
                    sample['mask'] = (sample['depth'] > 0)
                elif self.dataset in ['drive_foggy', 'drive_cloudy', 'drive_rainy']:
                    sample['depth'] = sample['depth'] / 256.0
                    sample['mask'] = (sample['depth'] > 0)
                elif 'DIML' in id.split(' ')[1]:
                    sample['depth'] = sample['depth'] /1000.0
                    sample['mask'] = (sample['depth'] >= 1e-5) & (sample['depth'] <= 10)       
                elif 'vkitti2' in id.split(' ')[1]:
                    sample['depth'] = sample['depth'] /100.0
                    sample['mask'] = (sample['depth'] >= 1e-5)
                elif 'blendedmvs' in id.split(' ')[1]:
                    sample['depth'] = sample['depth'] /1000.0
                    sample['mask'] = (sample['depth'] >= 1e-5)    
                elif 'pixel' in id.split(' ')[1]:
                    sample['depth'] = sample['depth'] 
                    sample['mask'] = (sample['depth'] >= 1e-5) 
                elif self.dataset=='DIODE':
                    mask1=torch.from_numpy(np.load(os.path.join(self.root,id.split(' ')[2]))).bool()
                    mask2=(sample['depth'] >= 1e-5) & (sample['depth'] <= 150)
                    sample['mask']=mask1 & mask2
                elif self.dataset == 'ETH3D':
                    sample['depth'] = sample['depth'] 
                    sample['mask'] = (sample['depth'] >= 1e-5)  & (sample['depth'] <= 200)          
                else:
                    sample['depth'] = sample['depth'] 
                    sample['mask'] = (sample['depth'] >= 1e-5)    
            else:
                sample['mask'] = False
           
            return sample

        need_recrop = True
        sample = resize_certain(sample, self.size)
        cnt = 0
        while(need_recrop):
            sample_temp = sample.copy()
            sample_temp = crop(sample_temp, self.size)
            need_recrop = False
            if 'depth' in sample.keys() and (torch.sum( torch.from_numpy(np.array(sample_temp['depth']).astype(np.float32)).float() > 0) == 0) and cnt < 1:
                need_recrop = True
                cnt+=1
        sample = sample_temp
        sample = hflip(sample, p=0.5)
        
        sample['depth'] = torch.from_numpy(np.array(sample['depth'])).float()
        sample['mask']=(sample['depth'] > 0.0)
        if self.cfg is not None and self.cfg.get('check_mask', False) and torch.sum(torch.from_numpy(np.array(sample['mask']).astype(np.float32)).float()) == 0:
                print(f"Warning: mask is all zeros for {sample['path']}")
       
        img_w, img_s1, img_ori = deepcopy(sample['image']), deepcopy(sample['image']), deepcopy(sample['image'])
        sample['img_w'] = img_w
       
    
        img_s1=perturbation(sample['path'], img_s1, self.cfg)
        
        sample['img_s1'] = img_s1
        sample['id'] = id
        sample['img_w'] = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])(sample['img_w'])
        sample['img_s1'] = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])(sample['img_s1'])
        
        del sample['image']
        return sample   

    def __len__(self):
        return len(self.ids)
