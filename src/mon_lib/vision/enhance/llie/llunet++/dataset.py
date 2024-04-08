import os

import torch.utils.data as data
from PIL import Image
from torchvision import transforms


def img_loader(path):
    img = Image.open(path)
    return img


def get_imgs_list(ori_dirs, ucc_dirs):
    img_list = []
    for ori_imgdir in ori_dirs:
        img_name   = (ori_imgdir.split('/')[-1]).split('.')[0]
        ucc_imgdir = os.path.dirname(ucc_dirs[0]) + '/' + img_name + '.png'

        if ucc_imgdir in ucc_dirs:
            img_list.append(tuple([ori_imgdir, ucc_imgdir]))

    return img_list


class dataset(data.Dataset):
    
    def __init__(self, ori_dirs, ucc_dirs, train=True):
        super(dataset, self).__init__()

        self.img_list = get_imgs_list(ori_dirs, ucc_dirs)
        if len(self.img_list) == 0:
            raise RuntimeError('Found 0 image pairs in given directories.')

        self.train = train
        
        print('Found {} pairs of training images'.format(len(self.img_list)))
        
    def __getitem__(self, index):
        img_paths = self.img_list[index]
        sample    = [img_loader(img_paths[i]) for i in range(len(img_paths))]

        transform = transforms.Compose([
            transforms.Resize([384, 384]),             
            transforms.ToTensor(),
        ])
        
        sample[0] = transform(sample[0])
        sample[1] = transform(sample[1])
        
        return sample

    def __len__(self):
        return len(self.img_list)
