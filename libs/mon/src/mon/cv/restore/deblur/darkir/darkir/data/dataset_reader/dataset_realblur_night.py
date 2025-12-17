import os
from glob import glob

# PyTorch library
from torch.utils.data import DataLoader, DistributedSampler

try:
    from .datapipeline import *
    from .utils import *
except:
    from datapipeline import *
    from utils import *

def main_dataset_realblur_night(rank = 1,
                         test_path='../../data/datasets/RealBlur-Night',
                         batch_size_test=1, 
                         verbose=False, 
                         num_workers=1, 
                         crop_type='Random',
                         world_size = 1):
    
    PATH_VALID = test_path
    
    paths_blur_valid = [os.path.join(PATH_VALID, 'realblur_dataset_test',path) for path in os.listdir(os.path.join(PATH_VALID, 'realblur_dataset_test'))]
    paths_sharp_valid = [os.path.join(PATH_VALID, 'realblur_dataset_test', path) for path in os.listdir(os.path.join(PATH_VALID, 'realblur_dataset_test'))]        

    list_blur_valid = paths_blur_valid
    list_sharp_valid = paths_sharp_valid

    # check if all the image routes are correct
    check_paths([list_blur_valid, list_sharp_valid])

    if verbose:
        print('Images in the subsets:')
        print("    -Images in the PATH_LOW_VALID folder: ", len(list_blur_valid))
        print("    -Images in the PATH_HIGH_VALID folder: ", len(list_sharp_valid))

    tensor_transform = transforms.ToTensor()

    # Load the datasets
    test_dataset = MyDataset_Crop(list_blur_valid, list_sharp_valid, cropsize=None,
                                  tensor_transform=tensor_transform, test=True, 
                                  crop_type=crop_type)
    if world_size > 1:
        # Now we need to apply the Distributed sampler
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, shuffle= True, rank=rank)

        samplers = []
        samplers.append(test_sampler)
        
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False, sampler=test_sampler)
    else:        
        # #Load the data loaders
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=True,
                                num_workers=num_workers, pin_memory=True, drop_last=False)
        samplers = None

    return test_loader, samplers

if __name__ == '__main__':
    test_loader, samplers = main_dataset_realblur_night(test_path='/mnt/valab-datasets/RealBlur-Night')

    print(len(test_loader))

    for i in range(0, 10):
        print(next(iter(test_loader))[0].shape)
    