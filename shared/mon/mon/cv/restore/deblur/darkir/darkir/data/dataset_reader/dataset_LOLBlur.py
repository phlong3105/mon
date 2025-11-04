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

def main_dataset_lolblur(rank = 1,
                         test_path='../../data/datasets/LOLBlur/test',
                         batch_size_test=1, 
                         verbose=False, 
                         num_workers=1, 
                         world_size = 1):
    
    PATH_VALID = test_path
    
    # paths to the blur and sharp sets of images
    paths_blur_valid = [os.path.join(PATH_VALID, 'low_blur_noise', path) for path in os.listdir(os.path.join(PATH_VALID, 'low_blur_noise'))]
    paths_sharp_valid = [os.path.join(PATH_VALID, 'high_sharp_scaled', path) for path in os.listdir(os.path.join(PATH_VALID, 'high_sharp_scaled'))]        
    
    # extract the images from their corresponding folders, now we get a list of lists

    paths_blur_valid = [[os.path.join(path_element, path_png) for path_png in os.listdir(path_element)] for path_element in paths_blur_valid ]
    paths_sharp_valid = [[os.path.join(path_element, path_png) for path_png in os.listdir(path_element)] for path_element in paths_sharp_valid ]


    list_blur_valid = flatten_list_comprehension(paths_blur_valid)
    list_sharp_valid = flatten_list_comprehension(paths_sharp_valid)

    # check if all the image routes are correct
    check_paths([list_blur_valid, list_sharp_valid])

    if verbose:
        print('Images in the subsets:')
        print("    -Images in the PATH_LOW_VALID folder: ", len(list_blur_valid))
        print("    -Images in the PATH_HIGH_VALID folder: ", len(list_sharp_valid))

    tensor_transform = transforms.ToTensor()


    # Load the dataset
    test_dataset = MyDataset_Crop(list_blur_valid, list_sharp_valid, cropsize=None,
                                  tensor_transform=tensor_transform, test=True)
    if world_size > 1:
        # Now we need to apply the Distributed sampler
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, shuffle= True, rank=rank)

        samplers = []
        # samplers = {'train': train_sampler, 'test': [test_sampler_gopro, test_sampler_lolblur]}
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
    test_loader, samplers = main_dataset_lolblur()
    
