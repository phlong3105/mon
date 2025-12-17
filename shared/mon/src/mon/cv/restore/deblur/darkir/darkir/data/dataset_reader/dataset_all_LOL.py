import os

# PyTorch library
from torch.utils.data import DataLoader, DistributedSampler

try:
    from .datapipeline import *
    from .utils import *
except:
    from datapipeline import *
    from utils import *

def main_dataset_all_lol(rank = 1, test_path='../../data/datasets/', batch_size_test=1, verbose=False,
                       num_workers=1, world_size = 1):

    # now load the LOLv2_real dataset
    PATH_VALID = os.path.join(test_path, 'LOL-v2/Real_captured', 'test')

    # paths to the blur and sharp sets of images
    paths_blur_valid = [os.path.join(PATH_VALID, 'Low', path) for path in os.listdir(os.path.join(PATH_VALID, 'Low'))]
    paths_sharp_valid = [os.path.join(PATH_VALID, 'Normal', path) for path in os.listdir(os.path.join(PATH_VALID, 'Normal'))]        
    

    list_blur_valid_LOLv2_real = paths_blur_valid
    list_sharp_valid_LOLv2_real = paths_sharp_valid

    check_paths([list_blur_valid_LOLv2_real, list_sharp_valid_LOLv2_real])

    if verbose:
        print('Images in the subsets of LOLv2-real:')
        print("    -Images in the PATH_LOW_VALID folder: ", len(list_blur_valid_LOLv2_real))
        print("    -Images in the PATH_HIGH_VALID folder: ", len(list_sharp_valid_LOLv2_real), '\n')

    #------------------------------------------------------------------------
    # now load the LOLv2_synth dataset    
    PATH_VALID = os.path.join(test_path, 'LOL-v2/Synthetic', 'test')
    
    # paths to the blur and sharp sets of images 
    paths_blur_valid = [os.path.join(PATH_VALID, 'Low', path) for path in os.listdir(os.path.join(PATH_VALID, 'Low'))]
    paths_sharp_valid = [os.path.join(PATH_VALID, 'Normal', path) for path in os.listdir(os.path.join(PATH_VALID, 'Normal'))]        

    list_blur_valid_LOLv2_synth = paths_blur_valid
    list_sharp_valid_LOLv2_synth = paths_sharp_valid

    # check if all the image routes are correct
    check_paths([list_blur_valid_LOLv2_synth, list_sharp_valid_LOLv2_synth])

    if verbose:
        print('Images in the subsets of LOLv2-synth:')
        print("    -Images in the PATH_LOW_VALID folder: ", len(list_blur_valid_LOLv2_synth))
        print("    -Images in the PATH_HIGH_VALID folder: ", len(list_sharp_valid_LOLv2_synth), '\n')    

    #------------------------------------------------------------------------  
    # finally the LOLBlur dataset
    PATH_VALID = os.path.join(test_path, 'LOLBlur', 'test')
    
    # paths to the blur and sharp sets of images
    
    paths_blur_valid = [os.path.join(PATH_VALID, 'low_blur_noise', path) for path in os.listdir(os.path.join(PATH_VALID, 'low_blur_noise'))]
    paths_sharp_valid = [os.path.join(PATH_VALID, 'high_sharp_scaled', path) for path in os.listdir(os.path.join(PATH_VALID, 'high_sharp_scaled'))]        
    
    # extract the images from their corresponding folders, now we get a list of lists
    paths_blur_valid = [[os.path.join(path_element, path_png) for path_png in os.listdir(path_element)] for path_element in paths_blur_valid ]
    paths_sharp_valid = [[os.path.join(path_element, path_png) for path_png in os.listdir(path_element)] for path_element in paths_sharp_valid ]


    list_blur_valid_lolblur = flatten_list_comprehension(paths_blur_valid)
    list_sharp_valid_lolblur = flatten_list_comprehension(paths_sharp_valid)

    # check if all the image routes are correct
    check_paths([list_blur_valid_lolblur, list_sharp_valid_lolblur])

    if verbose:
        print('Images in the subsets of LOL-Blur:')
        print("    -Images in the PATH_LOW_VALID folder: ", len(list_blur_valid_lolblur))
        print("    -Images in the PATH_HIGH_VALID folder: ", len(list_sharp_valid_lolblur), '\n')

    #------------------------------------------------------------------------  

    tensor_transform = transforms.ToTensor()

    # Load the datasets
    test_dataset_lolv2 = MyDataset_Crop(list_blur_valid_LOLv2_real, list_sharp_valid_LOLv2_real, cropsize=None,
                                  tensor_transform=tensor_transform, test=True)

    test_dataset_lolv2_synth = MyDataset_Crop(list_blur_valid_LOLv2_synth, list_sharp_valid_LOLv2_synth, cropsize=None,
                                  tensor_transform=tensor_transform, test=True)
    
    test_dataset_lolblur = MyDataset_Crop(list_blur_valid_lolblur, list_sharp_valid_lolblur, cropsize=None,
                                  tensor_transform=tensor_transform, test=True)

    if world_size > 1:
        # Now we need to apply the Distributed sampler
        test_sampler_lolv2 = DistributedSampler(test_dataset_lolv2, num_replicas=world_size, shuffle= True, rank=rank)
        test_sampler_lolv2_synth = DistributedSampler(test_dataset_lolv2_synth, num_replicas=world_size, shuffle= True, rank=rank)
        test_sampler_lolblur = DistributedSampler(test_dataset_lolblur, num_replicas=world_size, shuffle= True, rank=rank)

        samplers = []

        samplers.append(test_sampler_lolv2)
        samplers.append(test_sampler_lolv2_synth)
        samplers.append(test_sampler_lolblur)

        test_loader_lolv2 = DataLoader(dataset=test_dataset_lolv2, batch_size=batch_size_test, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False, sampler = test_sampler_lolv2)
        test_loader_lolv2_synth = DataLoader(dataset=test_dataset_lolv2_synth, batch_size=batch_size_test, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False, sampler = test_sampler_lolv2_synth)
        test_loader_lolblur = DataLoader(dataset=test_dataset_lolblur, batch_size=batch_size_test, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False, sampler= test_sampler_lolblur)
        
    else:
        test_loader_lolv2 = DataLoader(dataset=test_dataset_lolv2, batch_size=batch_size_test, shuffle=True,
                                num_workers=num_workers, pin_memory=True, drop_last=False, sampler = None)
        test_loader_lolv2_synth = DataLoader(dataset=test_dataset_lolv2_synth, batch_size=batch_size_test, shuffle=True,
                                num_workers=num_workers, pin_memory=True, drop_last=False, sampler = None)                              
        test_loader_lolblur = DataLoader(dataset=test_dataset_lolblur, batch_size=batch_size_test, shuffle=True,
                                num_workers=num_workers, pin_memory=True, drop_last=False, sampler= None)       
        samplers = None

    test_loader = {'lolblur':{'loader':test_loader_lolblur, 'adapter': False}, 'lolv2':{'loader':test_loader_lolv2, 'adapter': False}, 'lolv2_synth':{'loader':test_loader_lolv2_synth, 'adapter': False}}

    return test_loader, samplers

if __name__ == '__main__':
    
    test_loader, samplers= main_dataset_all_lol(verbose = True, test_path='/mnt/valab-datasets/')

    