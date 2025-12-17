import wandb
from torchvision.utils import make_grid


def init_wandb(rank, opt):
    '''
    Initiates wandb if needed.
    opt: a dictionary from the yaml config 
    '''
    if opt['wandb']['init'] and rank == 0:
        wandb.login()
        wandb.init(
            # set the wandb project where this run will be logged
            project=opt['wandb']['project'], entity=opt['wandb']['entity'], 
            name=opt['wandb']['name'], save_code=opt['wandb']['save_code'],
            config = opt,
            resume = opt['wandb']['resume'],
            id = opt['wandb']['id'],
            dir = opt['wandb']['dir']
           # notes= subprocess.check_output(["git", "log", "-1", "--pretty=%B"]).decode("utf-8").strip() #latest github commit 
        )       
    else:
        if rank==0: print('Not uploading to wandb')


def create_one_grid(dict_images):
    '''
    A function to create a grid of images to log in wandb.
    '''
    images, caption = [], []
    for k, v in dict_images.items():
        caption.append(k)
        images.append(v)

    n=len(images)
    images_array = make_grid(images, n)
    
    images = wandb.Image(images_array, caption = caption)
    
    return images


def create_grid(dict_images):
    '''
    A function to create all the grids of images to log in wandb.
    images: A dictionary of images 
    '''
    #first you need to assert that is a dictionary
    if type(next(iter(dict_images.values()))) != dict:
        dict_images = {'imgs': dict_images}    
    
    if len(dict_images) > 1:
        all_grids = {}
        for key, dict_img in dict_images.items():
            grid = create_one_grid(dict_images=dict_img)
            all_grids[f'{key}'] == grid
        
        return all_grids
    
    else:
        return create_one_grid(dict_images=dict_images['imgs'])


def logging_dict(metrics_train, metrics_eval, dict_images):
    '''
    Creates a logging dict to log results in wandb.
    '''
    # assert that you work with a proper dict
    if type(next(iter(metrics_eval.values()))) != dict:
        metrics_eval = {'metrics_eval': metrics_eval} 
    if type(next(iter(dict_images.values()))) != dict:
        dict_images = {'dict_images': dict_images}
    
    logger = {}    
    if len(metrics_eval) > 1:

        for (key_metric, metric), (key_imgs, imgs) in zip(metrics_eval.items(), dict_images.items()):
            for key, value in metric.items():
                logger[f'{key_metric}_{key}']= value
            logger[f'{key_imgs}'] = create_one_grid(dict_images=imgs)
    else:
        metrics_eval = metrics_eval['metrics_eval']
        grid = create_one_grid(dict_images['dict_images'])
        for key, value in metrics_eval.items():
            logger[f'{key}'] =value
        logger['grid'] = grid
        
    # finally load the metrics_train
    metrics_train.pop('best_psnr')
    for key, value in metrics_train.items():
        logger[f'{key}'] = value
        
    return logger

# def combine_dicts(dict1, dict2, names=['gopro', 'lolblur']):
#     '''
#     Combines two dicts.
#     '''
#     combined_dict = {f"{key}_{names[0]}": value for key, value in dict1.items()}
#     combined_dict.update({f"{key}_{names[1]}": value for key, value in dict2.items()})
#     return combined_dict


def create_path_models(opt):
    ''' 
    Creates a set of paths to save the model based on the config file.
    '''

    PATH_MODEL     = opt['path']
    return PATH_MODEL


if __name__ == '__main__':
    
    pass
