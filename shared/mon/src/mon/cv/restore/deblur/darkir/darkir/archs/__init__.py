import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
# from ptflops import get_model_complexity_info

from .DarkIR import DarkIR   


def create_model(opt, rank, device, adapter=False, torchrun=False):
    '''
    Creates the model.
    opt: a dictionary from the yaml config key network
    '''
    name  = opt['name']

    model = DarkIR(
        img_channel        = opt['img_channels'],
        width              = opt['width'],
        middle_blk_num_enc = opt['middle_blk_num_enc'],
        middle_blk_num_dec = opt['middle_blk_num_dec'],
        enc_blk_nums       = opt['enc_blk_nums'],
        dec_blk_nums       = opt['dec_blk_nums'],
        dilations          = opt['dilations'],
        extra_depth_wise   = opt['extra_depth_wise']
    )

    if rank == 0:
        # print(f'Using {name} network')
        input_size   = (3, 512, 512)
        # macs, params = get_model_complexity_info(model, input_size, print_per_layer_stat = False)
        # print(f'Computational complexity at {input_size}: {macs}')
        # print('Number of parameters: ', params)
    else:
        macs, params = None, None

    if torchrun:
        model.to(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=adapter)
    else:
        model.to(device)
    return model, macs, params


def create_optim_scheduler(opt, model):
    '''
    Returns the optim and its scheduler.
    opt: a dictionary of the yaml config file with the train key
    '''
    optim = torch.optim.AdamW( filter(lambda p: p.requires_grad, model.parameters()) , 
                            lr = opt['lr_initial'],
                            weight_decay = opt['weight_decay'],
                            betas = opt['betas'])
    
    if opt['lr_scheme'] == 'CosineAnnealing':
        scheduler = CosineAnnealingLR(optim, T_max=opt['epochs'], eta_min=opt['eta_min'])
    else: 
        raise NotImplementedError('scheduler not implemented')    
        
    return optim, scheduler


def load_weights(model, old_weights):
    '''
    Loads the weights of a pretrained model, picking only the weights that are
    in the new model.
    '''
    new_weights = model.state_dict()
    new_weights.update({k: v for k, v in old_weights.items() if k in new_weights})
    
    model.load_state_dict(new_weights)
    return model


def load_optim(optim, optim_weights):
    '''
    Loads the values of the optimizer picking only the weights that are in the new model.
    '''
    optim_new_weights = optim.state_dict()
    # optim_new_weights.load_state_dict(optim_weights)
    optim_new_weights.update({k:v for k, v in optim_weights.items() if k in optim_new_weights})
    return optim


def resume_model(model,
                 optim,
                 scheduler, 
                 path_model, 
                 rank,resume:str=None):
    '''
    Returns the loaded weights of model and optimizer if resume flag is True
    '''
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    if resume:
        checkpoints = torch.load(path_model, map_location=map_location, weights_only=False)
        weights = checkpoints['model_state_dict']
        model = load_weights(model, old_weights=weights)
        optim = load_optim(optim, optim_weights = checkpoints['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
        start_epochs = checkpoints['epoch']

        if rank == 0: print('Loaded weights')
    else:
        start_epochs = 0
        if rank==0: print('Starting from zero the training')
    
    return model, optim, scheduler, start_epochs


def find_different_keys(dict1, dict2):

# Finding different keys
    different_keys = set(dict1.keys()) ^ set(dict2.keys())

    return different_keys


def number_common_keys(dict1, dict2):
    # Finding common keys
    common_keys = set(dict1.keys()) & set(dict2.keys())

    # Counting the number of common keys
    common_keys_count = len(common_keys)
    return common_keys_count


# # Function to add 'modules_list' prefix after the first numeric index
# def add_middle_prefix(state_dict, middle_prefix, target_strings):
#     new_state_dict = {}
#     for key, value in state_dict.items():
#         for target in target_strings:
#             if target in key:
#                 parts = key.split('.')
#                 # Find the first numeric index after the target string
#                 for i, part in enumerate(parts):
#                     if part == target:
#                         # Insert the middle prefix after the first numeric index
#                         if i + 1 < len(parts) and parts[i + 1].isdigit():
#                             parts.insert(i + 2, middle_prefix)
#                             break
#                 new_key = '.'.join(parts)
#                 new_state_dict[new_key] = value
#                 break
#         else:
#             new_state_dict[key] = value
#     return new_state_dict

# # Function to adjust keys for 'middle_blks.' prefix
# def adjust_middle_blks_keys(state_dict, target_prefix, middle_prefix):
#     new_state_dict = {}
#     for key, value in state_dict.items():
#         if target_prefix in key:
#             parts = key.split('.')
#             # Find the target prefix and adjust the key
#             for i, part in enumerate(parts):
#                 if part == target_prefix.rstrip('.'):
#                     if i + 1 < len(parts) and parts[i + 1].isdigit():
#                         # Swap the numerical part and the middle prefix
#                         new_key = '.'.join(parts[:i + 1] + [middle_prefix] + parts[i + 1:i + 2] + parts[i + 2:])
#                         new_state_dict[new_key] = value
#                         break
#         else:
#             new_state_dict[key] = value
#     return new_state_dict

# def resume_nafnet(model,
#                  optim,
#                  scheduler, 
#                  path_adapter,
#                  path_model,
#                  rank, resume:str=None):
#     '''
#     Returns the loaded weights of model and optimizer if resume flag is True
#     '''
#     map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
#     #first load the model weights
#     checkpoints = torch.load(path_model, map_location=map_location, weights_only=False)
#     weights = checkpoints
#     if rank==0:
#         print(len(weights), len(model.state_dict().keys()))

#         different_keys = find_different_keys(weights, model.state_dict())
#         filtered_keys = {item for item in different_keys if 'adapter' not in item}
#         print(filtered_keys)
#         print(len(filtered_keys))
#     model = load_weights(model, old_weights=weights) 
#     #now if needed load the adapter weights
#     if resume:
#         checkpoints = torch.load(path_adapter, map_location=map_location, weights_only=False)
#         weights = checkpoints
#         model = load_weights(model, old_weights=weights)
#         # optim = load_optim(optim, optim_weights = checkpoints['optimizer_state_dict'])
#         scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
#         start_epochs = checkpoints['epoch']

#         if rank == 0: print('Loaded weights')
#     else:
#         start_epochs = 0
#         if rank == 0: print('Starting from zero the training')
    
#     return model, optim, scheduler, start_epochs


def save_checkpoint(model, optim, scheduler, metrics_eval, metrics_train, paths, adapter = False, rank = None):

    '''
    Save the .pt of the model after each epoch.
    '''
    best_psnr = metrics_train['best_psnr']
    if rank!=0: 
        return best_psnr
    
    if type(next(iter(metrics_eval.values()))) != dict:
        metrics_eval = {'metrics': metrics_eval}

    weights = model.state_dict()

    # Save the model after every epoch
    model_to_save = {
        'epoch': metrics_train['epoch'],
        'model_state_dict': weights,
        'optimizer_state_dict': optim.state_dict(),
        'loss': metrics_train['train_loss'],
        'scheduler_state_dict': scheduler.state_dict()
    }

    try:
        torch.save(model_to_save, paths['new'])

        # Save best model if new valid_psnr is higher than the best one
        if next(iter(metrics_eval.values()))['valid_psnr'] >= metrics_train['best_psnr']:
            torch.save(model_to_save, paths['best'])
            metrics_train['best_psnr'] = next(iter(metrics_eval.values()))['valid_psnr']  # update best psnr
    except Exception as e:
        print(f"Error saving model: {e}")
    return metrics_train['best_psnr']


__all__ = ['create_model', 'resume_model', 'create_optim_scheduler', 'save_checkpoint',
           'load_optim', 'load_weights']
