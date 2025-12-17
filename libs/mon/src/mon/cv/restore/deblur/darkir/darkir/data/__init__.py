from .dataset_reader.dataset_LOLBlur import main_dataset_lolblur
from .dataset_reader.dataset_all_LOL import main_dataset_all_lol
from .dataset_reader.dataset_real_LSRW import main_dataset_real_LSRW
from .dataset_reader.dataset_realblur_night import main_dataset_realblur_night
from .dataset_reader.dataset_dicm import main_dataset_dicm
from .dataset_reader.dataset_lime import main_dataset_lime
from .dataset_reader.dataset_mef import main_dataset_mef
from .dataset_reader.dataset_npe import main_dataset_npe
from .dataset_reader.dataset_vv import main_dataset_vv
from .dataset_reader.dataset_exdark import main_dataset_exdark

def create_test_data(rank, world_size, opt):
    '''
    opt: a dictionary from the yaml config key datasets 
    '''
    name = opt['name']
    test_path = opt['val']['test_path']
    batch_size_test=opt['val']['batch_size_test']
    verbose=opt['train']['verbose']
    num_workers=opt['train']['n_workers']
    
    if rank != 0:
        verbose = False
    samplers = None # TEmporal change!!
    if name == 'LOLBlur':
        test_loader, samplers = main_dataset_lolblur(rank = rank,
                                                test_path = test_path,
                                                batch_size_test=batch_size_test,
                                                verbose=verbose,
                                                num_workers=num_workers,
                                                world_size = world_size) 
    elif name == 'All_LOL':
        test_loader, samplers = main_dataset_all_lol(rank=rank, 
                                                test_path = test_path,
                                                batch_size_test=batch_size_test,
                                                verbose=verbose,
                                                num_workers=num_workers,
                                                world_size=world_size)   
    elif name == 'real_LSRW':
        test_loader, samplers = main_dataset_real_LSRW(rank=rank, 
                                                test_path = test_path,
                                                batch_size_test=batch_size_test,
                                                verbose=verbose,
                                                num_workers=num_workers,
                                                world_size=world_size)  
    elif name == 'RealBlur_Night':
        test_loader, samplers = main_dataset_realblur_night(rank = 1,
                                                test_path=test_path,
                                                batch_size_test=1, 
                                                verbose=False, 
                                                num_workers=1, 
                                                world_size = 1)
    elif name == 'DICM':
        test_loader, samplers = main_dataset_dicm(rank = 1,
                                                test_path=test_path,
                                                batch_size_test=1, 
                                                verbose=False, 
                                                num_workers=1, 
                                                world_size = 1)
    elif name == 'MEF':
        test_loader, samplers = main_dataset_mef(rank = 1,
                                                test_path=test_path,
                                                batch_size_test=1, 
                                                verbose=False, 
                                                num_workers=1, 
                                                world_size = 1)
    elif name == 'NPE':
        test_loader, samplers = main_dataset_npe(rank = 1,
                                                test_path=test_path,
                                                batch_size_test=1, 
                                                verbose=False, 
                                                num_workers=1, 
                                                world_size = 1)
    elif name == 'VV':
        test_loader, samplers = main_dataset_vv(rank = 1,
                                                test_path=test_path,
                                                batch_size_test=1, 
                                                verbose=False, 
                                                num_workers=1, 
                                                world_size = 1)
    elif name == 'LIME':
        test_loader, samplers = main_dataset_lime(rank = 1,
                                                test_path=test_path,
                                                batch_size_test=1, 
                                                verbose=False, 
                                                num_workers=1, 
                                                world_size = 1)

    elif name == 'ExDark':
        test_loader, samplers = main_dataset_exdark(rank = 1,
                                                test_path=test_path,
                                                batch_size_test=1, 
                                                verbose=False, 
                                                num_workers=1, 
                                                world_size = 1)

    else:
        raise NotImplementedError(f'{name} is not implemented')        
    if rank ==0: print(f'Using {name} Dataset')
    
    return test_loader, samplers


__all__ = ['create_test_data']