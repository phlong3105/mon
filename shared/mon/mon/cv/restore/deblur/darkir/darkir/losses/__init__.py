from .loss import CharbonnierLoss, EdgeLoss, EnhanceLoss, FrequencyLoss, L1Loss, MSELoss, SSIM, VGGLoss


def create_loss(opt, rank):
    
    '''
    Returns the needed losses for evaluating our model
    '''
    losses = dict()
    
    # first the pixel losses
    if opt['pixel_criterion'] == 'l1':
        pixel_loss = L1Loss()
    elif opt['pixel_criterion'] == 'l2':
        pixel_loss = MSELoss()
    elif opt['pixel_criterion'] == 'Charbonnier':
        pixel_loss = CharbonnierLoss()
    else:
        raise NotImplementedError('Pixel Criterion not implemented')

    losses['pixel_loss'] = pixel_loss.to(rank)
    if rank == 0: print(f"Using pixel loss {opt['pixel_criterion']} ")
    # now the perceptual loss
    if opt['perceptual']:     
        perceptual_loss = VGGLoss(loss_weight = opt['perceptual_weight'],
                                criterion = opt['perceptual_criterion'],
                                reduction = opt['perceptual_reduction']).to(rank)
        losses['perceptual_loss'] = perceptual_loss
        if rank==0: print(f"Using perceptual loss {opt['perceptual_criterion']} with weight {opt['perceptual_weight']}")
    # the edge loss
    if opt['edge']: 
        edge_loss = EdgeLoss(loss_weight = opt['edge_weight'],
                                criterion = opt['edge_criterion'],
                                reduction = opt['edge_reduction'],
                                rank = rank).to(rank)
        losses['edge_loss'] = edge_loss
        if rank==0: print(f"Using edge loss {opt['edge_criterion']} with weight {opt['edge_weight']}")
    # the frequency loss
    if opt['frequency']:
        frequency_loss = FrequencyLoss(loss_weight = opt['edge_weight'],
                                reduction = opt['edge_reduction'],
                                criterion = opt['frequency_criterion']).to(rank)
        losses['frequecy_loss'] = frequency_loss
        if rank==0: print(f"Using frequency loss {opt['frequency_criterion']} with weight {opt['frequency_weight']}")
    # the enhance loss
    if opt['enhance']:
        enhance_loss = EnhanceLoss(loss_weight= opt['enhance_weight'],
                                reduction = opt['enhance_reduction'],
                                criterion = opt['enhance_criterion']).to(rank)
        losses['enhance_loss'] = enhance_loss
        if rank==0: print(f"Using enhance loss {opt['enhance_criterion']} with weight {opt['enhance_weight']}")
    
    return losses


def calculate_loss(
    all_losses,
    enhanced_batch,
    high_batch,
    outside_batch = None,
    scale_factor=8):
    '''
    Returns the calculated values of the losses for optimization.
    outsize_batch: if None it doen't apply the enhance loss
    '''
    
    l_pixel = all_losses['pixel_loss'](enhanced_batch, high_batch)
    if 'perceptual_loss' in all_losses: 
        l_pixel += all_losses['perceptual_loss'](enhanced_batch, high_batch)
    if 'edge_loss' in all_losses: 
        l_pixel += all_losses['edge_loss'](enhanced_batch, high_batch)
    if 'frequency_loss' in all_losses:
        l_pixel += all_losses['frequency_loss'](enhanced_batch, high_batch)
    if 'enhance_loss' in all_losses and outside_batch is not None:
        l_pixel += all_losses['enhance_loss'](outside_batch, high_batch, scale_factor = scale_factor)    

    return l_pixel


__all__ = ['create_loss', 'calculate_loss', 'SSIM', 'VGGLoss']
