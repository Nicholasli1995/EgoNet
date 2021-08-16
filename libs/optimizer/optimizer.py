"""
Optimization utilities.

Author: Shichao Li
Contact: nicholas.li@connect.ust.hk
"""
import torch

def prepare_optim(model, cfgs):
    """
    Get optimizer and scheduler objects from model parameters.
    """      
    params = [ p for p in model.parameters() if p.requires_grad]
    lr = cfgs['optimizer']['lr']
    weight_decay = cfgs['optimizer']['weight_decay']
    momentum = cfgs['optimizer']['momentum']
    milestones = cfgs['optimizer']['milestones']
    gamma = cfgs['optimizer']['gamma']
    if cfgs['optimizer']['optim_type'] == 'adam':
        optimizer = torch.optim.Adam(params, 
                                     lr = lr, 
                                     weight_decay = weight_decay)
    elif cfgs['optimizer']['optim_type'] == 'sgd':
        optimizer = torch.optim.SGD(params, 
                                    lr = lr, 
                                    momentum = momentum,
                                    weight_decay = weight_decay)
    else:
        raise NotImplementedError
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                     milestones = milestones, 
                                                     gamma = gamma
                                                     )
    # A scheduler that automatically decreases the learning rate
#    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                                                           mode='min',
#                                                           factor=0.5,
#                                                           patience=10,
#                                                           verbose=True,
#                                                           min_lr=0.01)
    return optimizer, scheduler
