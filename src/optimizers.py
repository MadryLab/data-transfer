import numpy as np
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR, StepLR


class LRPolicy():
    def __init__(self, lr_schedule):
        self.lr_schedule = lr_schedule
    def __call__(self, epoch):
        return self.lr_schedule[epoch]

def get_optimizer_and_lr_scheduler(training_args, model, iters_per_epoch=1):
    scheduler_type = training_args['lr_scheduler']

    optimizer = SGD(model.parameters(), 
            lr=training_args['lr'], 
            weight_decay=training_args['weight_decay'], 
            momentum=training_args['momentum']
            )

    if scheduler_type == 'constant':
        scheduler = None
    elif scheduler_type == 'steplr':
        scheduler = StepLR(optimizer, step_size=training_args['step_size'], 
                                        gamma=training_args['gamma'])
    elif scheduler_type == 'multisteplr':
        scheduler = MultiStepLR(optimizer, milestones=training_args['lr_milestones'], 
                                        gamma=training_args['gamma'])    
    elif scheduler_type == 'cyclic':
        epochs = training_args['epochs']
        lr_peak_epoch = training_args['lr_peak_epoch']
        
        lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                        [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                        [0, 1, 0])
        scheduler = LambdaLR(optimizer, LRPolicy(lr_schedule)) 
    else:
        raise NotImplementedError("Unimplemented LR Scheduler Type")

    return optimizer, scheduler


