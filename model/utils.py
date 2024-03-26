import torch
    
     
def get_optimizers(cfg, ppnet): 
    
    joint_optimizer_specs = [
        {
            'params': ppnet.features.parameters(), 
            'lr': cfg.OPTIM.JOINT_OPTIMIZER_LAYERS.FEATURES, 
            'weight_decay': cfg.OPTIM.JOINT_OPTIMIZER_LAYERS.WEIGHT_DECAY
        }, # bias are now also being regularized
        {
            'params': ppnet.add_on_layers.parameters(), 
            'lr': cfg.OPTIM.JOINT_OPTIMIZER_LAYERS.ADD_ON_LAYERS,
            'weight_decay': cfg.OPTIM.JOINT_OPTIMIZER_LAYERS.WEIGHT_DECAY
        },
        {
            'params': ppnet.prototype_vectors, 
            'lr': cfg.OPTIM.JOINT_OPTIMIZER_LAYERS.PROTOTYPE_VECTORS
        },
        ]
    
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=cfg.OPTIM.JOINT_OPTIMIZER_LAYERS.LR_STEP_SIZE, gamma=0.1)

    warm_optimizer_specs = [
        {
            'params': ppnet.add_on_layers.parameters(), 
            'lr': cfg.OPTIM.WARM_OPTIMIZER_LAYERS.ADD_ON_LAYERS,
            'weight_decay': cfg.OPTIM.WARM_OPTIMIZER_LAYERS.WEIGHT_DECAY
        },
        {'params': ppnet.prototype_vectors, 
         'lr': cfg.OPTIM.WARM_OPTIMIZER_LAYERS.PROTOTYPE_VECTORS,
         },
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    last_layer_optimizer_specs = [
        {
            'params': ppnet.last_layer.parameters(), 
            'lr': cfg.OPTIM.LAST_LAYER_OPTIMIZER_LAYERS.LR
        }
    ]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    return joint_optimizer, joint_lr_scheduler, warm_optimizer, last_layer_optimizer