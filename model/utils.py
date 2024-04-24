import numpy as np
import torch


def position_encodings(self, x):
    
    """
        Position Encoding Idea:
        We want the dot product of two nearby encodings to be close to 1,
        and the dot product of two faraway encodings to be close to 0.

        We can achieve this by encoding positions into vectors along
        the unit circle between theta=0 and theta = pi/2.

        We append these vectors to the end of the latent space channels.
    """
    
    # x = F.normalize(x, dim=1)
    th = torch.linspace(0, torch.pi /2, x.shape[3], device=x.device)
    pos_1 = torch.cos(th)
    pos_2 = torch.sin(th)

    pos_vec = torch.stack([pos_1, pos_2], dim=0).repeat(x.shape[0], 1, 1).unsqueeze(2)

    return torch.cat([x, pos_vec], dim=1)
    
     
def get_optimizers(cfg, ppnet): 
    
    if cfg.DATASET.NAME == 'multimodal':
        
        last_layer_optimizer_specs = [
            {
                'params': ppnet.last_layer.parameters(), 
                'lr': cfg.OPTIM.LAST_LAYER_OPTIMIZER_LAYERS.LR
            }
        ]
        last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)
        
        return last_layer_optimizer
    
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

nucleotides = {"N": 0, "A": 1, "C": 2, "G": 3, "T": 4}
def decode_onehot(onehot, three_dim=True):
    if three_dim:
        onehot = onehot[:, 0, :]
    # Add another row encoding whether the nucleotide is unknown
    onehot = np.vstack([np.zeros(onehot.shape[1]), onehot])
    # Make the unknown nucleotide 1 if all other nucleotides are 0
    onehot[0] = 1 - onehot[1:].sum(0)
    return "".join([list(nucleotides.keys())[list(nucleotides.values()).index(i)] for i in onehot.argmax(0)])