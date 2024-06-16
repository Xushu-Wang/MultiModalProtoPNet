from typing import Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
                         
from model.utils import position_encodings
from model.ppnet import PPNet


class MultiModal_PPNet(nn.Module):
    def __init__(self, img_features, genetic_features, img_size, genetic_size, img_prototype_shape,
                 genetic_prototype_shape, proto_layer_rf_info, num_classes, init_weights=True,
                 prototype_activation_function='linear',
        ):

        super().__init__()
        
        self.image_net = PPNet(features=img_features,
                 img_size=img_size,
                 prototype_shape=img_prototype_shape,
                 proto_layer_rf_info=proto_layer_rf_info,
                 num_classes=num_classes,
                 init_weights=True,
                 prototype_distance_function='cosine',
                 prototype_activation_function=prototype_activation_function
                )
        
        self.genetic_net = PPNet(features=genetic_features, 
                 img_size=(4, 1, genetic_size),
                 prototype_shape=genetic_prototype_shape,
                 proto_layer_rf_info=None,
                 num_classes=num_classes,
                 init_weights=True,
                 prototype_distance_function='cosine',
                 prototype_activation_function=prototype_activation_function, 
                 genetics_mode=True,
                )
        
        self.last_layer = nn.Linear(2 * num_classes, num_classes)
        
        if init_weights:
            self.initialize_weights()
        
        
    def forward(self, x):
        
        img_logit, img_dist = self.image_net(x)
        genetic_logit, genetic_dist = self.genetic_net(x)
        
        combined_logits = torch.cat((img_logit, genetic_logit), dim=1)
        logits = self.last_layer(combined_logits)
        
        return logits, img_dist, genetic_dist
    
    
    def last_layer(self):
        for p in self.image_net.parameters():
            p.requires_grad = False
        for p in self.genetic_net.parameters():
            p.requires_grad = False
            
            
    def joint(self):
        for p in self.image_net.parameters():
            p.requires_grad = True
        for p in self.genetic_net.parameters():
            p.requires_grad = True
    
    
    def load_state_dict_img(self, datapath: str):
        pretrained_weights = torch.load(datapath)
        self.image_net.load_state_dict(pretrained_weights)
        
    def load_state_dict_genetic(self, datapath: str):
        pretrained_weights = torch.load(datapath)
        self.genetic_net.load_state_dict(pretrained_weights)
    
    def push_forward(self):
        pass
    
    
    def prune_prototypes(self):
        pass
    
    
    def initialize_weights(self):
        if isinstance(self.last_layer, nn.Linear):
            nn.init.xavier_uniform_(self.last_layer.weight)
            nn.init.constant_(self.last_layer.bias, 0)
        
        
        
    
        