from typing import Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
                         
from model.utils import position_encodings
from model.ppnet import PPNet
import prototype.push as push       

from prototype.receptive_field import compute_proto_layer_rf_info_v2




class MultiModal_PPNet(nn.Module):
    def __init__(self, image_ppnet, genetic_ppnet, num_classes, init_weights=True,
                 prototype_activation_function='linear',
                 init_multimodal_weights=True
        ):

        super().__init__()
        
        self.image_net = image_ppnet
        self.genetic_net = genetic_ppnet
        
        self.last_layer = nn.Linear(2 * num_classes, num_classes)
        
        if init_weights:
            self.initialize_weights()
        if init_multimodal_weights:
            with torch.no_grad():
                self.last_layer.weight[:,:] = 0
                print(self.last_layer.weight.shape)
                for i in range(num_classes):
                    self.last_layer.weight[i,i] = 1
                    self.last_layer.weight[i,i+num_classes] = 1
        
        
    def forward(self, x, y):
        img_logit, img_dist = self.image_net(x)
        genetic_logit, genetic_dist = self.genetic_net(y)
        
        combined_logits = torch.cat((img_logit, genetic_logit), dim=1)
        logits = self.last_layer(combined_logits)
        
        return logits, img_dist, genetic_dist
    
    def load_state_dict_img(self, datapath: str):
        pretrained_model = torch.load(datapath)
        
        self.image_net.load_state_dict(pretrained_model.state_dict())
        
    def load_state_dict_genetic(self, datapath: str):
        pretrained_model = torch.load(datapath)
        
        self.genetic_net.load_state_dict(pretrained_model.state_dict())
    
    
    def prune_prototypes(self):
        pass
    
    
    def initialize_weights(self):
        if isinstance(self.last_layer, nn.Linear):
            nn.init.xavier_uniform_(self.last_layer.weight)
            nn.init.constant_(self.last_layer.bias, 0)
        
        
        
    
        