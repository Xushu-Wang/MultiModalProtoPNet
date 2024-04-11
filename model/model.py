import torch

from model.features.genetics_features import GeneticCNN2D
from model.ppnet import PPNet
from prototype.receptive_field import compute_proto_layer_rf_info_v2
from model.multimodal_ppnet import MultiModal_PPNet

from model.features.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from model.features.densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from model.features.vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features


base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}




def construct_image_ppnet(base_architecture, pretrained=True, img_size=224,
                    prototype_shape=(2000, 512, 1, 1), num_classes=200,
                    prototype_activation_function='log',
                    add_on_layers_type='bottleneck'):
    
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])
    return PPNet(features=features,
                 img_size=img_size,
                 prototype_shape=prototype_shape,
                 proto_layer_rf_info=proto_layer_rf_info,
                 num_classes=num_classes,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 add_on_layers_type=add_on_layers_type)


def construct_genetic_ppnet(length:int, num_classes:int, prototype_shape, model_path:str, prototype_activation_function='log', use_cosine=True, init_with_last_layer=False, fix_prototypes=True):
    m = GeneticCNN2D(length, num_classes, include_connected_layer=False, remove_last_layer=init_with_last_layer)

    # Remove the fully connected layer
    weights = torch.load(model_path)

    for k in list(weights.keys()):
        if "conv" not in k:
            del weights[k]
    
    if init_with_last_layer:
        last_layer_weights = weights['conv3.weight']
        print(last_layer_weights.shape)
    else:
        last_layer_weights = None

    m.load_state_dict(weights)

    return PPNet(features=m, 
                 img_size=(4, 1, length), 
                 prototype_shape=prototype_shape,
                 proto_layer_rf_info=None, 
                 num_classes=num_classes,
                 init_weights=True, 
                 prototype_activation_function="linear", 
                 add_on_layers_type=None, 
                 genetics_mode=True, 
                 use_cosine=True,
                 init_with_last_layer=init_with_last_layer,
                 last_layer_weights=last_layer_weights,
                 fix_prototypes=fix_prototypes
    )
    
    
    
def construct_multimodal_ppnet(cfg):
    
    
    
    
    pass

def construct_ppnet(cfg): 
    if cfg.DATASET.NAME == "cub" or cfg.DATASET.NAME == 'bioscan': 
        return construct_image_ppnet(
            base_architecture=cfg.MODEL.BACKBONE,
            pretrained=True,
            img_size=cfg.DATASET.IMAGE_SIZE, 
            prototype_shape=cfg.MODEL.PROTOTYPE_SHAPE, 
            num_classes=cfg.DATASET.NUM_CLASSES, 
            prototype_activation_function=cfg.MODEL.PROTOTYPE_ACTIVATION_FUNCTION,
            add_on_layers_type=cfg.MODEL.ADD_ON_LAYERS_TYPE
        ).to(cfg.MODEL.DEVICE)
    elif cfg.DATASET.NAME == "genetics":
        if not cfg.MODEL.BACKBONE:
            raise ValueError("Model path not provided for genetics dataset (--backbone)")
        return construct_genetic_ppnet(
            length=cfg.DATASET.BIOSCAN.CHOP_LENGTH, 
            num_classes=cfg.DATASET.NUM_CLASSES, 
            prototype_shape=cfg.MODEL.PROTOTYPE_SHAPE, 
            model_path=cfg.MODEL.BACKBONE, 
            prototype_activation_function=cfg.MODEL.PROTOTYPE_ACTIVATION_FUNCTION, 
            use_cosine=cfg.MODEL.USE_COSINE,
            init_with_last_layer=cfg.MODEL.INIT_PPNET_WITH_LAST_LAYER,
            fix_prototypes=cfg.MODEL.FIX_PROTOTYPES
        ).to(cfg.MODEL.DEVICE)
    elif cfg.DATASET.NAME == "multimodal":
        pass
    else: 
        raise NotImplementedError