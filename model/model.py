import torch

from model.genetics_features import GeneticCNN
from model.ppnet import PPNet, base_architecture_to_features
from prototype.receptive_field import compute_proto_layer_rf_info_v2


def construct_ppnet(base_architecture, pretrained=True, img_size=224,
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


def construct_genetic_ppnet(length:int=720, num_classes:int=10, prototype_shape:int=(600, 24, 1, 1), model_path:str=None,prototype_activation_function='log', use_cosine:bool=False):
    m = GeneticCNN(length, num_classes, two_dimensional=True)

    # Remove the fully connected layer
    weights = torch.load(model_path)
    for k in list(weights.keys()):
        if "conv" not in k:
            del weights[k]
    m.load_state_dict(weights)

    return PPNet(m, (4, length), prototype_shape, None, num_classes, False, prototype_activation_function, None, True, use_cosine=use_cosine)



def construct_ppnet(cfg): 
    if cfg.DATASET.NAME == "cub": 
        return construct_ppnet(
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
            use_cosine=cfg.MODEL.USE_COSINE
        ).to(cfg.MODEL.DEVICE)
    else: 
        raise NotImplementedError