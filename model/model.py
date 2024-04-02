import torch

from model.genetics_features import GeneticCNN2D
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


def construct_genetic_ppnet(length:int, num_classes:int, prototype_shape, model_path:str, prototype_activation_function='log', use_cosine=True):
    m = GeneticCNN2D(length, num_classes, include_connected_layer=False, remove_last_layer=False)

    # Remove the fully connected layer
    weights = torch.load(model_path)

    for k in list(weights.keys()):
        if "conv" not in k:
            del weights[k]
    
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
        )

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