import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

class PPNet(nn.Module):
    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck',
                 genetics_mode=False, 
                 use_cosine=False,
                 init_with_last_layer=False,
                 last_layer_weights=None,
                 fix_prototypes=False,
        ):
        
        super(PPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4
        self.use_cosine = use_cosine
        self.init_with_last_layer = init_with_last_layer
        self.last_layer_weights = last_layer_weights
        self.fix_prototypes = fix_prototypes

        if self.init_with_last_layer:
            raise NotImplementedError("init_with_last_layer not supported")
            assert(last_layer_weights != None)

        if self.fix_prototypes:
            if self.prototype_shape[3] != 1:
                raise NotImplementedError("Fix_prototypes only supported for 1x1 prototypes")

        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        # Ensure that we're using linear with cosine similarity
        assert(not (use_cosine and prototype_activation_function != "linear"))

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        assert(self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        self.proto_layer_rf_info = proto_layer_rf_info

        # this has to be named features to allow the precise loading
        self.features = features

        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        elif genetics_mode:
            first_add_on_layer_in_channels = [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            if self.use_cosine:
                self.add_on_layers = nn.Sequential()
            else:
                proto_depth = self.prototype_shape[1]
                self.add_on_layers = nn.Sequential(
                    nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=proto_depth, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=proto_depth, out_channels=proto_depth, kernel_size=1),
                    nn.Sigmoid()
                    )
        
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)

        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False) # do not use bias

        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(x)
        x = self.add_on_layers(x)

        return x

    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        '''
        input of shape N * c * h * w
        filter of shape P * c * h1 * w1
        weight of shape P * c * h1 * w1
        '''
        input2 = input ** 2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)

        filter2 = filter ** 2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)

        weighted_filter = filter * weights
        weighted_inner_product = F.conv2d(input=input, weight=weighted_filter)

        # use broadcast
        intermediate_result = \
            - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = self.prototype_vectors ** 2

        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def prototype_distances(self, x):
        '''
        x is the raw input
        '''
        distances = self._l2_convolution(x)
        return distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def forward(self, x):
        conv_features = self.conv_features(x)
        if self.use_cosine:
            similarity = self.cosine_similarity(conv_features)
            max_similarities = F.max_pool2d(similarity,
                            kernel_size=(similarity.size()[2],
                                        similarity.size()[3]))
            min_distances = -1 * max_similarities
        else:
            distances = self.prototype_distances(conv_features)
            '''
            we cannot refactor the lines below for similarity scores
            because we need to return min_distances
            '''
            # global min pooling
            min_distances = -F.max_pool2d(-distances,
                                        kernel_size=(distances.size()[2],
                                                    distances.size()[3]))
        min_distances = min_distances.view(-1, self.num_prototypes)
        prototype_activations = self.distance_2_similarity(min_distances)
        logits = self.last_layer(prototype_activations)
        return logits, min_distances

    def find_offsetting_tensor(self, x, normalized_prototypes):
        """
        This finds the tensor used to offset each prototype to a different spatial location.
        """

        # TODO - This should really only be done once on initialization.
        # This is a major waste of time
        arange1 = torch.arange(normalized_prototypes.shape[0] // self.num_classes).view((normalized_prototypes.shape[0]  // self.num_classes, 1)).repeat((1, normalized_prototypes.shape[0]  // self.num_classes))
        indices = torch.LongTensor(torch.arange(normalized_prototypes.shape[0]  // self.num_classes))
        arange2 = (arange1 - indices) % (normalized_prototypes.shape[0]  // self.num_classes)
        arange3 = torch.arange(normalized_prototypes.shape[0]  // self.num_classes, x.shape[3])
        arange3 = arange3.view((1, x.shape[3] - normalized_prototypes.shape[0]  // self.num_classes))
        arange3 = arange3.repeat((normalized_prototypes.shape[0]  // self.num_classes, 1))
        
        arange4 = torch.concatenate((arange2, arange3), dim=1)
        arange4 = arange4.unsqueeze(1).unsqueeze(1)
        arange4 = arange4.repeat((1, x.shape[1], x.shape[2], 1))

        arange4 = arange4.repeat((40,1,1,1))
        arange4 = arange4.to(x.device)

        return arange4

    def cosine_similarity(self, x):
        sqrt_dims = (self.prototype_shape[2] * self.prototype_shape[3]) ** .5
        x_norm = F.normalize(x, dim=1) / sqrt_dims
        normalized_prototypes = F.normalize(self.prototype_vectors, dim=1) / sqrt_dims

        if self.fix_prototypes:
            offsetting_tensor = self.find_offsetting_tensor(x, normalized_prototypes)
            normalized_prototypes = F.pad(normalized_prototypes, (0, x.shape[3] - normalized_prototypes.shape[3], 0, 0))
            normalized_prototypes = torch.gather(normalized_prototypes, 3, offsetting_tensor)

        return F.conv2d(x_norm, normalized_prototypes)

    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        # Possibly better to go through and change push with this similarity metric
        conv_output = self.conv_features(x)
        if self.use_cosine:
            similarities = self.cosine_similarity(conv_output)
            distances = -1 * similarities
        else:
            distances = self._l2_convolution(conv_output)
        return conv_output, distances

    def prune_prototypes(self, prototypes_to_prune):
        '''
        prototypes_to_prune: a list of indices each in
        [0, current number of prototypes - 1] that indicates the prototypes to
        be removed
        '''
        prototypes_to_keep = list(set(range(self.num_prototypes)) - set(prototypes_to_prune))

        self.prototype_vectors = nn.Parameter(self.prototype_vectors.data[prototypes_to_keep, ...],
                                              requires_grad=True)

        self.prototype_shape = list(self.prototype_vectors.size())
        self.num_prototypes = self.prototype_shape[0]

        # changing self.last_layer in place
        # changing in_features and out_features make sure the numbers are consistent
        self.last_layer.in_features = self.num_prototypes
        self.last_layer.out_features = self.num_classes
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

        # self.ones is nn.Parameter
        self.ones = nn.Parameter(self.ones.data[prototypes_to_keep, ...],
                                 requires_grad=False)
        # self.prototype_class_identity is torch tensor
        # so it does not need .data access for value update
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]

    def __repr__(self):
        # PPNet(self, features, img_size, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            '\tproto_layer_rf_info: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.img_size,
                          self.prototype_shape,
                          self.proto_layer_rf_info,
                          self.num_classes,
                          self.epsilon)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)