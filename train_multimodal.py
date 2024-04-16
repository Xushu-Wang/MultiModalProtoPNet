import time
import torch
from utils.util import list_of_distances

import torch.nn
import torch.nn.functional as F



def _train_or_test_multimodal(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print):
    
    '''
    model: the multi-gpu model
    dataloader: 
    optimizer: if None, will be test evaluation
    '''
    
    is_train = optimizer is not None
    start = time.time()
    
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    
    total_cluster_cost = 0
    
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0

    class_correct_counts = None
    class_guess_counts = None

    for i, (image, genetics, label) in enumerate(dataloader):
        image = image.cuda()
        genetics = genetics.cuda()
        target = label.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, min_img_distances, min_genetic_distances = model(image, genetics)

            # compute loss
            cross_entropy = F.cross_entropy(output, target)

            # if class_specific:
            #     max_dist = (model.module.prototype_shape[1]
            #                 * model.module.prototype_shape[2]
            #                 * model.module.prototype_shape[3])

            #     # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
            #     # calculate cluster cost
            #     prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()
            #     inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
            #     cluster_cost = torch.mean(max_dist - inverted_distances)

            #     # calculate separation cost
            #     prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            #     inverted_distances_to_nontarget_prototypes, _ = \
            #         torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
            #     separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

            #     # calculate avg cluster cost
            #     avg_separation_cost = \
            #         torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
            #     avg_separation_cost = torch.mean(avg_separation_cost)
                
            #     if use_l1_mask:
            #         l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
            #         l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
            #     else:
            #         l1 = model.module.last_layer.weight.norm(p=1) 

            # else:
            #     min_distance, _ = torch.min(min_distances, dim=1)
            #     cluster_cost = torch.mean(min_distance)
            #     l1 = model.module.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            # total_cluster_cost += cluster_cost.item()
            # total_separation_cost += separation_cost.item()
            # total_avg_separation_cost += avg_separation_cost.item()

            # Calculate Balanced Accuracy
            if class_correct_counts == None:
                class_correct_counts = torch.zeros(output.shape[1], device=output.device)
                class_guess_counts = torch.zeros(output.shape[1], device=output.device)

            for i in range(output.shape[1]):
                class_correct_counts[i] += torch.sum((predicted == target) & (target == i))
                class_guess_counts[i] += torch.sum(target == i)

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                loss = cross_entropy
            else:
                loss = cross_entropy
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del min_img_distances
        del min_genetic_distances
        del predicted

    end = time.time()

    log('\ttime: \t{0}'.format(end -  start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    # log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    # if class_specific:
    #     log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
    #     log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    # log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    # log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    # p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    # with torch.no_grad():
    #     p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    # log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))
    log('\tbalanced accu:\t{0}'.format(torch.mean(class_correct_counts / class_guess_counts).item()))
    log(f'\tmode: \t{"train" if is_train else "test"}')

    return n_correct / n_examples


def train_multimodal(model, dataloader, optimizer, class_specific=False, coefs=None, log=print):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    return _train_or_test_multimodal(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)


def test_multimodal(model, dataloader, class_specific=False, log=print):
    log('\ttest')
    model.eval()
    return _train_or_test_multimodal(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log)


