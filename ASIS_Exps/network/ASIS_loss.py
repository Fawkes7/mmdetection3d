import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from network.ASIS_utils import unsorted_segment_sum

def get_loss(pred, ins_label, pred_sem_logit, sem_label, criterion):
    '''
    :param pred: predicted per-point embedding in [B,E,N], E=5
    :param ins_label: [B,N]
    :param pred_sem_logit: [B,C,N]
    :param sem_label: [B,N]
    :return:
    '''
    delta_v, delta_d = 0.5, 1.5
    param_var, param_dist, param_reg = 1., 1., 0.001

    sem_seg_loss = criterion(pred_sem_logit, sem_label)
    instance_seg_loss, l_var, l_dist, l_reg = discriminative_loss(pred, ins_label, pred.size(1),
                                                          delta_v, delta_d, param_var, param_dist, param_reg)
    return sem_seg_loss, instance_seg_loss, l_var, l_dist, l_reg


def discriminative_loss(pred, ins_label, dim=5,
                        delta_v=0.5, delta_d=1.5, param_var=1.0, param_dist=1.0, param_reg=0.001):
    l_disc_list, l_var_list, l_dist_list, l_reg_list = [], [], [], []
    B = pred.size(0)

    for i in range(B):
        l_disc, l_var, l_dist, l_reg = discriminative_loss_single(pred[i], ins_label[i], dim,
                                                                  delta_v, delta_d, param_var, param_dist, param_reg)
        l_disc_list.append(l_disc)
        l_var_list.append(l_var)
        l_dist_list.append(l_dist)
        l_reg_list.append(l_reg)

    out_l_disc = torch.stack(l_disc_list).sum() / B
    out_l_var = torch.stack(l_var_list).sum() / B
    out_l_dist = torch.stack(l_dist_list).sum() / B
    out_l_reg = torch.stack(l_reg_list).sum() / B
    return out_l_disc, out_l_var, out_l_dist, out_l_reg


def discriminative_loss_single(pred, ins_label, dim,
                               delta_v, delta_d, param_var, param_dist, param_reg):
    '''
    :param pred: [E,N]
    :param ins_label: [N]
    :param dim: E=5
    :param delta_v: cutoff variance distance
    :param delta_d: cutoff cluster distance
    :param param_var: weight for intra cluster variance
    :param param_dist: weight for inter cluster distances
    :param param_reg: weight regularization
    :return:
    '''
    E, N = pred.size()
    pred = pred.permute(1,0)  # [N,E]

    unique_labels, unique_idx, counts = torch.unique(ins_label, sorted=True, return_inverse=True, return_counts=True)  # [C], [N], [C]
    num_ins = unique_labels.size(0)  # C
    counts = counts.view(-1,1)  # [C,1]

    #ins_sum = torch.zeros(num_ins, dim).scatter_add_(dim=0, index=unique_idx, src=pred).to(pred.device)
    ins_sum = unsorted_segment_sum(pred, unique_idx, num_ins)
    # [N,E] selected by [N] into C groups, return [C,E]

    mu = ins_sum / counts  # averaged representation for each instance in [C,E]
    mu_expand = mu[unique_idx]  # expanded averaged instance representation in [N,E]

    # l_var, intra-class variance loss
    distance = torch.abs(pred - mu_expand).sum(dim=1)
    #distance = torch.norm(pred - mu_expand, p=1, dim=1)
    distance -= delta_v
    distance = torch.clamp_min(distance, 0.)
    distance = distance ** 2  # [N,E]
    l_var = unsorted_segment_sum(distance, unique_idx, num_ins)
    l_var = (l_var / counts).sum() / num_ins

    # l_dist, inter-class distance loss
    # Get distance for each pair of clusters like this (example when N=3):
    #   mu_1 - mu_1, mu_2 - mu_1, mu_3 - mu_1
    #   mu_1 - mu_2, mu_2 - mu_2, mu_3 - mu_2
    #   mu_1 - mu_3, mu_2 - mu_3, mu_3 - mu_3
    mu_interleaved_rep = mu.repeat(num_ins, 1)  # [C*C,E]
    mu_brand_rep = mu.repeat(1, num_ins)  # [C,E*C]
    mu_brand_rep = mu_brand_rep.reshape(num_ins * num_ins, dim)  # [C*C,E]
    mu_diff = mu_brand_rep - mu_interleaved_rep   # [C*C,E]
    mu_diff_norm = torch.norm(mu_diff, p=1, dim=1)
    mu_diff_norm = torch.clamp_min(2.0 * delta_d - mu_diff_norm, 0.)
    l_dist = (mu_diff_norm ** 2).sum() / num_ins / (num_ins-1)

    # l_reg
    l_reg = torch.norm(mu, p=1, dim=1).sum() / num_ins

    param_scale = 1.
    l_var = param_var * l_var
    l_dist = param_dist * l_dist
    l_reg = param_reg * l_reg

    #print(l_var, l_dist, l_reg)
    loss = param_scale * (l_var + l_dist + l_reg)
    return loss, l_var, l_dist, l_reg





