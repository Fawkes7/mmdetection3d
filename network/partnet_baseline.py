import torch
import numpy as np
import os
import torch.nn as nn
import torch.utils.data
from network.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation
from scipy.optimize import linear_sum_assignment
from network import utils

class Baseline(nn.Module):
    def __init__(self, num_part, num_ins, additional_channel=3, weight_decay=0):
        super(Baseline, self).__init__()

        # backbone
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=64, in_channel=3+3,
                                          mlp=[64,64,128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=3+128,
                                          mlp=[128,128,256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=3+256,
                                          mlp=[256,512,1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1024+256, mlp=[256,256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256+128, mlp=[256,128])
        self.fp1 = PointNetFeaturePropagation(in_channel=131+3, mlp=[128,128,128])

        # semantic seg branch
        self.semantic_conv1 = torch.nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, padding=0)
        self.semantic_bn1 = torch.nn.BatchNorm1d(num_features=256)
        self.semantic_conv2 = torch.nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, padding=0)
        self.semantic_bn2 = torch.nn.BatchNorm1d(num_features=256)
        self.semantic_conv3 = torch.nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1, padding=0)
        self.semantic_bn3 = torch.nn.BatchNorm1d(num_features=128)
        self.semantic_conv4 = torch.nn.Conv1d(in_channels=128, out_channels=num_part+1, kernel_size=1, padding=0)

        # instance seg branch
        self.ins_conv1 = torch.nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, padding=0)
        self.ins_bn1 = torch.nn.BatchNorm1d(num_features=256)


        self.ins_conv2 = torch.nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, padding=0)
        self.ins_bn2 = torch.nn.BatchNorm1d(num_features=256)
        self.ins_conv3 = torch.nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1, padding=0)
        self.ins_bn3 = torch.nn.BatchNorm1d(num_features=128)
        self.ins_conv4 = torch.nn.Conv1d(in_channels=128, out_channels=num_ins + 1, kernel_size=1, padding=0)

        self.conf_conv1 = torch.nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=1, padding=0, bias=True)
        self.conf_bn1 = torch.nn.BatchNorm1d(num_features=256)
        self.conf_conv2 = torch.nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, padding=0, bias=True)
        self.conf_bn2 = torch.nn.BatchNorm1d(num_features=256)
        self.conf_conv3 = torch.nn.Conv1d(in_channels=256, out_channels=num_ins, kernel_size=1, padding=0, bias=True)
        self.conf_bn3 = torch.nn.BatchNorm1d(num_features=num_ins)

    def forward(self, pc):
        B, C, N = pc.size()
        l0_xyz, l0_points = pc, pc

        # backbone
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        #print(l1_xyz.size(), l1_points.size())
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        #print(l2_xyz.size(), l2_points.size())
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        #print(l3_xyz.size(), l3_points.size())

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # xyz1, xyz2, points1, points2
        #print(l2_points.size())
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        #print(l1_points.size())
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat((l0_xyz, l0_points), dim=1), l1_points)
        #print(l0_points.size())

        # semantic seg branch
        seg_fm = l0_points
        seg_fm = self.semantic_bn1(self.semantic_conv1(seg_fm))
        seg_fm = self.semantic_bn2(self.semantic_conv2(seg_fm))
        seg_fm = self.semantic_bn3(self.semantic_conv3(seg_fm))
        seg_fm = self.semantic_conv4(seg_fm)   # [B, num+1, N]
        #print(seg_fm.size())

        # instance seg branch
        ins_fm = l0_points
        ins_fm = self.ins_bn1(self.ins_conv1(ins_fm))
        ins_fm = self.ins_bn2(self.ins_conv2(ins_fm))
        ins_fm = self.ins_bn3(self.ins_conv3(ins_fm))
        ins_fm = self.ins_conv4(ins_fm)  # [B, num+1, N]
        ins_fm = torch.softmax(ins_fm, dim=1)
        #print(ins_fm.size())
        ins_mask = ins_fm[:,:-1,:]
        other_mask = ins_fm[:,-1,:]
        #print(ins_mask.size(), other_mask.size())

        # instance seg confidence
        conf_fm = l3_points#.view(B,-1).contiguous()
        conf_fm = self.conf_bn1(self.conf_conv1(conf_fm))
        conf_fm = self.conf_bn2(self.conf_conv2(conf_fm))
        conf_fm = self.conf_bn3(self.conf_conv3(conf_fm)).squeeze(dim=-1)
        conf_fm = torch.sigmoid(conf_fm)
        #print(conf_fm.size())

        return seg_fm, ins_mask, other_mask, conf_fm

def seg_loss(seg_pred, seg_gt, criterion):
    '''
    :param seg_pred: [B, C, N]
    :param seg_gt: [B, N]
    :param criterion: CE loss
    '''
    B, _, N = seg_pred.size()
    loss = criterion(seg_pred, seg_gt)
    loss = loss.sum() / B / N
    return loss

def hugarian_matching(pred, gt, curnmasks):
    '''
    :param pred: [B, nMask, nPoint]
    :param gt: [B, nMask, nPoint]
    :param curnmask: [B]
    :return: matching idx: [B, nMask, 2]
    '''
    B, nMask, nPoint = pred.shape
    matching_score = np.matmul(gt, np.transpose(pred, axes=[0, 2, 1]))  # B x nmask x nmask
    matching_score = 1 - np.divide(matching_score, np.maximum(
        np.expand_dims(np.sum(pred, 2), 1) + np.sum(pred, 2, keepdims=True) - matching_score, 1e-8))
    matching_idx = np.zeros((B, nMask, 2)).astype('int32')
    curnmasks = curnmasks.astype('int32')
    for i, curnmask in enumerate(curnmasks):
        row_ind, col_ind = linear_sum_assignment(matching_score[i, :curnmask, :])
        matching_idx[i, :curnmask, 0] = row_ind
        matching_idx[i, :curnmask, 1] = col_ind
    return matching_idx

def iou(pred, gt, gt_conf, nPoint, nMask):
    '''
    :param pred: [B,K,N]
    :param gt: [B,K,N]
    :param gt_conf: [B,K]
    :param nPoint:
    :param nMask: num_ins
    '''
    matching_idx = hugarian_matching(pred.detach().cpu().numpy(), gt.detach().cpu().numpy(),
                                     gt_conf.sum(dim=-1) / nMask)  # [B, nMask, 2]
    # row
    matching_idx_row = matching_idx[:, :, 0]  # [B,nMask]
    idx = (matching_idx_row >= 0).nonzero()
    matching_idx_row = torch.cat((idx[:, 0].unsqueeze(-1), matching_idx_row.view(-1,1)), dim=1)  # [N,1], [B*nMask, 1] -> [N,2]
    gt_x_matched = torch.gather(gt, matching_idx_row).view(-1, nMask, nPoint)

    # col
    matching_idx_col = matching_idx[:, :, 1]  # [B,nMask]
    idx = (matching_idx_col >= 0).nonzero()
    matching_idx_col = torch.cat((idx[:, 0].unsqueeze(-1), matching_idx_col.view(-1,1)), dim=1)  # [N,1], [B*nMask, 1] -> [N,2]
    pred_x_matched = torch.gather(pred, matching_idx_col).view(-1, nMask, nPoint)

    # mean iou
    matching_score = (gt_x_matched * pred_x_matched).sum(dim=-1) / nPoint
    iou_all = torch.div(matching_score, gt_x_matched.sum(dim=2) + pred_x_matched.sum(dim=2) + 1e-8)
    mean_iou = torch.div((iou_all*gt_conf).sum(1), gt_conf.sum(-1) + 1e-8)
    return mean_iou

def ins_loss(pred, gt, gt_valid):
    '''
    :param pred: [B, K, N]
    :param gt: [B,K,N]
    :param gt_valid: [B,K]
    :return:
    '''
    B, num_ins, num_point = pred.size()
    mean_iou = iou(pred, gt, gt_valid, num_point, num_ins)
    mean_iou = mean_iou.sum() / B
    return -1.0 * mean_iou

def conf_loss(pred, gt_valid, iou_all, matching_idx):
    '''
    :param pred: [B, K]
    :param gt_valid: [B,K]
    :return:
    '''
    B, nMask = pred.size()

    matching_idx_col = matching_idx[:,:,1]
    idx = (matching_idx_col >= 0).nonzero()
    all_indeces = torch.cat((idx[:, 0].unsqueeze(-1), matching_idx_col.view(-1,1)), dim=1)  # [N,1], [B*nMask, 1] -> [N,2]
    all_indeces = all_indeces.view(B, nMask, 2)

    valid_idx = (matching_idx_col >= 0.5).nonzero()
    predicted_indices = torch.gather(all_indeces, valid_idx)
    valid_iou = torch.gather(iou_all, valid_idx)

    conf_target = utils.scatter_nd(predicted_indices, valid_iou, [B, nMask])  # [B,nMask]
    per_part_loss = (pred - conf_target)**2

    target_pos_mask = (conf_target > 0.1)
    target_neg_mask = 1.0 - target_pos_mask

    pos_per_shape_loss = torch.div((target_pos_mask * per_part_loss).sum(-1), torch.max(1e-6, target_pos_mask.sum(-1)))
    neg_per_shape_loss = torch.div((target_neg_mask * per_part_loss).sum(-1), torch.max(1e-6, target_neg_mask.sum(-1)))

    per_shape_loss = pos_per_shape_loss + neg_per_shape_loss
    return per_shape_loss.sum() / B

def other_ins_loss(other_mask_pred, gt_other_mask):
    '''
    :param other_mask_pred: [B,N]
    :param gt_other_mask:
    :return:
    '''
    B, nPoint = other_mask_pred.size()
    matching_score = (other_mask_pred, gt_other_mask).sum(-1)
    iou = torch.div(matching_score, other_mask_pred.sum(-1) + gt_other_mask.sum(-1) - matching_score + 1e-8)
    loss = -1.0 * iou.sum() / B
    return loss

def l21_norm(mask_pred, other_mask_pred):
    '''
    :param mask_pred: [B,K,N]
    :param other_mask_pred: [B,K]
    :return:
    '''
    B, nIns, nPoint = mask_pred.size()
    full_mask = torch.cat((mask_pred, other_mask_pred.unsqueeze(dim=-1).repeat(1,1,nPoint)), dim=1) + 1e-6
    l21_norm = torch.norm(torch.norm(full_mask, p=2, dim=-1), p=2, dim=-1) / nPoint
    return l21_norm.sum() / B


if __name__ == '__main__':
    pc = torch.randn(5, 3, 3000)
    network = Baseline(num_part=50, num_ins=17)
    seg_fm, ins_mask, other_mask, conf_fm = network(pc)
    criterion = torch.nn.CrossEntropyLoss()
    seg_loss = seg_loss(seg_fm, torch.zeros(5,3000).type(torch.LongTensor), criterion)
    print(seg_loss)
