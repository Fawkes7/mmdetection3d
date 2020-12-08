import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from network.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation
from scipy.optimize import linear_sum_assignment
from network.ASIS_utils import *
from network.ASIS_loss import *

class ASIS(nn.Module):
    def __init__(self, num_class, additional_channel=3, weight_decay=0):
        super(ASIS, self).__init__()


        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.1, nsample=32, in_channel=3+3,
                                          mlp=[32,32,64], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=32, in_channel=3+64,
                                          mlp=[64,64,128], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=0.4, nsample=32, in_channel=3+128,
                                          mlp=[128,128,256], group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=16, radius=0.8, nsample=32, in_channel=3+256,
                                          mlp=[256,256,512], group_all=False)

        # semantic seg
        self.sem_fp4 = PointNetFeaturePropagation(in_channel=512+256, mlp=[256, 256])
        self.sem_fp3 = PointNetFeaturePropagation(in_channel=256+128, mlp=[256, 256])
        self.sem_fp2 = PointNetFeaturePropagation(in_channel=256+64, mlp=[256, 128])
        self.sem_fp1 = PointNetFeaturePropagation(in_channel=128+3, mlp=[128,128,128])
        self.sem_fc1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.sem_bn1 = nn.BatchNorm1d(num_features=128)
        self.sem_fc2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.sem_bn2 = nn.BatchNorm1d(num_features=128)
        self.sem_fc2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.sem_bn2 = nn.BatchNorm1d(num_features=128)
        self.sem_fc3 = nn.Conv1d(in_channels=128, out_channels=num_class, kernel_size=1, stride=1, padding=0)

        # instance seg
        self.ins_fp4 = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256, 256])
        self.ins_fp3 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 256])
        self.ins_fp2 = PointNetFeaturePropagation(in_channel=256 + 64, mlp=[256, 128])
        self.ins_fp1 = PointNetFeaturePropagation(in_channel=128 + 3, mlp=[128, 128, 128])
        self.ins_fc1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride=1)
        self.ins_bn1 = nn.BatchNorm1d(num_features=128)
        self.ins_fc2 = nn.Conv1d(in_channels=128, out_channels=5, kernel_size=1, stride=1, padding=0)
        #self.ins_bn2 = nn.BatchNorm2d(num_features=5)

        self.k = 30

    def forward(self, pc):
        B, C, N = pc.size()
        l0_xyz, l0_points = pc, pc

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # print(l1_xyz.size(), l1_points.size())
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # print(l2_xyz.size(), l2_points.size())
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # print(l3_xyz.size(), l3_points.size())
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        # print(l4_xyz.size(), l4_points.size())

        # sem seg
        l3_points_sem = self.sem_fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        # print(l3_points_sem.size())
        l2_points_sem = self.sem_fp3(l2_xyz, l3_xyz, l2_points, l3_points_sem)
        # print(l2_points_sem.size())
        l1_points_sem = self.sem_fp2(l1_xyz, l2_xyz, l1_points, l2_points_sem)
        # print(l1_points_sem.size())
        l0_points_sem = self.sem_fp1(l0_xyz, l1_xyz, l0_points, l1_points_sem)
        # print(l0_points_sem.size())
        fms_sem = self.sem_bn1(self.sem_fc1(l0_points_sem))
        # print(fms_sem.size())
        fms_sem_cache = self.sem_bn2(self.sem_fc2(fms_sem))
        print(fms_sem_cache.size())

        # ins seg
        l3_points_ins = self.ins_fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        # print(l3_points_ins.size())
        l2_points_ins = self.ins_fp3(l2_xyz, l3_xyz, l2_points, l3_points_ins)
        # print(l2_points_ins.size())
        l1_points_ins = self.ins_fp2(l1_xyz, l2_xyz, l1_points, l2_points_ins)
        # print(l1_points_ins.size())
        l0_points_ins = self.ins_fp1(l0_xyz, l1_xyz, l0_points, l1_points_ins)
        # print(l0_points_ins.size())
        fms_ins = self.ins_bn1(self.ins_fc1(l0_points_ins))
        fms_ins += fms_sem_cache
        fms_ins = F.dropout(fms_ins, p=0.5, training=self.training)
        fms_ins = self.ins_fc2(fms_ins)  # [B,5,N]
        print(fms_ins.size())

        # fusion
        adj_matrix = pairwise_distance(fms_ins)
        nn_idx = knn_thre(adj_matrix, k=self.k).detach()
        print(nn_idx.size())

        fms_sem = get_local_feature(fms_sem, nn_idx=nn_idx, k=self.k)  # [B,C,K,N]
        fms_sem = torch.max(fms_sem, dim=-2, keepdim=False)[0]
        fms_sem = F.dropout(fms_sem, p=0.5, training=self.training).permute(0,2,1)
        fms_sem = self.sem_fc3(fms_sem)

        return fms_sem, fms_ins



if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    net = ASIS(num_class=12)
    pc = torch.randn((5,3,100), requires_grad=True)
    fm_sem, fm_ins = net(pc)
    print(fm_sem.size(), fm_ins.size())
    criterion = torch.nn.CrossEntropyLoss().to(fm_sem.device)
    gt = torch.randint(low=0, high=6, size=(5,100))
    gt_sem = torch.randint(low=0, high=12, size=(5,100))
    sem_loss, ins_loss, _, _, _ = get_loss(fm_ins, gt, fm_sem, gt_sem, criterion)  # pred, ins_label, pred_sem_logit, sem_label, criterion
    loss = sem_loss + ins_loss
    loss.backward()
    print(loss)