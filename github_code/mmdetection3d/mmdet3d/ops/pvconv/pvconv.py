import torch.nn as nn, torch

from mmdet3d.ops import trilinear_devoxelize, avg_voxelize
#from .shared_mlp import SharedMLP
from mmcv.cnn import ConvModule
from .se import SE3d


class AvgVoxelization(nn.Module):
    def __init__(self, resolution, normalize=True, eps=0):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps

    def forward(self, features, coords):
        coords = coords.detach()
        norm_coords = coords - coords.mean(2, keepdim=True)
        if self.normalize:
            norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
        else:
            norm_coords = (norm_coords + 1) / 2.0
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
        vox_coords = torch.round(norm_coords).to(torch.int32)
        return avg_voxelize(features, vox_coords, self.r), norm_coords

    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')


class PVConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, with_se=False, normalize=True, eps=0,
                 use_xyz=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.use_xyz = use_xyz
        if use_xyz:
            in_channels += 3

        assert in_channels > 0
        self.voxelization = AvgVoxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
         ]
        if with_se:
            voxel_layers.append(SE3d(out_channels))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.point_features = ConvModule(in_channels, out_channels, kernel_size=(1, ), stride=(1, ),
                                         conv_cfg=dict(type='Conv1d'), norm_cfg=dict(type='BN1d'))

    def forward(self, points_xyz, features, indices=None, target_xyz=None):
        # point_xyz: B * N * 3
        # features: B * C * N
        xyz = points_xyz.transpose(1, 2)

        if self.use_xyz:
            features = xyz if features is None else torch.cat([xyz, features], dim=1)
        voxel_features, voxel_coords = self.voxelization(features, xyz)
        voxel_features = self.voxel_layers(voxel_features)
        voxel_features = trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        point_features = self.point_features(features)
        #print(voxel_features.shape, point_features.shape)
        fused_features = voxel_features + self.point_features(features)
        return points_xyz, fused_features, indices
