import torch
from mmcv.runner import auto_fp16
from torch import nn as nn

from mmcv.cnn import ConvModule
from .base_pointnet import BasePointNet
from mmdet.models import BACKBONES

from mmdet3d.ops.pvconv.modules import create_pointnet_components, create_pointnet2_sa_components, \
    create_pointnet2_fp_modules, create_mlp_components


@BACKBONES.register_module()
class PVCNN(BasePointNet):
    def __init__(self, in_channels, blocks=((64, 1, 32), (64, 2, 16), (128, 1, 16), (1024, 1, None)),
                 width_multiplier=1, voxel_resolution_multiplier=1, with_se=False, normalize=True,
                 dense_connections=False, use_global_features=False, use_xyz=True):
        super().__init__()
        assert in_channels >= 3
        self.dense_connections = dense_connections
        self.use_global_features = use_global_features
        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=blocks, in_channels=in_channels - 3, with_se=with_se, normalize=normalize,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,
            use_xyz=use_xyz
        )
        self.point_features = nn.ModuleList(layers)
        layers, channels_cloud = create_mlp_components(
            in_channels=channels_point, out_channels=[256, 128],
            classifier=False, dim=1, width_multiplier=width_multiplier)
        self.cloud_features = nn.Sequential(*layers)

    @auto_fp16(apply_to=('points',))
    def forward(self, points):
        xyz, features = self._split_point_feats(points)
        #features = points.transpose(1, 2)
        batch, num_points = xyz.shape[:2]
        out_features_list = []
        for i in range(len(self.point_features)):
            if isinstance(self.point_features[i], ConvModule):
                features = self.point_features[i](features)
            else:
                xyz, features, _ = self.point_features[i](xyz, features)
            if self.dense_connections or i == len(self.point_features) - 1:
                out_features_list.append(features)
        if self.use_global_features:
            global_features = self.cloud_features(features.max(dim=-1, keepdim=False).values)
            out_features_list.append(global_features.unsqueeze(-1).repeat([1, 1, num_points]))
        return torch.cat(out_features_list, dim=1)
