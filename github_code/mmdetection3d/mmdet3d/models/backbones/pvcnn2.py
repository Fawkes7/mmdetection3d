import torch
from mmcv.runner import auto_fp16
from torch import nn as nn


from mmcv.cnn import ConvModule
from mmdet3d.ops import PVConv, PointFPModule, PointSAModule
from mmdet.models import BACKBONES
from .base_pointnet import BasePointNet
from mmdet3d.ops.pvconv.modules import create_pointnet_components, create_pointnet2_sa_components, \
    create_pointnet2_fp_modules, create_mlp_components


@BACKBONES.register_module()
class PVCNN2(BasePointNet):
    def __init__(self, in_channels, sa_blocks=(
                ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
                ((64, 3, 16), (256, 0.2, 32, (64, 128))),
                ((128, 3, 8), (64, 0.4, 32, (128, 256))),
                (None, (16, 0.8, 32, (256, 256, 512))),),
                fp_blocks=(((256, 256), (256, 1, 8)),
                    ((256, 256), (256, 1, 8)),
                    ((256, 128), (128, 2, 16)),
                    ((128, 128, 64), (64, 1, 32)),),
                width_multiplier=1, voxel_resolution_multiplier=1, with_se=True, normalize=True, use_xyz=True):
        super().__init__()
        assert in_channels >= 3
        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=sa_blocks, extra_feature_channels=in_channels - 3, with_se=with_se, normalize=normalize,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier, use_xyz=use_xyz)
        self.sa_layers = nn.ModuleList(sa_layers)

        sa_in_channels[0] = in_channels - 3
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels, with_se=with_se,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)
        self.fp_layers = nn.ModuleList(fp_layers)

    @auto_fp16(apply_to=('points',))
    def forward(self, points):
        xyz, features = self._split_point_feats(points)
        features_input = features
        #print(xyz.shape, features.shape)
        #features = points.transpose(1, 2)
        # batch, num_points = xyz.shape[:2]
        coords_list, in_features_list = [], []
        for sa_blocks in self.sa_layers:
            #print(features.shape)
            in_features_list.append(features)
            coords_list.append(xyz)
            for sa_block in sa_blocks:
                if isinstance(sa_block, ConvModule):
                    features = sa_block(features)
                else:
                    #print(sa_block, xyz, features)
                    xyz, features, _ = sa_block(xyz, features)

        for fp_idx_inv, fp_blocks in enumerate(self.fp_layers):
            fp_idx = len(self.fp_layers) - fp_idx_inv - 1
            for fp_block in fp_blocks:
                if isinstance(fp_block, PointFPModule):
                    #print(fp_idx, coords_list[fp_idx].shape, xyz.shape, in_features_list[fp_idx].shape, features.shape)
                    features = fp_block(coords_list[fp_idx], xyz, in_features_list[fp_idx], features)
                    xyz = coords_list[fp_idx]
                if isinstance(fp_block, ConvModule):
                    features = fp_block(features)
                elif isinstance(fp_block, PVConv):
                    xyz, features, _ = fp_block(xyz, features)

        return features
