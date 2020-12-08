import mmdet3d.models.backbones
from mmdet.models import BACKBONES
from mmdet.apis.train import set_random_seed
from torch_geometric.datasets import ShapeNet
import torch.nn as nn, torch, pointnet2, h5py, pytorch_lightning as pl, os.path as osp, os
from torch.utils.data import DataLoader
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG


class Pointnet2_SSG_CLS_MM3D(PointNet2ClassificationSSG):
    def _build_model(self):
        self.backbone = BACKBONES.module_dict['PointNet2SASSG'](6, num_points=(512, 128, None),
                                                                radius=(0.2, 0.4, None),
                                                                num_samples=(64, 64, None),
                                                                sa_channels=(
                                                                    (64, 64, 128), (128, 128, 256), (256, 512, 1024)),
                                                                fp_channels=(),
                                                                sa_cfg=dict(
                                                                    type='PointSAModule',
                                                                    pool_mod='max',
                                                                    use_xyz=True,
                                                                    normalize_xyz=False))
        self.fc = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 40),
        )

    def forward(self, pc):
        all = self.backbone(pc)
        features = all['fp_features'][0][..., 0]
        predict = self.fc(features)
        return predict


class Pointnet2_MSG_CLS_MM3D(Pointnet2_SSG_CLS_MM3D):
    def _build_model(self):
        self.backbone = BACKBONES.module_dict['PointNet2SAMSG'](6, num_points=(512, 128, None),
                                                                radii=((0.1, 0.2, 0.4), (0.2, 0.4, 0.8), ()),
                                                                num_samples=((16, 32, 128), (32, 64, 128), ()),
                                                                sa_channels=(
                                                                ((32, 32, 64), (64, 64, 128), (64, 96, 128)),
                                                                ((64, 64, 128), (128, 128, 256), (128, 128, 256)),
                                                                (256, 512, 1024)),
                                                                dilated_group=(False, False, False),
                                                                fps_mods=(('D-FPS'), ('D-FPS'), ('D-FPS')),
                                                                fps_sample_range_lists=((-1), (-1), (-1)),
                                                                aggregation_channels=(None, None, None),
                                                                fp_channels=()
                                                                )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 40),
        )


mode = 'msg'

if mode == 'ssg':
    hparams = dict()
    hparams['batch_size'] = 24
    hparams['num_points'] = 4096
    hparams['optimizer.weight_decay'] = 0.0
    hparams['optimizer.lr'] = 1e-3
    hparams['optimizer.lr_decay'] = 0.7
    hparams['optimizer.bn_momentum'] = 0.5
    hparams['optimizer.bnm_decay'] = 0.5
    hparams['optimizer.decay_step'] = 2e4
    model = Pointnet2_SSG_CLS_MM3D(hparams).cuda()

    early_stop_callback = pl.callbacks.EarlyStopping(patience=5)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=2,
        filepath=os.path.join("mmpn2-ssg-cls", "{epoch}-{val_loss:.2f}-{val_acc:.3f}"),
        verbose=True,
    )
    trainer = pl.Trainer(
        gpus=[0, ],
        max_epochs=200,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        distributed_backend='dp',
    )
    trainer.fit(model)
elif mode == 'msg':
    hparams = dict()
    hparams['batch_size'] = 16
    hparams['num_points'] = 4096
    hparams['optimizer.weight_decay'] = 0.0
    hparams['optimizer.lr'] = 1e-3
    hparams['optimizer.lr_decay'] = 0.7
    hparams['optimizer.bn_momentum'] = 0.5
    hparams['optimizer.bnm_decay'] = 0.5
    hparams['optimizer.decay_step'] = 2e4
    model = Pointnet2_MSG_CLS_MM3D(hparams).cuda()

    early_stop_callback = pl.callbacks.EarlyStopping(patience=5)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=2,
        filepath=os.path.join("mmpn2-msg-cls", "{epoch}-{val_loss:.2f}-{val_acc:.3f}"),
        verbose=True,
    )
    trainer = pl.Trainer(
        gpus=[0, ],
        max_epochs=200,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        distributed_backend='dp',
    )
    trainer.fit(model)
