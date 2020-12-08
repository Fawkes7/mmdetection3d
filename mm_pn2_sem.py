import mmdet3d.models.backbones
from mmdet.models import BACKBONES
from mmdet.apis.train import set_random_seed
from torch_geometric.datasets import ShapeNet
import torch.nn as nn, torch, pointnet2, h5py, pytorch_lightning as pl, os.path as osp, os
from torch.utils.data import DataLoader
from pointnet2.models.pointnet2_ssg_sem import PointNet2SemSegSSG


class Pointnet2_SSG_SEM_MM3D(PointNet2SemSegSSG):
    def _build_model(self):
        self.backbone = BACKBONES.module_dict['PointNet2SASSG'](9, num_points=(1024, 256, 64, 16),
                                                                radius=(0.1, 0.2, 0.4, 0.8),
                                                                num_samples=(32, 32, 32, 32),
                                                                sa_channels=(
                                                                    (32, 32, 64), (64, 64, 128), (128, 128, 256),
                                                                    (256, 256, 512)),
                                                                fp_channels=(
                                                                    (256, 256), (256, 256), (256, 128),
                                                                    (128, 128, 128)),
                                                                sa_cfg=dict(
                                                                    type='PointSAModule',
                                                                    pool_mod='max',
                                                                    use_xyz=True,
                                                                    normalize_xyz=False))
        self.fc = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 13, kernel_size=1),
        )

    def forward(self, pc):
        all = self.backbone(pc)
        features = all['fp_features'][-1]
        predict = self.fc(features)
        return predict


class Pointnet2_MSG_SEM_MM3D(Pointnet2_SSG_SEM_MM3D):
    def _build_model(self):
        self.backbone = BACKBONES.module_dict['PointNet2SAMSG'](9, num_points=(1024, 256, 64, 16),
                                                                radii=((0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.8)),
                                                                num_samples=((16, 32), (16, 32), (16, 32), (16, 32)),
                                                                sa_channels=(
                                                                    ((16, 16, 32), (32, 32, 64)),
                                                                    ((64, 64, 128), (64, 96, 128)),
                                                                    ((128, 196, 256), (128, 196, 256)),
                                                                    ((256, 256, 512), (256, 384, 512))
                                                                ),
                                                                fp_channels=((512, 512), (512, 512), (256, 256), (128, 128)),
                                                                dilated_group=(False, False, False, False),
                                                                fps_mods=(('D-FPS'), ('D-FPS'), ('D-FPS'), ('D-FPS')),
                                                                fps_sample_range_lists=((-1), (-1), (-1), (-1)),
                                                                aggregation_channels=(None, None, None, None),
                                                                )
        self.fc = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 13, kernel_size=1),
        )


mode = 'msg'

hparams = dict()
hparams['batch_size'] = 24
hparams['num_points'] = 4096
hparams['optimizer.weight_decay'] = 0.0
hparams['optimizer.lr'] = 1e-3
hparams['optimizer.lr_decay'] = 0.5
hparams['optimizer.bn_momentum'] = 0.5
hparams['optimizer.bnm_decay'] = 0.5
hparams['optimizer.decay_step'] = 3e5

if mode == 'ssg':
    model = Pointnet2_SSG_SEM_MM3D(hparams).cuda()

    early_stop_callback = pl.callbacks.EarlyStopping(patience=5)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=2,
        filepath=os.path.join("mmpn2-ssg-sem", "{epoch}-{val_loss:.2f}-{val_acc:.3f}"),
        verbose=True,
    )
    trainer = pl.Trainer(
        gpus=[0, ],
        max_epochs=50,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        distributed_backend='dp',
    )
    trainer.fit(model)
elif mode == 'msg':
    model = Pointnet2_MSG_SEM_MM3D(hparams).cuda()

    early_stop_callback = pl.callbacks.EarlyStopping(patience=5)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=2,
        filepath=os.path.join("mmpn2-msg-sem", "{epoch}-{val_loss:.2f}-{val_acc:.3f}"),
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
