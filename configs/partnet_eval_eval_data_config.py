#import numpy as np, os.path as osp

dataset_type = 'ScanNetDataset'
cat = 'Laptop'
level = 1
train_root = f'/home/lz/data/Projects/Vision/3D/code/working_code/partnet_dataset/{cat}-{level}/train/'
test_root = f'/home/lz/data/Projects/Vision/3D/code/working_code/partnet_dataset/{cat}-{level}/test/'
val_root = f'/home/lz/data/Projects/Vision/3D/code/working_code/partnet_dataset/{cat}-{level}/val/'

'''
with open(f'/home/lz/data/dataset/PartNet/stats/after_merging2_label_ids/{cat}-level-{level}.txt', 'r') as fin:
    class_names = tuple([item.rstrip().split()[1] for item in fin.readlines()])
    print(class_names)
    tuple(np.arange(len(class_names)))),
'''
class_names = ['laptop/screen_side', 'laptop/base_side']

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5),
            dict(type='IndoorPointSample', num_points=40000),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    test=dict(
        type=dataset_type,
        data_root=val_root,
        ann_file=val_root + 'info.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))