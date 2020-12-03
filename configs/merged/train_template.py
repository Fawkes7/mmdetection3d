import json, pickle, pathlib



config_folder = pathlib.Path('/home/lz/data/Projects/Vision/3D/code/working_code/configs/')
cat_info = json.load(open(str(config_folder / 'partnet_category.json')))
cat = cat_info['cat']
level = cat_info['level']
working_code = pathlib.Path(cat_info['working_code'])
mmdet3d_folder = pathlib.Path(cat_info['mmdet3d'])

config_root = mmdet3d_folder / 'configs/_base_/'
data_root = working_code / 'configs/data/'
model_root = working_code / 'configs/model/'
schedule_root = working_code / 'configs/schedules/'

working_code = pathlib.Path(cat_info['working_code'])
partnet = pathlib.Path(cat_info['partnet'])


train_root = "&lformat trainroot &rformat"
test_root = "&lformat testroot &rformat"
val_root = "&lformat valroot &rformat"

#train_root = working_code / f'partnet_dataset/{cat}-{level}/train/'
#test_root = working_code / f'partnet_dataset/{cat}-{level}/test/'
#val_root = working_code / f'partnet_dataset/{cat}-{level}/val/'

#partnet / f'stats/after_merging2_label_ids/{cat}-level-{level}.txt'

with open(str("&lformat partname &rformat"), 'r') as fin:
    class_names = [item.rstrip().split()[1] for item in fin.readlines()]

_base_ = ['votenet.py']
dataset_type = 'ScanNetDataset'


mean_size = pickle.load(open("&lformat meansize &rformat", 'rb'))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        shift_height=False,
        load_dim=3,
        use_dim=[0, 1, 2]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.087266, 0.087266],
        scale_ratio_range=[1.0, 1.0],
        shift_height=True),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        shift_height=False,
        load_dim=3,
        use_dim=[0, 1, 2]),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
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
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            data_root=train_root,
            ann_file=str(train_root / 'info.pkl'),
            pipeline=train_pipeline,
            filter_empty_gt=False,
            classes=class_names,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        data_root=val_root,
        ann_file=str(val_root / 'info.pkl'),
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        data_root=test_root,
        ann_file=str(test_root / 'info.pkl'),
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))

model = dict(
    bbox_head=dict(
        num_classes=len(mean_size),
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=len(mean_size),
            num_dir_bins=1,
            with_rot=False,
            mean_sizes=mean_size,)
    )
)