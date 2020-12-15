import json, pathlib
import os
# config_folder = pathlib.Path(__file__).parent.parent
cur_path = pathlib.Path(os.path.abspath(os.getcwd()))
config_folder = cur_path.parent
print(config_folder)

cat_info = json.load(open(str(config_folder / 'partnet_category.json')))
cat = cat_info['cat']
level = cat_info['level']
working_code = pathlib.Path(cat_info['working_code'])
partnet = pathlib.Path(cat_info['partnet'])

dataset_type = 'ScanNetDataset'

train_root = working_code / f'partnet_dataset/{cat}-{level}/train/'
test_root = working_code / f'partnet_dataset/{cat}-{level}/test/'
val_root = working_code / f'partnet_dataset/{cat}-{level}/val/'

with open(str(partnet / f'stats/after_merging2_label_ids/{cat}-level-{level}.txt'), 'r') as fin:
    class_names = [item.rstrip().split()[1] for item in fin.readlines()]

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
    test=dict(
        type=dataset_type,
        data_root=train_root,
        ann_file=str(train_root / 'info.pkl'),
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))