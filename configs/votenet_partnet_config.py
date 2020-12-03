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

mean_size = pickle.load(open(str(working_code / f'partnet_dataset/{cat}-{level}/train/mean_size.pkl'), 'rb'))

_base_ = [
    str(data_root / 'partnet_train_data_config.py'),
    str(model_root / 'votenet.py'),
    str(schedule_root / 'schedule_partnet.py'), str(config_root / 'default_runtime.py')
]

# model settings
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