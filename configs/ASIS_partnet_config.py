import json, pickle, pathlib
import os
# all config file is in the parent folder
cur_path = pathlib.Path(os.path.abspath(os.getcwd()))
config_folder = cur_path.parent.parent / 'configs'
# category info
cat_info = json.load(open(str(config_folder / 'partnet_category.json')))

cat = cat_info['cat']
level = cat_info['level']

working_code = pathlib.Path(cat_info['working_code'])
mmdet3d_folder = pathlib.Path(cat_info['mmdet3d'])

config_root = working_code / 'configs/_base_/'
data_config = working_code / 'configs/partnet_data_config.py'
data_root = working_code / 'configs/data/'
model_root = working_code / 'configs/model/'
schedule_root = working_code / 'configs/schedules/'

# mean_size = pickle.load(open(str(working_code / f'partnet_dataset/{cat}-{level}/train/mean_size.pkl'), 'rb'))

_base_ = [
    str(data_config),
    str(config_root / 'network/ASIS.py'),
    str(config_root / 'schedules/schedule_3x.py'),
    str(config_root + 'default_runtime.py'),
    str(data_root / 'partnet_train_data_config.py'),
    str(model_root / 'ASIS.py'),
    str(schedule_root / 'schedule_partnet.py'), str(config_root / 'default_runtime.py')
]
print(_base_)
# model settings

model = dict(num_class=12)
