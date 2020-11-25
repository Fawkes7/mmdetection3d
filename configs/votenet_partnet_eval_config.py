
config_root = '/home/lz/data/Projects/Vision/3D/github_resources/mmdetection3d/configs/_base_/'
'''
import os.path as osp, pickle
mean_size = pickle.load(
    open('/home/lz/data/Projects/Vision/3D/code/working_code/partnet_dataset/Laptop-1/train/mean_size.pkl', 'rb'))
print(mean_size)
'''

mean_size = [[1.343528389930725, 0.8696274757385254, 0.2557317912578583],
             [1.3536276817321777, 0.08139896392822266, 0.9777247905731201]]

_base_ = [
    '/home/lz/data/Projects/Vision/3D/code/working_code/configs/partnet_eval_train_data_config.py',
    config_root + 'models/votenet.py',
    config_root + 'schedules/schedule_3x.py', config_root + 'default_runtime.py'
]

# model settings
model = dict(
    bbox_head=dict(
        num_classes=2,
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=2,
            num_dir_bins=1,
            with_rot=False,
            mean_sizes=mean_size,)
    )
)