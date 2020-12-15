import h5py, numpy as np, pickle
from pc_util import point_cloud_to_bbox
from pathlib import Path
import mmcv, os.path as osp
from collections import defaultdict

out_root = "/home/haoyuan/data/ASIS"

root_h5data = Path('/home/haoyuan/data/h5PartNet/')
point_root = root_h5data / 'ins_seg_h5_for_detection'

with open(point_root / 'shape_names.txt') as f:
    lines = f.readlines()
# for l in lines:
    
print(lines)

