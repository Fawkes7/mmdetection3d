import h5py
import os
import json


# load h5 files
stat_in_fn = '../../stats/after_merging_label_ids/%s-level-%d.txt' % (FLAGS.category, FLAGS.level_id)
print('Reading from ', stat_in_fn)
with open(stat_in_fn, 'r') as fin:
    part_name_list = [item.rstrip().split()[1] for item in fin.readlines()]
print('Part Name List: ', part_name_list)
data_in_dir = '../../../h5PartNet/ins_seg_h5_for_detection/%s-%d/' % (FLAGS.category, FLAGS.level_id)
train_h5_fn_list = []
for item in os.listdir(data_in_dir):
    if item.endswith('.h5') and item.startswith('train-'):
        train_h5_fn_list.append(item)

val_h5_fn_list = []
for item in os.listdir(data_in_dir):
    if item.endswith('.h5') and item.startswith('val-'):
        val_h5_fn_list.append(item)


def load_h5(fn):
    with h5py.File(fn, 'r') as fin:
        pts = fin['pts'][:]
        gt_label = fin['gt_label'][:]
        gt_mask = fin['gt_mask'][:]
        gt_valid = fin['gt_valid'][:]
        gt_other_mask = fin['gt_other_mask'][:]
        return pts, gt_label, gt_mask, gt_valid, gt_other_mask

def load_json(fn):
    with open(fn, 'r') as fin:
        return json.load(fin)

def load_data(fn):
    cur_json_fn = fn.replace('.h5', '.json')
    record = load_json(cur_json_fn)
    pts, gt_label, gt_mask, gt_valid, gt_other_mask = load_h5(fn)
    return pts, gt_label, gt_mask, gt_valid, gt_other_mask, record

def train_one_batch():
    for item in train_h5_fn_list:
        cur_h5_fn = os.path.join(data_in_dir, item)
        print('Reading data from ', cur_h5_fn)
        pts, gt_label, gt_mask, gt_valid, gt_other_mask, _ = load_data(cur_h5_fn)

def val_one_batch():
    for item in val_h5_fn_list:
        cur_h5_fn = os.path.join(data_in_dir, item)
        print('Reading data from ', cur_h5_fn)
        pts, gt_label, gt_mask, gt_valid, gt_other_mask, record = load_data(cur_h5_fn)
