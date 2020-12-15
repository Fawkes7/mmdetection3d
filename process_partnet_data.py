import h5py, numpy as np, pickle
from pc_util import point_cloud_to_bbox
from pathlib import Path
import mmcv, os.path as osp
from collections import defaultdict
import os

def generate_scannet_like_from_h5(files, part_name_lists, output_dir, mode):
    mmcv.mkdir_or_exist(output_dir)

    example_index = 0
    ret = []

    if mode != 'test':
        mean_size = defaultdict(list)
        for file in files:
            file_h5 = h5py.File(file, 'r')
            points = file_h5['pts']
            part_masks = file_h5['gt_mask']
            sem_labels = file_h5['gt_label']
            valids = file_h5['gt_valid']
            other_masks = file_h5['gt_other_mask']
            for i in range(points.shape[0]):
                point = points[i]
                part_mask = part_masks[i]
                sem_label = sem_labels[i]
                if i < 2:
                    print(i, "sem_labels", sem_labels)
                valid = valids[i]
                boxes = []
                names = []
                classes = []
                if not valid.any():
                    continue

                part_label = sem_label.copy()
                for j in range(part_mask.shape[0]):
                    part_mask_j = part_mask[j]
                    if not valid[j] or not part_mask_j.any():
                        continue
                    box = point_cloud_to_bbox(point[part_mask_j])
                    part_label[part_mask_j] = len(boxes)
                    boxes.append(box)
                    classes.append(int(sem_label[part_mask_j][0]) - 1)

                    if classes[-1] >= len(part_name_lists):
                        print(classes[-1], part_name_lists)
                        exit(0)
                    names.append(part_name_lists[classes[-1]])
                    mean_size[classes[-1]].append(box[3:])
                boxes = np.array(boxes)

                point_file = osp.join(output_dir, f'point_{example_index}.bin')
                instance_file = osp.join(output_dir, f'instance_{example_index}.bin')
                sem_file = osp.join(output_dir, f'sem_{example_index}.bin')
                point = np.array(point).astype(np.float32)
                part_label = np.array(part_label).astype(np.long)
                sem_label = np.array(sem_label).astype(np.long)


                # print(f'point_{example_index}.bin'))
                # print(point.shape)
                # exit(0)
                point_xyz_feat = np.concatenate([point, point], axis=-1)
                point.tofile(point_file)
                part_label.tofile(instance_file)
                sem_label.tofile(sem_file)

                annotations = {'gt_num': boxes.shape[0]}
                #print(boxes.shape, valid.any())
                annotations['location'] = boxes[:, :3]
                annotations['dimensions'] = boxes[:, 3:6]
                annotations['gt_boxes_upright_depth'] = boxes
                annotations['index'] = np.arange(annotations['gt_num'], dtype=np.int32)
                annotations['name'] = names
                annotations['class'] = np.array(classes, dtype=np.int32)
                example = {'point_cloud': {'num_features': 3, 'lidar_idx': f'point_{example_index}'},
                           'pts_path': f'point_{example_index}.bin',
                           'pts_instance_mask_path': f'instance_{example_index}.bin',
                           'pts_semantic_mask_path': f'sem_{example_index}.bin',
                           'annos': annotations
                           }
                ret.append(example)
                example_index += 1
        if mode == 'train':
            mean_size_ret = []
            for i in range(len(mean_size)):
                mean_size_ret.append(np.mean(np.array(mean_size[i]), axis=0).tolist())
            pickle.dump(mean_size_ret, open(osp.join(output_dir, f'mean_size.pkl'), 'wb'))
    else:
        for file in files:
            file_h5 = h5py.File(file, 'r')
            points = file_h5['pts']
            for i in range(points.shape[0]):
                point = points[i]
                point_file = osp.join(output_dir, f'point_{example_index}.bin')
                point = np.array(point).astype(np.float32)
                point.tofile(point_file)
                example = {'point_cloud': {'num_features': 3, 'lidar_idx': f'point_{example_index}'},
                           'pts_path': f'point_{example_index}.bin', }
                ret.append(example)
                example_index += 1
    pickle.dump(ret, open(osp.join(output_dir, 'info.pkl'), 'wb'))


def generate_scannet_like(cat='Laptop', level=1, mode='train'):
    print(f'Generate {cat} {level} {mode}')
    # use your personalized datapath
    out_root = f"/home/haoyuan/data/partnet_dataset/{cat}-{level}/{mode}"
    root_h5data = Path('/home/haoyuan/data/h5PartNet')
    root = Path(f'{root_h5data}/ins_seg_h5_for_detection/{cat}-{level}/')
    files = sorted([str(_) for _ in root.glob(f'{mode}*.h5')])

    with open(f'{root_h5data}/stats/after_merging2_label_ids/{cat}-level-{level}.txt', 'r') as fin:
        part_name_list = [item.rstrip().split()[1] for item in fin.readlines()]
    generate_scannet_like_from_h5(files, part_name_list, out_root, mode=mode)


if __name__ == '__main__':
    import json

    cur_path = Path(os.path.abspath(os.getcwd()))
    config_folder = cur_path / 'configs'
    cat_info = json.load(open(config_folder /'partnet_category.json'))
    cat_a = cat_info['cat']
    level_a = cat_info['level']
    generate_scannet_like(cat_a, level_a, mode='train')
    generate_scannet_like(cat_a, level_a, mode='test')
    generate_scannet_like(cat_a, level_a, mode='val')
