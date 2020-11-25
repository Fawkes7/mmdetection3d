import pycgal, mmdet3d, h5py, mmcv, os.path as osp, numpy as np, os, shutil, pandas as pd, pickle, json
from pathlib import Path
from pc_util import write_ply_color, point_cloud_to_bbox, render_pts_with_label, write_bbox, write_bbox_color, write_bbox_color_json
from progressbar import ProgressBar


item_template = f'/home/lz/data/Projects/Vision/3D/code/working_code/html_template/one_pcd.html'
item2_template = f'/home/lz/data/Projects/Vision/3D/code/working_code/html_template/two_pcd.html'
table_template = f'/home/lz/data/Projects/Vision/3D/code/working_code/html_template/table.html'


def make_relpath_dict(d: dict, base_dir):
    def f(x):
        if isinstance(x, str):
            return osp.relpath(x, base_dir) if osp.exists(x) else x
        elif isinstance(x, list):
            return [f(y) for y in x]
        elif isinstance(x, dict):
            return {k: f(x[k]) for k in x.keys()}
    return f(d)


class GoodDict(dict):
    def __missing__(self, key):
        return "/.missing"


def html_from_template(template_file, out_file, fix_missing=True, **kwargs):
    with open(template_file, "r") as f:
        template_string = f.read()

    template_string = template_string.replace('{', '{{')
    template_string = template_string.replace('}', '}}')

    # create real format fields
    template_string = template_string.replace('&lformat ', '{')
    template_string = template_string.replace(' &rformat', '}')

    d = GoodDict(kwargs) if fix_missing else kwargs
    # print(template_string, d)
    outfile_string = template_string.format_map(d)
    # print(outfile_string)
    #exit(0)
    with open(out_file, "w") as f:
        f.write(outfile_string)


def build_website_from_mmdet3d(annotations, dataset_dir, output_dir, pred=None):
    dataset_path = Path(dataset_dir)
    item_dir = osp.join(output_dir, 'item')
    shutil.rmtree(output_dir, True)
    mmcv.mkdir_or_exist(item_dir)
    num = len(annotations)
    num = 10
    table = []
    item_file_format = """<a href="./item/{page_name}" title="{page_title}"> {page_title} </a>"""
    color_map_item_format = """<h2 style="display:block"> <span style = "color:{color_name}"> {class_name} </span> </h2>"""

    bar = ProgressBar()
    for i in bar(range(num)):
        annotation = annotations[i]
        gt_box = annotation['annos']['gt_boxes_upright_depth']
        num_gt_box = annotation['annos']['gt_num']
        box_class = annotation['annos']['class']
        point_file = annotation['pts_path']
        instance_file = annotation['pts_instance_mask_path']
        sem_file = annotation['pts_semantic_mask_path']


        #annotation['class']
        #annotation['name']

        point_file = str(dataset_path / point_file)
        sem_file = str(dataset_path / instance_file)
        instance_file = str(dataset_path / instance_file)

        point = np.fromfile(point_file, np.float32).reshape(-1, 3)
        #print(point.shape, point_file)
        #sem = np.fromfile(sem_file).astype(int)
        instance = np.fromfile(instance_file, np.long)
        pi_path = osp.join(item_dir, f'point_instance_{i}.ply')
        bbox_path = osp.join(item_dir,  f'gt_box_{i}.ply')
        json_bbox_path = osp.join(item_dir,  f'gt_box_{i}.json')

        gt_box[:, -3:] *= 1.01

        colors = write_bbox_color_json(gt_box, box_class, json_bbox_path)
        write_bbox_color(gt_box, box_class, bbox_path)
        write_ply_color(point, instance, pi_path)

        def rgb_to_hex(rgb):
            ans = '#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])
            return ans

        rpi_path = osp.relpath(pi_path, item_dir)
        rbbox_path = osp.relpath(bbox_path, item_dir)
        rjbbox_path = osp.relpath(json_bbox_path, item_dir)

        table.append([item_file_format.format(page_name=f'{i}.html', page_title=f'Instance {i}')] +
                     [0 for i in range(len(part_name_list))])
        if pred is None:
            # html_from_template(item_template, out_file=osp.join(item_dir, f'{i}.html'),
            #                    file_path=f"'{rjbbox_path}'", file_name="'gt_box'",
            #                    file_type="'json_box'")

            color_item = ''
            for label in colors:
                c = rgb_to_hex(colors[label])
                l = part_name_list[label]
                color_item = color_item + color_map_item_format.format(color_name=c, class_name=l)

            html_from_template(item_template, out_file=osp.join(item_dir, f'{i}.html'),
                               file_path=f"'{rpi_path}', '{rjbbox_path}'", file_name="'pc_instance', 'gt_box'",
                               file_type="'pcd', 'json_box'", color_map=color_item)
        else:
            boxes = pred[i]['boxes_3d'].tensor.numpy()[:, :6]
            score = pred[i]['scores_3d'].numpy()
            label = pred[i]['labels_3d'].numpy()

            boxes = boxes[score > 0.1]
            label = label[score > 0.1]
            #print(boxes.shape, label.shape)

            pred_bbox_path = osp.join(item_dir, f'pred_box_{i}.ply')
            json_pred_bbox_path = osp.join(item_dir, f'pred_box_{i}.json')
            write_bbox_color(boxes, label, pred_bbox_path)
            colors_ = write_bbox_color_json(boxes, label, json_pred_bbox_path)
            colors.update(colors_)
            color_item = ''
            for label in colors:
                c = rgb_to_hex(colors[label])
                l = part_name_list[label]
                color_item = color_item + color_map_item_format.format(color_name=c, class_name=l)

            pred_rjbbox_path = osp.relpath(json_pred_bbox_path, item_dir)
            html_from_template(item2_template, out_file=osp.join(item_dir, f'{i}.html'),
                               file_path=f"'{rpi_path}', '{rjbbox_path}', '{pred_rjbbox_path}'",
                               file_name="'pc_instance', 'gt_box', 'pred_box'",
                               file_type="'pcd', 'json_box', 'json_box'", color_map=color_item)

    df = pd.DataFrame(table, columns=['Obj'] + part_name_list)
    main_table = df.to_html(escape=False, classes="table sortable is-striped is-hoverable", border=0)
    html_from_template(table_template, out_file=osp.join(output_dir, 'main_page.html'), table_string=main_table)


cat = 'Laptop'
level = str(1)
mode = 'train'

output_dir = f'/home/lz/data/Projects/Vision/3D/code/working_code/partnet_visualizer/{cat}-{level}/{mode}'
dataset_dir = f'/home/lz/data/Projects/Vision/3D/code/working_code/partnet_dataset/{cat}-{level}/{mode}/'
info_file = f'partnet_dataset/{cat}-{level}/{mode}/info.pkl'
annotations = pickle.load(open(info_file, 'rb'))
pred = pickle.load(open('/home/lz/data/Projects/Vision/3D/github_resources/mmdetection3d/train.pkl', 'rb'))


with open(f'/home/lz/data/dataset/PartNet/stats/after_merging2_label_ids/{cat}-level-{level}.txt', 'r') as fin:
    part_name_list = [item.rstrip().split()[1] for item in fin.readlines()]

build_website_from_mmdet3d(annotations, dataset_dir, output_dir, pred=pred)
import os
os.system('rm -r /var/www/html/partnet_visualizer/')
os.system('cp -r partnet_visualizer/ /var/www/html/')


