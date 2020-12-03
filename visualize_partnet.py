import pycgal, mmdet3d, h5py, mmcv, os.path as osp, numpy as np, os, shutil, pandas as pd, pickle, json, torch
import json, pathlib
from pathlib import Path
from pc_util import write_ply_color, point_cloud_to_bbox, render_pts_with_label, write_bbox, write_bbox_color, write_bbox_color_json
from progressbar import ProgressBar
from mmdet3d.core.evaluation.indoor_eval import eval_map_recall, eval_det_cls
from collections import defaultdict


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


def generate_infos(gt_annos, dt_annos, metric, box_type_3d=None, box_mode_3d=None):
    assert len(dt_annos) == len(gt_annos)
    pred, gt = {}, {}
    for img_id in range(len(dt_annos)):
        det_anno = dt_annos[img_id]
        for i in range(len(det_anno['labels_3d'])):
            label = det_anno['labels_3d'].numpy()[i]
            bbox = det_anno['boxes_3d'].convert_to(box_mode_3d)[i]
            score = det_anno['scores_3d'].numpy()[i]
            if label not in pred:
                pred[int(label)] = {}
            if img_id not in pred[label]:
                pred[int(label)][img_id] = []
            if label not in gt:
                gt[int(label)] = {}
            if img_id not in gt[label]:
                gt[int(label)][img_id] = []
            pred[int(label)][img_id].append((bbox, score))

        gt_anno = gt_annos[img_id]
        if gt_anno['gt_num'] != 0:
            gt_boxes = box_type_3d(gt_anno['gt_boxes_upright_depth'],
                                   box_dim=gt_anno['gt_boxes_upright_depth'].shape[-1],
                                   origin=(0.5, 0.5, 0.5)).convert_to(box_mode_3d)
            labels_3d = gt_anno['class']
        else:
            gt_boxes = box_type_3d(np.array([], dtype=np.float32))
            labels_3d = np.array([], dtype=np.int64)

        for i in range(len(labels_3d)):
            label = labels_3d[i]
            bbox = gt_boxes[i]
            if label not in gt:
                gt[label] = {}
            if img_id not in gt[label]:
                gt[label][img_id] = []
            gt[label][img_id].append(bbox)

    def zero_dict():
        return defaultdict(lambda :0.0)
    recalls = [defaultdict(zero_dict) for _ in range(len(metric))]
    precisions = [defaultdict(zero_dict) for _ in range(len(metric))]
    aps = [defaultdict(zero_dict) for _ in range(len(metric))]
    aabb_ious = defaultdict(zero_dict)

    for classname in gt.keys():
        for img_id in gt[classname].keys():
            #print(classname, img_id)
            if img_id not in pred[classname] or img_id not in gt[classname]:
                #print(type(img_id), type(list(pred[classname].keys())[0]))
                #rint(gt[classname].keys())
                #exit(0)
                #print()
                continue
            pred_id = {img_id: pred[classname][img_id]}
            gt_id = {img_id: gt[classname][img_id]}
            # print(len(gt[classname][img_id]), type(gt[classname][img_id]))
            # Generate box from gt
            cur_gt_num = len(gt[classname][img_id])
            cur_pred_num = len(pred[classname][img_id])

            if cur_gt_num == 0 or cur_pred_num == 0:
                continue

            pred_cur = torch.zeros((cur_pred_num, 7), dtype=torch.float32)
            score_cur = np.zeros(cur_pred_num, dtype=np.float32)
            gt_cur = torch.zeros([cur_gt_num, 7], dtype=torch.float32)
            for i in range(cur_gt_num):
                gt_cur[i] = gt[classname][img_id][i].tensor
            for i, (box, score) in enumerate(pred[classname][img_id]):
                pred_cur[i] = box.tensor
                score_cur[i] = score

            gt_bbox = box.new_box(gt_cur)
            pred_bbox = box.new_box(pred_cur)

            eval_results = eval_det_cls(pred_id, gt_id, iou_thr=metric)
            for idx, eval_res in enumerate(eval_results):
                recalls[idx][img_id][classname] = eval_res[0]
                precisions[idx][img_id][classname] = eval_res[1]
                aps[idx][img_id][classname] = eval_res[2]
            # print(pred[classname][img_id])
            # print(gt[classname][img_id])
            # print(aps[img_id][classname])#, aabb_ious[img_id][classname])
            # print(pred_bbox, gt_bbox)
            # exit(0)
            #print(img_id, classname)
            #exit(0)
            '''
            #if img_id == 9 and classname == part_name_list.index('keyboard/frame'):
            #    pass
                #print()
                #print(pred_bbox, pred_cur, score_cur)
                #print('------------------')
                #print(gt_bbox, gt_cur)
                #print('------------------')
                #print(box_type_3d.overlaps(pred_bbox, gt_bbox).numpy())
                #exit(0)
            #'''
            aabb_ious[img_id][classname] = np.mean(np.max(box_type_3d.overlaps(pred_bbox, gt_bbox).numpy(), axis=0))
    return aps, precisions, recalls, aabb_ious


def evaluate(gt_infos, pre_dict, part_name_list, iou_th=(0.25, 0.5, 0.75, 0.9)):
    from mmdet3d.core.evaluation import indoor_eval
    from mmdet3d.core.bbox.structures.box_3d_mode import Box3DMode, DepthInstance3DBoxes
    gt_annos = [info['annos'] for info in gt_infos]
    label2cat = {i: cat_id for i, cat_id in enumerate(part_name_list)}
    ret_dict = indoor_eval(
        gt_annos,
        pre_dict,
        iou_th,
        label2cat,
        logger=None,
        box_type_3d=DepthInstance3DBoxes,
        box_mode_3d=Box3DMode.DEPTH)
    #print(ret_dict)
    #exit(0)
    info_image = generate_infos(gt_annos, pre_dict, iou_th, DepthInstance3DBoxes, Box3DMode.DEPTH)
    return ret_dict, info_image


def build_website_from_mmdet3d(annotations, dataset_dir, output_dir, pred, part_name_list, iou_th=(0.5, )):
    ret_dict, (aps, precisions, recalls, aabb_ious) = evaluate(annotations, pred, part_name_list, iou_th)
    #print(ret_dict)
    #exit(0)
    dataset_path = Path(dataset_dir)
    item_dir = osp.join(output_dir, 'item')
    shutil.rmtree(output_dir, True)
    mmcv.mkdir_or_exist(item_dir)
    num = len(annotations)
    #num = 10
    table = []

    data_name = []
    overall_string = ""

    for j, part_name in enumerate(part_name_list):
        data_name.append(f'{part_name}_mIOU')
        overall_string += f"<th>{np.mean([aabb_ious[i][j] for i in aabb_ious if j in aabb_ious[i]]):.2f}</th>"

    for part_name in part_name_list:
        for iou in iou_th:
            data_name.append(f'{part_name}_AP @ {iou:.2f}')
            res = ret_dict[f'{part_name}_AP_{iou:.2f}']
            overall_string += f"<th>{res:.2f}</th>"
    for part_name in part_name_list:
        for iou in iou_th:
            data_name.append(f'{part_name}_AR @ {iou:.2f}')
            res = ret_dict[f'{part_name}_rec_{iou:.2f}']
            overall_string += f"<th>{res:.2f}</th>"

    item_file_format = """<a href="./item/{page_name}" title="{page_title}"> {page_title} </a>"""
    color_map_item_format = """<h2 style="display:block"> <span style = "color:{color_name}"> {class_name} </span> </h2>"""
    bar = ProgressBar()
    for i in bar(range(num)):
        #if i != 9:
        #    continue
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

        colors = write_bbox_color_json(gt_box, box_class, json_bbox_path)
        write_bbox_color(gt_box, box_class, bbox_path)
        write_ply_color(point, instance, pi_path)

        def rgb_to_hex(rgb):
            ans = '#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])
            return ans

        rpi_path = osp.relpath(pi_path, item_dir)
        rbbox_path = osp.relpath(bbox_path, item_dir)
        rjbbox_path = osp.relpath(json_bbox_path, item_dir)
        next_page_rel_path = osp.relpath(osp.join(item_dir,  f'{(i + 1) % num}.html'), item_dir)
        last_page_rel_path = osp.relpath(osp.join(item_dir,  f'{(i - 1 + num) % num}.html'), item_dir)
        table_item_i = [item_file_format.format(page_name=f'{i}.html', page_title=f'Instance {i}')]
        #for k in range(len(iou_th)):
        for j, part_name in enumerate(part_name_list):
            table_item_i.append(f'{aabb_ious[i][j]:.2f}')

        for k in range(len(iou_th)):
            for j, part_name in enumerate(part_name_list):
                if j in aps[k][i]:
                    table_item_i.append(f'{float(aps[k][i][j]):.2f}')
                else:
                    table_item_i.append('N/A')
        for k in range(len(iou_th)):
            for j, part_name in enumerate(part_name_list):
                if j in recalls[k][i]:
                    table_item_i.append(f'{float(recalls[k][i][j][-1]):.2f}')
                else:
                    table_item_i.append('N/A')
        boxes = pred[i]['boxes_3d'].tensor.numpy()[:, :6]
        boxes[:, 2] += boxes[:, 5] * 0.5
        #print(boxes, gt_box)
        #exit(0)
        score = pred[i]['scores_3d'].numpy()
        label = pred[i]['labels_3d'].numpy()

        flag = score > 0.1
        # flag = np.logical_and(score > 0.1, label == 0)
        boxes = boxes[flag]
        label = label[flag]
        #print(boxes.shape, label, score[score > 0.1])
        table.append(table_item_i)
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
                           file_type="'pcd', 'json_box', 'json_box'", color_map=color_item,
                           last_page=last_page_rel_path, next_page=next_page_rel_path)

    df = pd.DataFrame(table, columns=['Object', ] + data_name)
    main_table = df.to_html(escape=False, classes="table sortable is-striped is-hoverable", border=0, index=False)
    html_from_template(table_template, out_file=osp.join(output_dir, 'main_page.html'), table_string=main_table,
                       overal_info=overall_string)


def build_website_from_mmdet3d_without_pred(annotations, dataset_dir, output_dir, part_name_list):
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
        #exit(0)
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
        next_page_rel_path = osp.relpath(osp.join(item_dir,  f'{(i + 1) % num}.html'), item_dir)
        last_page_rel_path = osp.relpath(osp.join(item_dir,  f'{(i - 1 + num) % num}.html'), item_dir)
        table.append([item_file_format.format(page_name=f'{i}.html', page_title=f'Instance {i}')] +
                     [0 for i in range(len(part_name_list))])
        color_item = ''
        for label in colors:
            c = rgb_to_hex(colors[label])
            l = part_name_list[label]
            color_item = color_item + color_map_item_format.format(color_name=c, class_name=l)

        html_from_template(item_template, out_file=osp.join(item_dir, f'{i}.html'),
                           file_path=f"'{rpi_path}', '{rjbbox_path}'", file_name="'pc_instance', 'gt_box'",
                           file_type="'pcd', 'json_box'", color_map=color_item,
                           last_page=last_page_rel_path, next_page=next_page_rel_path)

    df = pd.DataFrame(table, columns=['Obj'] + part_name_list)
    main_table = df.to_html(escape=False, classes="table sortable is-striped is-hoverable", border=0, index=False)
    html_from_template(table_template, out_file=osp.join(output_dir, 'main_page.html'), table_string=main_table)


def generate_website(cat=None, level=None, mode='train'):
    # config_folder = pathlib.Path(__file__).parent.parent
    config_folder = pathlib.Path('/home/lz/data/Projects/Vision/3D/code/working_code/configs/')
    cat_info = json.load(open(str(config_folder / 'partnet_category.json')))
    cat = cat_info['cat'] if cat is None else cat
    level = cat_info['level'] if cat is None else level
    working_code = pathlib.Path(cat_info['working_code'])
    partnet = pathlib.Path(cat_info['partnet'])
    mmdet3d_folder = pathlib.Path(cat_info['mmdet3d'])

    output_dir = str(working_code / f'partnet_visualizer/{cat}-{level}/{mode}')
    dataset_dir = str(working_code / f'partnet_dataset/{cat}-{level}/{mode}/')
    info_file = str(working_code / f'partnet_dataset/{cat}-{level}/{mode}/info.pkl')
    annotations = pickle.load(open(info_file, 'rb'))
    with open(str(partnet / f'stats/after_merging2_label_ids/{cat}-level-{level}.txt'), 'r') as fin:
        part_name_list = [item.rstrip().split()[1] for item in fin.readlines()]
    pred = pickle.load(open(str(mmdet3d_folder / f'results/{mode}_{cat}-{level}.pkl'), 'rb'))
    build_website_from_mmdet3d(annotations, dataset_dir, output_dir, pred, part_name_list)
    #build_website_from_mmdet3d_without_pred(annotations, dataset_dir, output_dir, part_name_list)


if __name__ == '__main__':
    cat_info = json.load(open('/home/lz/data/Projects/Vision/3D/code/working_code/configs/partnet_category.json'))
    cat = cat_info['cat']
    level = str(cat_info['level'])
    mode = 'train'



