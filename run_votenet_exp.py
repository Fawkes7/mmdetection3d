import json, os, pathlib
from visualize_partnet import html_from_template, generate_website
from process_partnet_data import generate_scannet_like


def run_one(cat, level, generate_data=True, run_train=True, run_test=True):
    working_folder = pathlib.Path(__file__).absolute().parent
    template_json = str(working_folder / 'configs/partnet_category_template.json')
    target_json = str(working_folder / 'configs/partnet_category.json')
    html_from_template(template_json, target_json, cat=cat, level=level, workingcode=str(working_folder))
    partnet_folder = json.load(open(target_json))['partnet']

    if generate_data:
        generate_scannet_like(cat, level, mode='train', partnet_dir=partnet_folder)
        generate_scannet_like(cat, level, mode='test', partnet_dir=partnet_folder)
        generate_scannet_like(cat, level, mode='val', partnet_dir=partnet_folder)

    if run_train:
        template_sh = str(working_folder / 'scripts/train_votenet_template.sh')
        target_sh = str(working_folder / 'scripts/train_votenet.sh')
        html_from_template(template_sh, target_sh, log_name=f'{cat}-{level}')
        os.system('bash ' + target_sh)

    if run_test:
        template_sh = str(working_folder / 'scripts/test_votenet_template.sh')
        target_sh = str(working_folder / 'scripts/test_votenet.sh')
        mode = 'train'
        html_from_template(template_sh, target_sh, log_name=f'{cat}-{level}', save_name=f'{mode}_{cat}-{level}')
        os.system('bash ' + target_sh)
        generate_website(cat, level, mode=mode)


if __name__ == '__main__':
    data_folder = '/home/lz/data/dataset/PartNet/ins_seg_h5_for_detection'
    data_path = pathlib.Path(data_folder)
    #run_one('Bed', '1', generate_data=False, run_train=False, run_test=True)
    #exit(0)

    for folder in sorted(data_path.glob('*')):
        if folder.is_dir():
            name = folder.name
            cat, level = name.split('-')
            if name != 'Bed-1':
                print('Skip name ' + name)
                continue
            print(f'Run {cat} {level}')
            run_one(cat, level, generate_data=False, run_train=False, run_test=True)

#cat_info = json.load(open('/home/lz/data/Projects/Vision/3D/code/working_code/configs/partnet_category.json'))
#cat = cat_info['cat']
#level = cat_info['level']

