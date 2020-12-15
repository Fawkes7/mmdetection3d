import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import json
from pathlib import Path

class PartNetDataset(Dataset):
    """PartNet dataset for semantic segmantation and instance segmentation"""
    def __init__(self, cat_info, mode):
        cat = cat_info['cat']
        level = cat_info['level']

        label_path = str(Path(cat_info["partnet"]) / f"stats/after_merging2_label_ids/{cat}-level-{level}.txt")
        assert os.path.exists(label_path)
        with open(label_path, 'r') as fin:
            class_names = [item.rstrip().split()[1] for item in fin.readlines()]

        generated_data = Path(cat_info["generated_data"])
        data_path = generated_data / f"partnet_dataset/{cat}-{level}/{mode}"
        info_path = str(generated_data / f"partnet_dataset/{cat}-{level}/{mode}/info.pkl")
        assert os.path.exists(info_path)
        self.annotations = pickle.load(open(info_path, 'rb'))
        self.data_path = data_path

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        annotation = self.annotations[idx]
        # str
        point_file = annotation['pts_path']
        instance_file = annotation['pts_instance_mask_path']
        sem_file = annotation['pts_semantic_mask_path']

        # Path
        point_file = str(self.data_path / point_file)
        sem_file = str(self.data_path / sem_file)
        instance_file = str(self.data_path / instance_file)
        sem = np.fromfile(sem_file, dtype=np.long).astype(int)
        point = np.fromfile(point_file, np.float32).reshape(-1, 3)
        instance = np.fromfile(instance_file, np.long)

        # names, class are not used
        names = annotation['annos']['name']
        classes = annotation['annos']['class']
        return point, sem, instance+1  #, names, classes



if __name__=='__main__':
    cur_path = Path(os.path.abspath(os.getcwd()))
    config_folder = cur_path.parent.parent / 'configs/'
    cat_info = json.load(open(str(config_folder / 'partnet_category.json')))
    dataset = PartNetDataset(cat_info, 'train')
    print(torch.unique(torch.tensor(dataset[0][1])))
    print(dataset[0][0].shape, dataset[1][1], dataset[1][2])