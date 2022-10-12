#! /usr/bin/env/python3

"""
- create train/val/test/deploy? split from 1-folder-1-annotation-file-setup
- balance the dataset (pos/neg imgs)
- augmentations (opt)
- get polygon images, create masks from polygons
- generate/save dataset objects
- transforms
"""

import os
import numpy as np
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch


class WeedDataset(Dataset):
    """ weed dataset"""

    def __init__(self, ann_file, img_dir, mask_dir, transforms=None):
        
        print(ann_file)
        annotations = json.load(open(os.path.join(root_dir, 'metadata', ann_file)))
        self.annotations = list(annotations.values())
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms

        # load all image files, sorting them to ensure aligned (dictionaries are unsorted)
        self.imgs = list(sorted(os.listdir(self.img_dir)))
        self.masks = list(sorted(os.listdir(self.mask_dir)))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image

        # get mask

        # get bbox

        
