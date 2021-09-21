#! /usr/bin/env python

"""
script to create dataset objects from csv files
"""

# specify label file
# set transforms, etc
# create dataset
# create dataloader
# save files

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from deepweeds_dataset import DeepWeedsDataset, Rescale, RandomAffine, RandomColorJitter, RandomPixelIntensityScaling
from deepweeds_dataset import RandomHorizontalFlip, RandomRotate, RandomResizedCrop, ToTensor, Compose
from deepweeds_dataset import CLASSES, CLASS_NAMES

torch.manual_seed(42)

# img folder
img_dir = os.path.join('images')

# label file
lbl_file = 'deployment_labels.csv'
lbl_path = os.path.join('labels', lbl_file)

# transforms
tforms = Compose([
        Rescale(256),
        RandomRotate(360),
        RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0)),
        RandomColorJitter(brightness=(0, 0.1), hue=(-0.01, 0.01)), # TODO unsure about these values
        RandomPixelIntensityScaling(),
        RandomAffine(degrees=5, translate=(0.05, 0.05)),
        RandomHorizontalFlip(prob=0.5),
        ToTensor()
    ])

num_epochs = 500
learning_rate = 0.001 # LR halved if val loss did not decrease after 16 epochs
momentum = 0.9
batch_size = 32
shuffle = True
num_workers = 10

dataset = DeepWeedsDataset(csv_file=lbl_path,
                           root_dir=img_dir,
                           transform=tforms)

nimg = len(dataset)
print('full_dataset length =', nimg)

tedl = DataLoader(dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers)