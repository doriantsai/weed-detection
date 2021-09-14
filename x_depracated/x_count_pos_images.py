#! /usr/bin/env python

"""
script to count given image folders into negative/positive components
"""

import os
import json
import glob
from weed_detection.PreProcessingToolbox import PreProcessingToolbox
import random
import numpy as np
import shutil


# given folder locations
# given types of files to look for (glob) --> images/annotations
# glob all relevant files
# combine into singular json file with folder locations (might require new tag added)
# output all photos into single folder via symlinks

# dataset location/root_dir
dataset_name = '2021-03-26_MFS_Horehound'
root_dir = os.path.join('/home/dorian/Data/agkelpie', dataset_name)
# img folder
# img_dir = os.path.join(root_dir, 'images_test')
# annotation folder/file
ann_dir = os.path.join(root_dir, 'metadata')
ann_file = '2021-03-26_MFS_Horehound_balanced_val.json'
ann_path = os.path.join(ann_dir, ann_file)

print(ann_file)

# ========================================================================
# load annotations file
ann_dict = json.load(open(ann_path))
ann_list = list(ann_dict.values())

# iterate through ann_list, if ann has regions, it is positive, else negative
ann_pos = {}
idx_pos = []
ann_neg = {}
idx_neg = []
PPT = PreProcessingToolbox()
for i, ann in enumerate(ann_list):
    img_name = ann['filename']
    reg = ann['regions']
    if bool(reg):
        # if regions is not empty, we have a positive image
        idx_pos.append(i)
        ann_pos = PPT.sample_dict(ann_pos, ann)
    else:
        idx_neg.append(i)
        ann_neg = PPT.sample_dict(ann_neg, ann)

print(f'pos images: {len(ann_pos)}')
print(f'neg images: {len(ann_neg)}')


import code
code.interact(local=dict(globals(), **locals()))

