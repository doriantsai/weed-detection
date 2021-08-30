#! /usr/bin/env python

"""
script to generate tussock dataset from weeds reference library setup,
grab all image files
grab all json files
create one large merged json file for entire dataset
create specified dataset folder and populate with simlink of images
for all relevant images based on specific days/folders
"""

import os
import json
import glob
from weed_detection.PreProcessingToolbox import PreProcessingToolbox

# given folder locations
# given types of files to look for (glob) --> images/annotations
# glob all relevant files
# combine into singular json file with folder locations (might require new tag added)
# output all photos into single folder via symlinks

# dataset location/root_dir
dataset_name = '2021-03-25_MFS_Tussock'
root_dir = os.path.join('/home/dorian/Data/AOS_TussockDataset', dataset_name)
# img folder
img_dir = os.path.join(root_dir, 'images')
# annotation folder/file
ann_dir = os.path.join(root_dir, 'metadata')
ann_file = '2021-03-25_Tussock_MFS.json'
ann_path = os.path.join(ann_dir, ann_file)

# TODO want to filter the negative images vs positive images and put them into respective folders
ann_dict = json.load(open(ann_path))
ann_list = list(ann_dict.values())

# iterate through ann_list, if ann has regions, it is positive, else negative
ann_pos = {}
ann_neg = {}
# TODO I am here

# create folder for symlinks
out_dir = os.path.join('/home/dorian/Data/AOS_TussockDataset/2021-03-25_Tussock_MFS')
img_pos_out_dir = os.path.join(out_dir, 'images_pos')
img_neg_out_dir = os.path.join(out_dir, 'images_neg')
ann_out_dir = os.path.join(out_dir, 'metadata')
os.makedirs(img_pos_out_dir, exist_ok=True)
os.makedirs(img_neg_out_dir, exist_ok=True)
os.makedirs(ann_out_dir, exist_ok=True)

# create symlinks
print('creating sym links')
for i, img_path in enumerate(glob_img):
    # print(f'{i+1} / {len(glob_img)}')
    img_name = os.path.basename(img_path)

    dst_file = os.path.join(img_out_dir, img_name)
    if os.path.exists(dst_file):
        os.unlink(dst_file)
    else:
        os.symlink(img_path, dst_file)

# check number of symlinks in folder:
print(f'num images in glob = {len(glob_img)}')
img_list = os.listdir(img_out_dir)
print(f'num imgs in out_dir = {len(img_list)}')
print('should be the same')

# merge annotations into one:
ann_out = '2021-03-25_Tussock_MFS.json'
ann_out_path = os.path.join(ann_out_dir, ann_out)
ProTool = PreProcessingToolbox()
res = ProTool.combine_annotations(ann_files=glob_ann,
                                  ann_dir=False,
                                  ann_out=ann_out_path)

# check number of items in annotation file
ann_check = json.load(open(ann_out_path))
print(f'num entries in ann_out = {len(ann_check)}')

import code
code.interact(local=dict(globals(), **locals()))
