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
import random
import numpy as np
import shutil


def copy_symlinks_from_dict(ann_in, img_src_dir, img_dst_dir):
    os.makedirs(img_dst_dir, exist_ok=True)
    ann_list = list(ann_in.values())
    for ann in ann_list:
        img_name = ann['filename'] # looping over dictionary gives just the key?
        img_path = os.path.join(img_src_dir, img_name)
        dst_file = os.path.join(img_dst_dir, img_name)
        if os.path.exists(dst_file):
            os.unlink(dst_file)
        # create sym link
        os.symlink(img_path, dst_file)

    return True

def create_sym_ann(img_in_dir, img_out_dir, ann_in_dict, ann_file_out):
    
    # os.makedirs(img_out_dir, exist_ok=True)
    # create symlinks
    # print('creating sym links')
    # TODO make function for ann_pos, vs negative case
    copy_symlinks_from_dict(ann_in_dict, img_in_dir, img_out_dir)

    # create annotations file
    ann_in_path = os.path.join(ann_file_out)
    with open(ann_in_path, 'w') as f:
        json.dump(ann_in_dict, f, indent=4)
    
    return ann_in_path

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
ann_file = '2021-03-25_MFS_Tussock.json'
ann_path = os.path.join(ann_dir, ann_file)


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


# create folder for symlinks
print('create folders')
out_dir = os.path.join('/home/dorian/Data/AOS_TussockDataset/2021-03-25_MFS_Tussock')


ann_out_dir = os.path.join(out_dir, 'metadata')
os.makedirs(ann_out_dir, exist_ok=True)

img_neg_out_dir = os.path.join(out_dir, 'images_neg')
img_pos_out_dir = os.path.join(out_dir, 'images_pos')

# create symbolic links and annotations files
ann_pos_out = ann_file[:-5] + '_pos.json'
ann_neg_out = ann_file[:-5] + '_neg.json'
ann_pos_out = os.path.join(ann_dir, ann_pos_out)
ann_neg_out = os.path.join(ann_dir, ann_neg_out)
ann_pos_out = create_sym_ann(img_dir, img_pos_out_dir, ann_pos, ann_pos_out)
ann_neg_out = create_sym_ann(img_dir, img_neg_out_dir, ann_neg, ann_neg_out)

# check number of symlinks in folder:
print(f'num images in ann_pos = {len(ann_pos)}')
img_list = os.listdir(img_pos_out_dir)
print(f'num imgs in img_pos_out_dir = {len(img_list)}')
print('should be the same')

print(f'num images in ann_neg = {len(ann_neg)}')
img_list = os.listdir(img_neg_out_dir)
print(f'num imgs in img_neg_out_dir = {len(img_list)}')
print('should be the same')

# ========================================================================
# now we want to par down negative images to 50:50 of img_pos
# want to randomly select/remove negative images
ann_neg_list = list(ann_neg.values())
n_delete = len(ann_neg) - len(ann_pos)
to_delete = set(random.sample(range(len(ann_neg)), n_delete))
ann_neg_trim_list = [x for k, x in enumerate(ann_neg_list) if not k in to_delete]
print(f'len ann_neg_trim = {len(ann_neg_trim_list)}')

# convert ann_neg_trim_list to dictionary
ann_neg_trim = {}
for ann in ann_neg_trim_list:
    ann_neg_trim = PPT.sample_dict(ann_neg_trim, ann)

# need to clear out img_neg_trim_dir before loading it up
img_neg_trim_dir = os.path.join(out_dir, 'images_neg_trim')
if os.path.isdir(img_neg_trim_dir):
    shutil.rmtree(img_neg_trim_dir)
# os.makedirs(img_neg_trim_dir, exist_ok=True)

ann_neg_trim_out = os.path.join(ann_neg_out[:-5] + '_trim.json')
ann_neg_trim_out = create_sym_ann(img_neg_out_dir, img_neg_trim_dir, ann_neg_trim, ann_neg_trim_out)

print(f'num images in ann_neg_trim = {len(ann_neg_trim)}')
img_list = os.listdir(img_neg_trim_dir)
print(f'num imgs in img_neg_trim_dir = {len(img_list)}')
print('should be the same')


# ========================================================================

# finally, we combine positive, negative image directories and their respective json files
# img_neg_trim_dir
# img_pos_out_dir

img_dir_out = os.path.join(out_dir, 'images_balanced')
if os.path.isdir(img_dir_out):
    shutil.rmtree(img_dir_out)
# os.makedirs(img_dir_out, exist_ok=True)

img_pos_list = os.listdir(img_pos_out_dir)
img_neg_list = os.listdir(img_neg_trim_dir)

# unfortunately, copytree copies the actual files, not the symlinks
# shutil.copytree(src=img_pos_out_dir, dst=img_dir_out, dirs_exist_ok=True)
# shutil.copytree(src=img_neg_trim_dir, dst=img_dir_out, dirs_exist_ok=True)
copy_symlinks_from_dict(ann_pos, img_pos_out_dir, img_dir_out)
copy_symlinks_from_dict(ann_neg_trim, img_neg_trim_dir, img_dir_out)

img_out_list = os.listdir(img_dir_out)

print(f'img_pos_list +  img_neg_list = {len(img_pos_list)} + {len(img_neg_list)} = {len(img_pos_list) + len(img_neg_list)}')
print(f'img_out_list = {len(img_out_list)}')

# now, combine the annotations files
ann_combine = [ann_pos_out, ann_neg_trim_out]
ann_out = ann_file[:-5] + '_balanced.json'
ann_out = os.path.join(ann_dir, ann_out)
res = PPT.combine_annotations(ann_combine, ann_dir=False, ann_out=ann_out)

# check:
ann_out_dict = json.load(open(ann_out))
print(f'len of ann_out_dict = {len(ann_out_dict)}')

import code
code.interact(local=dict(globals(), **locals()))

