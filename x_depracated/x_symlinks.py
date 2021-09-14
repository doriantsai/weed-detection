#! /usr/bin/env python

"""
script to create symbolic links of image files given a json file
"""

# given json file
# given img_dir of said files, which might be in multiple img_dir
# (eg, subfolders)
# speficied in Tussock_v2, Day0, Day1, Day2
# create symbolic links into making new folders for
# Sym0

import os
import json
import glob

# dataset location
root_dir = os.path.join('/home/dorian/Data/agkelpie/Tussock_v2')

# annotations
ann_master = 'annotations_tussock_21032526_G507_allpoly.json'
ann_path = os.path.join(root_dir, 'Annotations', ann_master)
ann_dict = json.load(open(ann_path))
ann_list = list(ann_dict.values())

# first, find all subfolders that start with "DayX" thave have images in them
res = glob.glob('/home/dorian/Data/agkelpie/Tussock_v2/Images/Day*/*.png', recursive=True)

for r in res:
    print(res)

# get just the image names
img_names = []
for i, r in enumerate(res):
    img_name = os.path.basename(r)
    print(f'{i}\t{img_name}')
    img_names.append(img_name)

# TODO apply filter to res or img_names
# eg, specific day or location
# NOTE ideally, the json file has this sort of information, and we can reference it to filter/limit our image selection

# create sym link folder
save_folder = os.path.join(root_dir, 'Sym0')
os.makedirs(save_folder, exist_ok=True)

# create symlinks
for i, img in enumerate(img_names):
    dst_file = os.path.join(save_folder, img)
    if os.path.exists(dst_file):
        os.unlink(dst_file)
    else:
        os.symlink(os.path.join(res[i]), dst_file)

print('done sym linking')

import code
code.interact(local=dict(globals(), **locals()))
