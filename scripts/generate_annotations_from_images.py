#! /usr/bin/env python

"""
generate annotations file given images in a folder and master annotations file
NOTE this is exactly sync_annotations, that's silly DEPRACATED
TODO fit this into PreProcessingToolbox
"""

import json
import os
from weed_detection.PreProcessingToolbox import PreProcessingToolbox

# define master annotations file, has all the latest
# find out all images in given directory --> images_list
# get all annotations from master_anno that correspond to images_list
# output them into single annotations file to out_dir

data_dir = '/home/dorian/Data/AOS_TussockDataset/Tussock_v4_poly'

# annotation master file, has all the latest annotations
ann_master = 'annotations_tussock_21032526_G507_all1.json'
ann_master_path = os.path.join(data_dir, 'Annotations', ann_master)
ann_master_dict = json.load(open(ann_master_path))
ann_master_list = list(ann_master_dict.values())

# get annotation image names as a list
ann_master_names = [a['filename'] for a in ann_master_list]

# image folder, for which we need to create an annotations file
img_dir = os.path.join(data_dir, 'Images', 'All')
img_list = os.listdir(img_dir)

# print image names
print('image names')
for i, img in enumerate(img_list):
    print(f'{i}:\t{img}')

# create a dictionary with keys: list entries as values are to indices
ind_dict = dict((k, i) for i, k in enumerate(ann_master_names))

# find intersection set
# img_match = set(img_list) & set(ann_master_names)
img_match = set(ind_dict).intersection(img_list)
img_match.sort()

print(f'matching images: {len(img_match)}')

# compile list of indices of the intersection
indices = [ind_dict[x] for x in img_match]


# for each img in img_match, take the corresponding annotation and apply it to dictionary
PPT = PreProcessingToolbox()
ann_out_dict = {}
for i in indices:
    sample = ann_master_list[i]
    # create new annotations file
    ann_dict = PPT.sample_dict(ann_out_dict, sample)

# create annotations file

import code
code.interact(local=dict(globals(), **locals()))
