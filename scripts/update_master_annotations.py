#! /usr/bin/env python

"""
script to take the latest annotations (polygons/points) and add them to the master file
"""

import os
import json
import copy
from weed_detection.PreProcessingToolbox import PreProcessingToolbox

# setup folders/files
# read in poly annotation file
# read in old master file (which has the negative images and all the positive images)
# for every image not in poly file, add into poly file
# this might be easier than finding and updating existing entries in the master file

# folder/file locations/paths
dataset_name = 'Tussock_v4_poly'

# folder locations and file names
root_dir = os.path.join('/home',
                        'dorian',
                        'Data',
                        'AOS_TussockDataset',
                        dataset_name)
ann_dir = os.path.join(root_dir, 'Annotations')

# corresponding annotations file to Images/All
ann_all_file = '20210819-MFS-01-bootprogress-570-occlusion24_json.json'
ann_all_path = os.path.join(ann_dir, ann_all_file)
# annotation files Master (contains all images - we don't touch this file, just
# use it as a reference/check)
ann_master_file_in = 'annotations_tussock_21032526_G507_master.json'
ann_master_file_out = 'annotations_tussock_21032526_G507_master1.json'

ann_master_path = os.path.join(ann_dir, ann_master_file_in)
ann_path_out = os.path.join(ann_dir, ann_master_file_out)

# load both annotation files
ann_poly_dict = json.load(open(ann_all_path))
ann_master_dict = json.load(open(ann_master_path))

# convert to lists
ann_poly = list(ann_poly_dict.values())
ann_master = list(ann_master_dict.values())

ann_poly_names = [s['filename'] for s in ann_poly]
ann_master_names = [s['filename'] for s in ann_master]

ann_poly_out = copy.copy(ann_poly_dict)
# use sets to find all image in ann_master not within ann_poly
ProTool = PreProcessingToolbox()
# ann_add = {}
for ann in ann_master:
    fname = ann['filename']
    if not fname in ann_poly_names:
        
        ann_poly_out = ProTool.sample_dict(ann_poly_out, ann)


print(f'length of ann_poly_out = {len(ann_poly_out)}')

# save annotations out file
print('writing annotations file out')
with open(ann_path_out, 'w') as f:
    json.dump(ann_poly_out, f, indent=4)

import code
code.interact(local=dict(globals(), **locals()))