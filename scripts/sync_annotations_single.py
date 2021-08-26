#! /usr/bin/env python

"""
sync annotations file for single image folder
"""

import os
import json
from weed_detection.PreProcessingToolbox import PreProcessingToolbox

dataset_name = 'Tussock_v4_poly'

# folder locations and file names
root_dir = os.path.join('/home',
                        'dorian',
                        'Data',
                        'AOS_TussockDataset',
                        dataset_name)



# corresponding annotations file to Images/All
ann_all_file = 'annotations_tussock_21032526_G507_all.json'

# annotation files Master (contains all images - we don't touch this file, just
# use it as a reference/check)
ann_master_file = 'annotations_tussock_21032526_G507_master.json'
ann_dir = os.path.join(root_dir, 'Annotations')
ann_master_path = os.path.join(ann_dir, ann_master_file)

# annotation files out
# ann_file_out = 'annotations_tussock_21032526_G507_polysubset.json'
ann_file_out = 'annotations_tussock_21032526_G507_allpoly.json'
ann_path_out = os.path.join(ann_dir, ann_file_out)

# folder containing all images to be used for testing/training/validation
img_folder = os.path.join(root_dir, 'Images', 'All')

# use preprocessing toolbox
ProTool = PreProcessingToolbox()
res = ProTool.sync_annotations(img_folder, ann_master_path, ann_path_out)

img_files = os.listdir(img_folder)
n_imgs = len(img_files)
print(f'img folder count = {n_imgs}')


ann_dict = json.load(open(ann_path_out))

n_ann_file = len(ann_dict)
print(f'ann file count = {n_ann_file}')

print('should be the same')

import code
code.interact(local=dict(globals(), **locals()))
