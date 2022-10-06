#! /usr/bin/env python

"""
sync annotations file for single image folder
img_folder = folder of images that we want a matching annotations file
ann_master_file = annotations file that does have all the annotations for all images in database
ann_file_out = output annotations corresponding to images in img_folder
"""

import os
import json
from weed_detection.PreProcessingToolbox import PreProcessingToolbox

# folder locations and file names
root_dir = os.path.join('/home/agkelpie/Data/03_Tagged/2021-10-13/Yellangelo/Serrated Tussock/')

# folder containing all images to be used for development (testing/training/val) and deployment
img_folder = os.path.join(root_dir, 'images')
ann_dir = os.path.join(root_dir, 'metadata')
print('img_dir = ' + img_folder)

# annotation files Master (contains all images - we don't touch this file, just
# use it as a reference/check)
ann_master_dir = os.path.join('/home/agkelpie/Data/03_Tagged/2021-10-13/Yellangelo/Serrated Tussock/metadata')
ann_master_file = 'Yellangelo-Final.json'  # has the polygons, now has is_processed
ann_master_path = os.path.join(ann_master_dir, ann_master_file)
print('ann_master_file = ' + ann_master_file)
# annotation files out

ann_file_out = 'test.json'
ann_path_out = os.path.join(ann_dir, ann_file_out)
print('ann_file_out = ' + ann_file_out)

# use preprocessing toolbox
ProTool = PreProcessingToolbox()
res = ProTool.sync_annotations(img_folder, ann_master_path, ann_path_out)

img_files = os.listdir(img_folder)
n_imgs = len(img_files)
print(f'img folder count = {n_imgs}')

ann_dict = json.load(open(ann_path_out))
n_ann_file = len(ann_dict)
print(f'ann file count = {n_ann_file}')

print('exect the count to be different, due to DPI skipping images')

print('TODO: count the is_processed tag, then compare')
# start looking for is_processed==1 images:

ann_list = list(ann_dict.values())

# find all dictionary entries that match img_dir
img_proc = [s['filename'] for s in ann_list if s['file_attributes']['is_processed'] == str(1) ]
len(img_proc)


import code
code.interact(local=dict(globals(), **locals()))
