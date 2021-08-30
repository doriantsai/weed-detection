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
root_dir = os.path.join('/home/dorian/Data/AOS_TussockDataset/03_Tagged/2021-03-25/Location_1')
# root_dir = os.path.join('/home/dorian/Data/AOS_TussockDataset/03_Tagged/2021-03-25/Location_2')
# root_dir = os.path.join('/home/dorian/Data/AOS_TussockDataset/03_Tagged/2021-03-26/Location_1')

# folder containing all images to be used for development (testing/training/val) and deployment
img_folder = os.path.join(root_dir, 'images')
ann_dir = os.path.join(root_dir, 'metadata')
print('img_dir = ' + img_folder)

# annotation files Master (contains all images - we don't touch this file, just
# use it as a reference/check)
ann_master_dir = os.path.join('/home/dorian/Data/AOS_TussockDataset/Tussock_v4_poly/Annotations')
ann_master_file = 'annotations_tussock_21032526_G507_master1.json'  # has the polygons
ann_master_path = os.path.join(ann_master_dir, ann_master_file)
print('ann_master_file = ' + ann_master_file)
# annotation files out

ann_file_out = 'Thursday_25-03-21_G507_location1_positive-tags_labels_polygons.json'
# ann_file_out = 'Thursday_25-03-21_G507_location2_positive-tags_labels_polygons.json'
# ann_file_out = 'Thursday_26-03-21_G507_location1_positive-tags_labels_polygons.json'
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

print('should be the same')

import code
code.interact(local=dict(globals(), **locals()))
