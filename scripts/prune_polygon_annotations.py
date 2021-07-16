#! /usr/bin/env python

"""
script to prune annotations file to just polygons
"""

# load json file
# for each annotation
#   sort through each region
#   check if number of box/poly/point makes three
#   else return warning/note/print which file
#   regardless, for each region, compute centroid
#   compute distances to each centroid
#   based on some threshold, declare correspondence

import os
import json
from weed_detection.PreProcessingToolbox import PreProcessingToolbox

# dataset location
# folder = os.path.join('/home', 'dt-cronus', 'Dropbox', 'QUT_WeedImaging')
folder = os.path.join('/home', 'dorian', 'Data', 'AOS_TussockDataset')
dataset_name = 'Tussock_v4_poly286'
root_dir = os.path.join(folder, dataset_name)
ann_dir = os.path.join(root_dir, 'Annotations')
img_dir = os.path.join(root_dir, 'Images', 'All')

# folder for polygon images
# poly_dir = os.path.join(root_dir, 'Images', 'All_Poly')
# os.makedirs(poly_dir, exist_ok=True)

# location of json file + name
ann_master_file = 'via_project_07Jul2021_08h00m_240_test.json'
ann_master_path = os.path.join(ann_dir, ann_master_file)

# I think I just have to sync the json file with the images in the All folder
ann_file_out = 'via_project_07Jul2021_08h00m_240_test_justpoly.json'
ann_path_out = os.path.join(ann_dir, ann_file_out)
PreTool = PreProcessingToolbox()
PreTool.sync_annotations(img_dir, ann_master_path, ann_path_out)

# check, how many items in ann_file?
json_name = ann_file_out
json_path = os.path.join(ann_dir, json_name)

ann_dict = json.load(open(json_path))
ann_list = list(ann_dict.values())
print(f'length of ann_list = {len(ann_list)}')
print('should be = 286, number of images in img_dir')
img_list = os.listdir(img_dir)
print(f'number of images in img_dir = {len(img_list)}')

print('done syncing polygon annotations with image directory')
import code
code.interact(local=dict(globals(), **locals()))
