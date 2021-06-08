#! /usr/bin/env python

""" script to split iamge data by calling PreProcessingToolbox.split_image_data
"""

import os
from weed_detection.PreProcessingToolbox import PreProcessingToolbox


# set folder locations init PT object PT.split_image_data

dataset_name = 'Tussock_v2'
# folder locations and file names
root_dir = os.path.join('/home',
                        'dorian',
                        'Data',
                        'AOS_TussockDataset',
                        dataset_name)

# folder containing all images to be used for testing/training/validation
all_folder = os.path.join(root_dir, 'Images', 'All')
# corresponding annotations file to Images/All
# ann_all_file = 'annotations_tussock_21032526_G507_all.json'
ann_all_file = 'via_project_29Apr2021_17h43m_json_bbox_poly_pt.json'

# annotation files Master (contains all images - we don't touch this file, just
# use it as a reference/check)
# ann_master_file = 'annotations_tussock_21032526_G507_master.json'
ann_master_file = 'via_project_29Apr2021_17h43m_json_bbox_poly_pt.json'

# annotation files out
ann_train_file = 'annotations_tussock_21032526_G507_train.json'
ann_test_file = 'annotations_tussock_21032526_G507_test.json'
ann_val_file = 'annotations_tussock_21032526_G507_val.json'


# create PT object
ProTool = PreProcessingToolbox()
img_folders, ann_files = ProTool.split_image_data(root_dir,
                                                    all_folder,
                                                    ann_master_file,
                                                    ann_all_file,
                                                    ann_train_file,
                                                    ann_val_file,
                                                    ann_test_file)

# NOTE split_image_data calls sync_annotations ensure that image folders and
# annotation files are in sync here, we will just check by counting the number
# of images in all_folder and comparing them to the sum of the number of images
# in img_folders, which should be equal

# we will also check/compare the length of the annotation files

all_img_files = os.listdir(all_folder)
n_all = len(all_img_files)
print('all_folder image count: {}'.format(n_all))

train_img_files = os.listdir(img_folders[0])
n_train = len(train_img_files)
print('train_folder image count: {}'.format(n_train))

test_img_files = os.listdir(img_folders[1])
n_test = len(test_img_files)
print('test_folder image count: {}'.format(n_test))

val_img_files = os.listdir(img_folders[2])
n_val = len(val_img_files)
print('val_folder image count: {}'.format(n_val))

n_sum = n_train + n_test + n_val
if n_sum == n_all:
    print('success: n_sum == n_all')
else:
    print('warning: n_sum != n_all')
    print('n_sum = {}'.format(n_sum))
    print('n_all = {}'.format(n_all))

import code
code.interact(local=dict(globals(), **locals()))

