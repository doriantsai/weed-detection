#! /usr/bin/env python

""" script to re-sync training and testing image data
"""

import os
from weed_detection.PreProcessingToolbox import PreProcessingToolbox


# set folder locations init PT object PT.split_image_data

# dataset_name = 'Tussock_v3_neg_test'
dataset_name = 'Tussock_v3_neg_train_test'

# folder locations and file names
root_dir = os.path.join('/home',
                        'dorian',
                        'Data',
                        'agkelpie',
                        dataset_name)

# folder containing all images to be used for testing/training/validation
all_folder = os.path.join(root_dir, 'Images', 'All')
# corresponding annotations file to Images/All
ann_all_file = 'annotations_tussock_21032526_G507_all.json'

# annotation files Master (contains all images - we don't touch this file, just
# use it as a reference/check)
ann_master_file = 'annotations_tussock_21032526_G507_master.json'
ann_dir = os.path.join(root_dir, 'Annotations')
ann_master_path = os.path.join(ann_dir, ann_master_file)

# annotation files out
ann_train_file = 'annotations_tussock_21032526_G507_train.json'
ann_test_file = 'annotations_tussock_21032526_G507_test.json'
ann_val_file = 'annotations_tussock_21032526_G507_val.json'

ann_paths = [os.path.join(ann_dir, ann_train_file),
             os.path.join(ann_dir, ann_test_file),
             os.path.join(ann_dir, ann_val_file)]

train_folder = os.path.join(root_dir, 'Images', 'Train')
test_folder = os.path.join(root_dir, 'Images', 'Test')
val_folder = os.path.join(root_dir, 'Images', 'Validation')

img_folders = [train_folder, test_folder, val_folder]

# create PT object
ProTool = PreProcessingToolbox()
res = ProTool.sync_annotations(img_folders[0], ann_master_path, ann_paths[0])
res = ProTool.sync_annotations(img_folders[1], ann_master_path, ann_paths[1])
res = ProTool.sync_annotations(img_folders[2], ann_master_path, ann_paths[2])


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

