#! /usr/bin/env python

""" agkelpie weed pipeline,
    - create image folders from dataserver using symbolic links
    - create balanced image folders + annotation files
    - create masks
    - split image data
    - create dataset objects
    - call train model
    - generate pr curve
"""

import os
import weed_detection.WeedModel as WeedModel
from weed_detection.PreProcessingToolbox import PreProcessingToolbox

# setup file/folder locations
dataserver_dir = os.path.join('/home/dorian/agkelpie/03_Tagged')
dataset_name = '2021-03-26_MFS_Horehound_v0'

# glob string patterns to find the images and metadata (annotations) files, respectively
img_dir_patterns=['/2021-03-26/Location_2/images/']
ann_dir_patterns=['/2021-03-26/Location_2/metadata/']

ppt = PreProcessingToolbox()

# ==========================
# create symbolic links to image folders
# ==========================
ann_dataset_path, root_dir = ppt.generate_symbolic_links(dataserver_dir,
                                                        dataset_name,
                                                        img_dir_patterns,
                                                        ann_dir_patterns)

# ==========================
# create balanced image folder and annotation file from symbolic links
# ==========================
img_bal_dir, ann_bal_path = ppt.generate_dataset_from_symbolic_links(root_dir,
                                                             ann_dataset_path)

# ==========================
# create masks
# ==========================
model_type = 'poly'
res, mask_dir = ppt.create_masks_from_poly(img_bal_dir, ann_bal_path)

# ==========================
# split image data
# ==========================
# split into train/test/val folders w respective json files

ann_train_file = ann_bal_path[:-5] + '_train.json'
ann_test_file = ann_bal_path[:-5] + '_test.json'
ann_val_file = ann_bal_path[:-5] + '_val.json'

img_dirs, ann_files = ppt.split_image_data(root_dir,
                                        img_bal_dir,
                                        ann_bal_path,
                                        ann_bal_path,
                                        ann_train_file,
                                        ann_val_file,
                                        ann_test_file,
                                        annotation_type=model_type,
                                        mask_folder=mask_dir,
                                        ann_dir=False)

