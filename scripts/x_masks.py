#! /usr/bin/env python

"""
debugging masks
"""

import os
import json
from weed_detection.PreProcessingToolbox import PreProcessingToolbox
from weed_detection.WeedModel import WeedModel
from weed_detection.WeedDatasetPoly import Compose, Rescale, ToTensor

CREATE_MASKS = False

# folder/file locations/paths
dataset_name = 'Tussock_v4_poly'

# folder locations and file names
root_dir = os.path.join('/home',
                        'dorian',
                        'Data',
                        'AOS_TussockDataset',
                        dataset_name)
ann_dir = os.path.join(root_dir, 'Annotations')

ann_file = '20210819-MFS-01-bootprogress-570-occlusion24_json.json'
ann_path = os.path.join(ann_dir, ann_file)


img_dir = os.path.join(root_dir, 'Images', 'All')

ann_file_out = 'test_masks.json'
ann_out_path = os.path.join(ann_dir, ann_file_out)
ProTool = PreProcessingToolbox()
ann_out_file = ProTool.sync_annotations(img_dir, ann_path, ann_out_path)

# create masks folder
mask_dir = os.path.join(root_dir, 'Masks')
mask_dir_all = os.path.join(mask_dir, 'All')
if CREATE_MASKS:
    # make masks

    ProTool.create_masks_from_poly(img_dir, ann_out_file, mask_dir_all)

    # check how many images there are
    img_list = os.listdir(img_dir)

    # check how many masks there are:
    mask_list = os.listdir(mask_dir_all)

    print(f'number of images: {len(img_list)}')
    print(f'number of masks: {len(mask_list)}')

# so we have just the polygons in test_masks.json
# now create a dataset for this json
Tussock = WeedModel()
# set hyper parameters of dataset
batch_size = 10
num_workers = 10
learning_rate = 0.005 # 0.002
momentum = 0.9 # 0.8
weight_decay = 0.0001
num_epochs = 200
step_size = round(num_epochs / 2)
shuffle = True
rescale_size = int(1024)
# make a hyperparameter dictionary
hp={}
hp['batch_size'] = batch_size
hp['num_workers'] = num_workers
hp['learning_rate'] = learning_rate
hp['momentum'] = momentum
hp['step_size'] = step_size
hp['weight_decay'] = weight_decay
hp['num_epochs'] = num_epochs
hp['shuffle'] = shuffle
hp['rescale_size'] = rescale_size
tforms = Compose([Rescale(rescale_size),
                  ToTensor()])
ds, dl = Tussock.create_dataset_dataloader(root_dir=root_dir,
                                          json_file=ann_file_out,
                                          transforms=tforms,
                                          hp=hp,
                                          annotation_type='poly',
                                          img_dir=img_dir,
                                          mask_dir=mask_dir_all)

# try dataset, get_item
i = 100
ds[i]

import code
code.interact(local=dict(globals(), **locals()))
