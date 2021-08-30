#! /usr/bin/env python

"""
train FasterRCNN pipeline

given json file, appropriate images folder
- sync annotations file with image folder
- split image data
- create dataset objects
- train model
"""
import os
import json
from weed_detection.PreProcessingToolbox import PreProcessingToolbox
from weed_detection.WeedModel import WeedModel

# folder/file locations/paths
dataset_name = 'Tussock_v4_poly'

# folder locations and file names
root_dir = os.path.join('/home',
                        'dorian',
                        'Data',
                        'AOS_TussockDataset',
                        dataset_name)
ann_dir = os.path.join(root_dir, 'Annotations')

ann_file = 'annotations_tussock_21032526_G507_master1.json'
ann_path = os.path.join(ann_dir, ann_file)


img_dir = os.path.join(root_dir, 'Images', 'All')

# ============================================================
# sync annotation file with images 
ann_file_out = 'annotations_tussock_21032526_G507_all1.json'
ann_out_path = os.path.join(ann_dir, ann_file_out)
ProTool = PreProcessingToolbox()
ann_out_file = ProTool.sync_annotations(img_dir, ann_path, ann_out_path)


# ============================================================
# split image data
print('splitting image data')
ann_all_file = 'annotations_tussock_21032526_G507_all1.json'

# annotation files Master (contains all images - we don't touch this file, just
# use it as a reference/check)
ann_master_file = 'annotations_tussock_21032526_G507_master1.json'  # we are using master file as allpoly, because it contains all images

# annotation files out
# TODO will want to use the same images (json files) as maskrcnn
ann_train_file = 'annotations_tussock_21032526_G507_train.json'
ann_test_file = 'annotations_tussock_21032526_G507_test.json'
ann_val_file = 'annotations_tussock_21032526_G507_val.json'

img_folders, ann_files = ProTool.split_image_data(root_dir,
                                                    img_dir,
                                                    ann_master_file,
                                                    ann_all_file,
                                                    ann_train_file,
                                                    ann_val_file,
                                                    ann_test_file)


# ============================================================

print('creating datasets')

# set hyper parameters of dataset
batch_size = 10
num_workers = 10
learning_rate = 0.005 # 0.002
momentum = 0.9 # 0.8
weight_decay = 0.0001
num_epochs = 100
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

hp_train = hp
hp_test = hp
hp_test['shuffle'] = False

# init object
Tussock_FasterRCNN = WeedModel()
# save all datasets/dataloaders in a .pkl file
# dataset_name_save = dataset_name + '_shortgrass'
dataset_name_save = dataset_name + '_FasterRCNN'
dataset_path = Tussock_FasterRCNN.create_train_test_val_datasets(img_folders,
                                                      ann_files,
                                                      hp,
                                                      dataset_name_save,
                                                      annotation_type='box')

# ============================================================
# train model
model, model_save_path = Tussock_FasterRCNN.train(model_name = dataset_name,
                                       dataset_path=dataset_path)
print('finished training model: {0}'.format(model_save_path))


import code
code.interact(local=dict(globals(), **locals()))
