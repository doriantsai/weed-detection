#! /usr/bin/env python

""" script to create dataset/dataloader objects using output from
PPT.split_image_data() """

import os
import weed_detection.WeedModel as WeedModel

# setup folder locations init object call object

# NOTE: copied/pasted output from PPT.split_iamge_data()
# dataset_name = 'Tussock_v3_augment'
dataset_name = 'Tussock_v0_mini'
# dataset_name = 'Tussock_v3_neg_train_test'
root_dir = os.path.join('/home',
                        'dorian',
                        'Data',
                        'AOS_TussockDataset',
                        dataset_name)
img_folders = [os.path.join(root_dir, 'Images','Train'),
               os.path.join(root_dir, 'Images', 'Test'),
               os.path.join(root_dir, 'Images', 'Validation')]

mask_dir = os.path.join(root_dir, 'Masks')
mask_folders = [os.path.join(mask_dir, 'Train'),
               os.path.join(mask_dir, 'Test'),
               os.path.join(mask_dir, 'Validation')]
all_mask_dir = os.path.join(mask_dir, 'All')


# ann_files = [os.path.join(root_dir, 'Annotations', 'annotations_train_augmented_combined.json'),
#             os.path.join(root_dir, 'Annotations', 'annotations_tussock_21032526_G507_test.json'),
#             os.path.join(root_dir, 'Annotations', 'annotations_tussock_21032526_G507_val.json')]

ann_files = [os.path.join(root_dir, 'Annotations', 'annotations_tussock_21032526_G507_train.json'),
            os.path.join(root_dir, 'Annotations', 'annotations_tussock_21032526_G507_test.json'),
            os.path.join(root_dir, 'Annotations', 'annotations_tussock_21032526_G507_val.json')]

# set hyper parameters of dataset
batch_size = 10
num_workers = 10
learning_rate = 0.005
momentum = 0.9
weight_decay = 0.0001
num_epochs = 50
step_size = round(num_epochs / 2)
shuffle = True
rescale_size = int(256)

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

# dataset_name = 'Tussock_v2'

# init object
Tussock = WeedModel()
# save all datasets/dataloaders in a .pkl file
dataset_path = Tussock.create_train_test_val_datasets(img_folders,
                                                      ann_files,
                                                      hp,
                                                      dataset_name,
                                                      annotation_type='poly',
                                                      mask_folders=mask_folders)

# TODO open pkl file and confirm that datasets match image length, but some are
# "augmented"?
print('dataset_path = {}'.format(dataset_path))

# test forward pass
dso = Tussock.load_dataset_objects(dataset_path)
dataset = dso['ds_train']
dataloader = dso['dl_train']


import code
code.interact(local=dict(globals(), **locals()))