#! /usr/bin/env python

""" script to create dataset/dataloader objects using output from
PPT.split_image_data() """

import os
import weed_detection.WeedModel as WeedModel

# setup folder locations
# init object
# call object

# NOTE: copied/pasted output from PPT.split_iamge_data()
img_folders = ['/home/dorian/Data/AOS_TussockDataset/Tussock_v1/Images/Train',
                '/home/dorian/Data/AOS_TussockDataset/Tussock_v1/Images/Test',
                '/home/dorian/Data/AOS_TussockDataset/Tussock_v1/Images/Validation']

ann_files = ['/home/dorian/Data/AOS_TussockDataset/Tussock_v1/Annotations/annotations_tussock_21032526_G507_train.json',
            '/home/dorian/Data/AOS_TussockDataset/Tussock_v1/Annotations/annotations_tussock_21032526_G507_test.json',
            '/home/dorian/Data/AOS_TussockDataset/Tussock_v1/Annotations/annotations_tussock_21032526_G507_val.json']

# set hyper parameters of dataset
batch_size = 2
num_workers = 10
learning_rate = 0.005
momentum = 0.9
weight_decay = 0.0001
num_epochs = 10
step_size = round(num_epochs / 2)
shuffle = True
rescale_size = 2056

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

dataset_name = 'Tussock_v1'

# init object
Tussock = WeedModel()
# save all datasets/dataloaders in a .pkl file
dataset_path = Tussock.create_train_test_val_datasets(img_folders,
                                                      ann_files,
                                                      hp,
                                                      dataset_name)

# TODO open pkl file and confirm that datasets match image length, but some are "augmented"?
print('dataset_path = {}'.format(dataset_path))



