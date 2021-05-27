#! /usr/bin/env python

""" script to test image size """

import os
from weed_detection.WeedModel import WeedModel as WM



# setup parameters/folders:
dataset_folder = 'Tussock_v2_mini'
root_dir = os.path.join('/home', 'dorian','Data','AOS_TussockDataset', dataset_folder)

ann_files = [os.path.join(root_dir, 'Annotations', 'annotations_tussock_21032526_G507_train.json'),
            os.path.join(root_dir, 'Annotations', 'annotations_tussock_21032526_G507_test.json'),
            os.path.join(root_dir, 'Annotations', 'annotations_tussock_21032526_G507_val.json')]

img_folders = [os.path.join(root_dir, 'Images','Train'),
               os.path.join(root_dir, 'Images', 'Test'),
               os.path.join(root_dir, 'Images', 'Validation')]

# setup model for rescale sizes of:
image_sizes = [256, 512, 1024, 2056]

model_names = []
for i in range(len(image_sizes)):
    model_names.append(dataset_folder + '_' + str(image_sizes[i]))

dataset_names = model_names


# set hyper parameters of dataset
batch_size = 10
num_workers = 10
learning_rate = 0.005
momentum = 0.9
weight_decay = 0.0001
num_epochs = 100
step_size = round(num_epochs / 2)
shuffle = True

for i in range(len(image_sizes)):
    rescale_size = image_sizes[i]

    #  make a hyperparameter dictionary
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

    # create dataset, which defines the hyperparameters for rescale size
    WeedModel = WM(model_name=model_names[i])
    ds_path = WeedModel.create_train_test_val_datasets(img_folders, ann_files, hp, dataset_names[i])

    # train model
    WeedModel.train(model_name=model_names[i], dataset_path=ds_path)

    # delete weed model, because of space
    del(WeedModel)
    


# train model with rescale size set for each scale index
# do model comparison on each model with the same datasets (including both positive and negative images)

# get PR curve for all

# also, get a model inference times, (JUST model inference)
# plot on a graph vs time to compute for a single image

# also, create a mini test set of 100 images for training faster!

