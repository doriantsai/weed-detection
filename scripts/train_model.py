#! /usr/bin/env python


""" script to train model after running split_image_data.py and create_datasets.py """

import os
import time
import pickle

# from weed_detection.WeedDataset import WeedDataset as WD
from weed_detection.WeedModel import WeedModel as WM


# create datasets
# folder locations of dataset files
dataset_file = os.path.join('dataset_objects', 'Tussock_v1', 'Tussock_v1.pkl')

# load dataset files via unpacking the pkl file
if os.path.isfile(dataset_file):
    with open(dataset_file, 'rb') as f:
        ds_train = pickle.load(f)
        ds_test = pickle.load(f)
        ds_val = pickle.load(f)
        dl_train = pickle.load(f)
        dl_test = pickle.load(f)
        dl_val = pickle.load(f)
        hp_train = pickle.load(f)
        hp_test = pickle.load(f)
        dataset_name = pickle.load(f)

# create WM object
Tussock = WM()

# call WM.train
Tussock.train(model_name='tussock_test',
              dataset_path=dataset_file)

print(Tussock.model_name)
print(Tussock.model_path)

import code
code.interact(local=dict(globals(), **locals()))