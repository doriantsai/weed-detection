#! /usr/bin/env python

""" script to generate a single pr curve """

import os
import pickle
from weed_detection.WeedModel import WeedModel as WM

# init WM object
# load model
# call prcurve function

# load dataset objects
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

# init WM object
Tussock = WM()

# load model
save_model_path = os.path.join('output', 
                               'tussock_test_2021-05-16_16_13', 
                               'tussock_test.pth')
Tussock.load_model(save_model_path)

# TODO call prcurve functions