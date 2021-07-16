#! /usr/bin/env python


""" script to train model after running split_image_data.py and create_datasets.py """

import os
# import time
# import pickle

# from weed_detection.WeedDataset import WeedDataset as WD
from weed_detection.WeedModel import WeedModel as WM


# create WM object
Tussock = WM()

# folder locations of dataset files
# dataset_name = 'Tussock_v3_neg_test'
# dataset_name = 'Tussock_v2'
# dataset_name = 'Tussock_v3_augment'
# dataset_name = 'Tussock_v0_mini'
dataset_name = 'Tussock_v4_poly286'
dataset_file = os.path.join('dataset_objects', dataset_name, dataset_name + '.pkl')

# call WM.train
Tussock.train(model_name=dataset_name,
              dataset_path=dataset_file)


import code
code.interact(local=dict(globals(), **locals()))