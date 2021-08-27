#! /usr/bin/env python

""" script to generate a single pr curve """

import os
import pickle
import numpy as np
from weed_detection.WeedModel import WeedModel as WM

# init WM object
# load model
# call prcurve function

# load dataset objects
dataset_name = 'Tussock_v3_neg_test'
dataset_file = os.path.join('dataset_objects', dataset_name, dataset_name + '.pkl')

# load dataset files via unpacking the pkl file
WeedModel = WM()
print('loading dataset: {}'.format(dataset_file))
dso = WeedModel.load_dataset_objects(dataset_file)

dataset = dso['ds_train']
# test dataset - should be orderd
i = 0
for image, sample in dataset:
    image_name = sample['image_id']
    print('{}: image_id: {}'.format(i, image_name))

    # if i > 1:
    #     break
    i += 1