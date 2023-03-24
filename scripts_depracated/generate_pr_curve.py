#! /usr/bin/env python

""" script to generate a single pr curve """

import os
import pickle
import numpy as np
from weed_detection.WeedModel import WeedModel as WM

# init WM object
# load model
# call prcurve function

# model_name = 'Tussock_v3_neg_train_test'
# model_folder = 'Tussock_v3_neg_train_test'
# dataset_name = 'Tussock_v3_neg_train_test'


# for horehound
# model_names = ['2021-03-26_MFS_Horehound_v0_2021-09-15_21_12']
# model_descriptions = ['Hh_MaskRCNN']
# model_types = ['poly']
# model_epochs = [25]
# dataset_names = ['2021-03-26_MFS_Horehound_v0']

model_names = ['2021-03-25_MFS_Tussock_v0_2021-09-16_08_55']
model_descriptions = ['St_MaskRCNN']
model_types = ['poly']
model_epochs = [25]
dataset_names = ['2021-03-25_MFS_Tussock_v0']



models={'name': model_names,
        'folder': model_names,
        'description': model_descriptions,
        'type': model_types,
        'epoch': model_epochs}
datasets = [os.path.join('dataset_objects', d, d + '.pkl') for d in dataset_names]

Horehound = WM(model_name=model_names[0],
               model_folder=model_names[0])
# import code
# code.interact(local=dict(globals(), **locals()))
Horehound.compare_models(models,
                         datasets,
                         load_prcurve=False,
                         show_fig=True)


import code
code.interact(local=dict(globals(), **locals()))