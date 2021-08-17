#! /usr/bin/env python

"""
pr curves for classifier, between development and deployment conditions
"""

from sklearn.preprocessing import label_binarize
import numpy as np
import os
import pandas as pd
import pickle
from deepweeds_dataset import DeepWeedsDataset, Rescale, RandomCrop, ToTensor, CLASSES, CLASS_NAMES

########### classes #############
CLASS_NAMES = ('Chinee apple',
                'Lantana',
                'Parkinsonia',
                'Parthenium',
                'Prickly acacia',
                'Rubber vine',
                'Siam weed',
                'Snake weed',
                'Negative')
CLASSES = np.arange(0, len(CLASS_NAMES))
CLASS_DICT = {i: CLASS_NAMES[i] for i in range(0, len(CLASSES))}

# load model
model_file = 'dw_r50_s500_i0.pth'
model_path = os.path.join('saved_model', model_file)

# load data
data_file = 'development_labels_trim.pkl'
data_path = os.path.join('dataset_objects', data_file)

with open(data_path, 'rb') as file:
    td = pickle.load(file)
    vd = pickle.load(file)
    test_dataset = pickle.load(file)
    tdl = pickle.load(file)
    vdl = pickle.load(file)
    test_loader = pickle.load(file)


print(td)

import code
code.interact(local=dict(globals(), **locals()))