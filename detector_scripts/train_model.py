#! /usr/bin/env python

import os
import time
import pickle

from weed_detection.WeedDataset import WeedDataset as WD
from weed_detection.WeedModel import WeedModel as WM

# folder locations of dataset files
dataset_file = os.path.join('output', 'dataset', 'blah.pkl')

# load dataset files via unpacking the pkl file

# create WM object
# call WM.train
