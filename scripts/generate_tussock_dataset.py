#! /usr/bin/env python

"""
script to generate tussock dataset from weeds reference library setup,
grab all image files
grab all json files
create one large merged json file for entire dataset
create specified dataset folder and populate with simlink of images
for all relevant images based on specific days/folders
"""

import os
import json
import glob
from weed_detection.PreProcessingToolbox import PreProcessingToolbox
import random
import numpy as np
import shutil

# dataset location/root_dir
dataset_name = '2021-03-25_MFS_Tussock'
root_dir = os.path.join('/home/dorian/Data/agkelpie', dataset_name)

# annotation file
ann_dir = os.path.join(root_dir, 'metadata')
ann_file = '2021-03-25_MFS_Tussock_ed20210909.json'
ann_path = os.path.join(ann_dir, ann_file)

ppt = PreProcessingToolbox()
img_out_dir, ann_out_file = ppt.generate_dataset_from_symbolic_links(root_dir, ann_path)


import code
code.interact(local=dict(globals(), **locals()))

