#! /usr/bin/env python

"""
script to go into weeds reference library setup,
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

ppt = PreProcessingToolbox()
res = ppt.generate_symbolic_links(data_server_dir='/home/dorian/Data/agkelpie/03_Tagged',
                                      dataset_name='2021-03-25_MFS_Tussock',
                                      img_dir_patterns=['/2021-03-25/*/images/', '/2021-03-26/Location_1/images/'],
                                      ann_dir_patterns=['/2021-03-25/*/metadata/', '/2021-03-26/Location_1/metadata/'])


import code
code.interact(local=dict(globals(), **locals()))
