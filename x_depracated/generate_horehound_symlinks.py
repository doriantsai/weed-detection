#! /usr/bin/env python

"""
script to go into weeds reference library setup,
grab all image files
grab all json files
create one large merged json file for entire dataset
create specified dataset folder and populate with simlink of images
for all relevant images based on specific days/folders
generate horeound symlinks from S3 tbucket structure
"""

from weed_detection.PreProcessingToolbox import PreProcessingToolbox



ppt = PreProcessingToolbox()
res = ppt.generate_symbolic_links(data_server_dir='/home/dorian/Data/agkelpie/03_Tagged',
                                    dataset_name='2021-03-26_MFS_Horehound',
                                    img_dir_patterns=['/2021-03-26/Location_2/images/'],
                                    ann_dir_patterns=['/2021-03-26/Location_2/metadata/'])

import code
code.interact(local=dict(globals(), **locals()))
