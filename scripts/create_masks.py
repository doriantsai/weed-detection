#! /usr/bin/env python

"""
script to generate masks from images with polygon annotations
"""

import os
import json
from weed_detection.PreProcessingToolbox import PreProcessingToolbox


# folder locations + database name
db_name = 'Tussock_v4_poly'
root_dir = os.path.join('/home', 'dorian', 'Data', 'AOS_TussockDataset',
                            db_name)

img_dir_in = os.path.join(root_dir, 'Images', 'All')
# ann_file_name = 'via_project_07Jul2021_08h00m_240_test_allpoly.json'
# ann_file_name = 'via_project_07Jul2021_08h00m_240_polysubset_bootstrap.json'
ann_file_name = '20210819-MFS-01-bootprogress-570-occlusion24_json.json'
ann_file_path = os.path.join(root_dir, 'Annotations', ann_file_name)
img_dir_out = os.path.join(root_dir, 'Masks', 'All')


# init object
ppt = PreProcessingToolbox()
# call object
ppt.create_masks_from_poly(img_dir_in, ann_file_path, img_dir_out)

# check how many images there are
img_list = os.listdir(img_dir_in)

# check how many masks there are:
mask_list = os.listdir(img_dir_out)

print(f'number of images: {len(img_list)}')
print(f'number of masks: {len(mask_list)}')

print('done generating masks from polygons')

