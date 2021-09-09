#! /usr/bin/env python

"""
script to generate masks from images with polygon annotations
"""

import os
import json
from weed_detection.PreProcessingToolbox import PreProcessingToolbox
import matplotlib.pyplot as plt

# folder locations + database name
dataset_name = '2021-03-25_MFS_Tussock'
root_dir = os.path.join('/home', 'dorian', 'Data', 'AOS_TussockDataset',
                            dataset_name)

img_dir_in = os.path.join(root_dir, 'images')
# ann_file_name = 'via_project_07Jul2021_08h00m_240_test_allpoly.json'
# ann_file_name = 'via_project_07Jul2021_08h00m_240_polysubset_bootstrap.json'
ann_file_name = '2021-03-25_MFS_Tussock_ed20210909.json'
ann_file_path = os.path.join(root_dir, 'metadata', ann_file_name)
img_dir_out = os.path.join(root_dir, 'masks')


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

# test/show a mask
i = 400
mask = plt.imread(os.path.join(img_dir_out, mask_list[i]))
plt.imshow(mask)
plt.show()

import code
code.interact(local=dict(globals(), **locals()))
