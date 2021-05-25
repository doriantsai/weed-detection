#! usr/bin/env python

""" augment training data with image transforms """

# to compensate for low data, we can augment our training data with original images transformed
# from the classes defined in WeedDataset.py

# TODO

# make a new copy of the dataset images (eg Tussock_v2)
# split image data randomly

# set image directories
# set annotation files
# init WeedDataset transforms
# for each transform
    # apply to image
    # append to image_name the transform/degree of effect
    # save image into Image/Train folder
    # add info to annotations file

#
# do model comparison with two datasets

import os
from weed_detection.WeedModel import WeedModel as WM
from weed_detection.PreProcessingToolbox import PreProcessingToolbox as PT

# load dataset objects
dataset_name = 'Tussock_v3_augment'
dataset_file = os.path.join('dataset_objects', dataset_name, dataset_name + '.pkl')

# load dataset files via unpacking the pkl file
WeedModel = WM()
print('loading dataset: {}'.format(dataset_file))
dso = WeedModel.load_dataset_objects(dataset_file)

# HACK temporarily do ds_val since it is the smallest
dataset = dso['ds_val']


# folder locations and file names
root_dir = os.path.join('/home',
                        'dorian',
                        'Data',
                        'AOS_TussockDataset',
                        dataset_name)
img_dir = os.path.join(root_dir, 'Images', 'Validation')
ann_dir = 'Annotations'
ann_in = os.path.join('annotations_tussock_21032526_G507_val.json')

# 0: vert flip
# 1: horz flip
# 2: blur
# 3: bright
# 4: contrast
# 5: hue
# 6: saturation
tform_select = 3

ann_out = os.path.join('annotations_val_transform_combined.json')


ProTool = PT()
ProTool.augment_training_data(root_dir,
                              img_dir,
                              ann_in,
                              tform_select,
                              ann_out=ann_out)

# TODO confirm that augmented data works/json files are correct
# test dataset - should be orderd
# i = 0
# for image, sample in dataset:
#     image_name = sample['image_id']
#     print('{}: image_id: {}'.format(i, image_name))

#     # if i > 1:
#     #     break
#     i += 1
print('done augmenting training data')

import code
code.interact(local=dict(globals(), **locals()))