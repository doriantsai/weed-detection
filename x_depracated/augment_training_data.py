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
import shutil
from weed_detection.WeedModel import WeedModel as WM
from weed_detection.PreProcessingToolbox import PreProcessingToolbox as PT

# load dataset objects
dataset_name = 'Tussock_v3_augment'
# dataset_file = os.path.join('dataset_objects', dataset_name, dataset_name + '.pkl')

# load dataset files via unpacking the pkl file
# WeedModel = WM()
# print('loading dataset: {}'.format(dataset_file))
# dso = WeedModel.load_dataset_objects(dataset_file)

# dataset = dso['ds_test']

ProTool = PT()




# folder locations and file names
root_dir = os.path.join('/home',
                        'dorian',
                        'Data',
                        'agkelpie',
                        dataset_name)
img_dir = os.path.join(root_dir, 'Images', 'Train')
ann_dir = 'Annotations'
ann_in = os.path.join('annotations_tussock_21032526_G507_train.json')

tform_vector = [0, 1, 2, 3, 4, 6]
for i in range(len(tform_vector)):
    # 0: vert flip
    # 1: horz flip
    # 2: blur
    # 3: bright
    # 4: contrast
    # 5: hue
    # 6: saturation
    tform_select = tform_vector[i]

    ann_out = os.path.join('annotations_train_augmented.json')


    if i == 0:
        ann_append = False
        rm_folder = True
    else:
        ann_append = True
        rm_folder = False
    ProTool.augment_training_data(root_dir,
                                img_dir,
                                ann_in,
                                tform_select,
                                ann_out=ann_out,
                                ann_append=ann_append,
                                rm_folder=rm_folder)


# at the end, copy all augmented images into training folder
# then commence training?

# lastly, combine augmented training data json with original image training data json
print('combining augmented json with original training json')
ProTool.combine_annotations([ann_in, ann_out],
                             ann_dir=os.path.join(root_dir, ann_dir),
                             ann_out='annotations_train_augmented_combined.json')

# copy all Augmented Images into Train folder
print('copying augmented image files to training folder')
save_folder = os.path.join(root_dir, 'Images', 'Augmented')
files = os.listdir(save_folder)
for f in files:
    shutil.copyfile(os.path.join(save_folder, f),
                    os.path.join(img_dir, f))

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