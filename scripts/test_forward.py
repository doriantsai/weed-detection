#! /usr/bin/env python

""" test forward method of maskrcnn pipeline """

import os
from weed_detection.PreProcessingToolbox import PreProcessingToolbox
from weed_detection.WeedDatasetPoly import WeedDatasetPoly as WDP
from weed_detection.WeedModel import WeedModel as WM
import torch

# create WM object
Tussock = WM()

dataset_name = 'Tussock_v0_mini'
# dataset_file = os.path.join('dataset_objects', dataset_name, dataset_name, + '.pkl')
root_dir = os.path.join('/home',
                        'dorian',
                        'Data',
                        'AOS_TussockDataset',
                        dataset_name)

img_folder = os.path.join(root_dir, 'Images')
img_folders = [os.path.join(img_folder, 'Train'),
               os.path.join(img_folder, 'Test'),
               os.path.join(img_folder, 'Validation')]

mask_dir = os.path.join(root_dir, 'Masks')
mask_folders = [os.path.join(mask_dir, 'Train'),
               os.path.join(mask_dir, 'Test'),
               os.path.join(mask_dir, 'Validation')]
all_mask_dir = os.path.join(mask_dir, 'All')


ann_files = [os.path.join(root_dir, 'Annotations', 'annotations_tussock_21032526_G507_train.json'),
            os.path.join(root_dir, 'Annotations', 'annotations_tussock_21032526_G507_test.json'),
            os.path.join(root_dir, 'Annotations', 'annotations_tussock_21032526_G507_val.json')]


# TODO split image data
# run split_image_data.py
# folder containing all images to be used for testing/training/validation
all_folder = os.path.join(root_dir, 'Images', 'All')
ann_master_file = 'via_project_29Apr2021_17h43m_json_bbox_poly_pt.json'
ann_all_file = 'via_project_29Apr2021_17h43m_json_bbox_poly_pt.json'

SPLIT = False
if SPLIT:
    ProTool = PreProcessingToolbox()
    img_folders, ann_files = ProTool.split_image_data(root_dir,
                                                    all_folder,
                                                    ann_master_file,
                                                    ann_all_file,
                                                    ann_files[0],
                                                    ann_files[2],
                                                    ann_files[1],
                                                    mask_folder=all_mask_dir,
                                                    annotation_type='poly')

# TODO create dataset objects
#
# set hyper parameters of dataset
batch_size = 10
num_workers = 10
learning_rate = 0.005
momentum = 0.9
weight_decay = 0.0005
num_epochs = 10
step_size = 3 # round(num_epochs / 2)
shuffle = True
rescale_size = int(256)

# make a hyperparameter dictionary
hp={}
hp['batch_size'] = batch_size
hp['num_workers'] = num_workers
hp['learning_rate'] = learning_rate
hp['momentum'] = momentum
hp['step_size'] = step_size
hp['weight_decay'] = weight_decay
hp['num_epochs'] = num_epochs
hp['shuffle'] = shuffle
hp['rescale_size'] = rescale_size

# init object
Tussock = WM()
# save all datasets/dataloaders in a .pkl file
dataset_path = Tussock.create_train_test_val_datasets(img_folders,
                                                      ann_files,
                                                      hp,
                                                      dataset_name,
                                                      annotation_type='poly',
                                                      mask_folders=mask_folders)


# test forward pass
dso = Tussock.load_dataset_objects(dataset_path)
dataset = dso['ds_train']
dataloader = dso['dl_train']

# for training
images, targets = next(iter(dataloader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]

# import code
# code.interact(local=dict(globals(), **locals()))

model = Tussock.build_maskrcnn_model(num_classes=2)

output = model(images, targets)
print(output)

# for inference
model.eval()  # I don't think this works, but maybe
x = [torch.rand(3, 256, 256), torch.rand(3, 256, 256)]
predictions = model(x)
print(predictions)

# call WM.train
# Tussock.train(model_name=dataset_name,
#               dataset_path=dataset_path)

i = 0
img = images[i]
targ = targets[i]
mask = targ['masks']

print(img.shape)
print(mask.shape)

import numpy as np
import matplotlib.pyplot as plt
img = img.cpu().numpy()
mask = mask.cpu().numpy()

img = np.transpose(img, (1,2,0))
mask = np.transpose(mask, (1,2,0))


fig, ax = plt.subplots(2,1)
ax[0].imshow(img)
ax[1].imshow(mask)
plt.show()

import code
code.interact(local=dict(globals(), **locals()))