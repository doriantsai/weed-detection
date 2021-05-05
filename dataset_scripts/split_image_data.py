#! /usr/bin/env python

"""
main dataset prep script
randomly dump images from All into Train/Val/Test, given a split
sync annotation files with respective images
"""

# specify image folder for All images
# specify image folders for Train/Val/Test
# specify split ratio given numbers
# do the random split

import os
import pickle
import torch
import WeedDataset as WD
import shutil
from sync_annotations_with_imagefolder import sync_annotations_with_folder


def copy_images_to_folder(dataset, all_folder, save_folder):
    for image, sample in dataset:
        image_id = sample['image_id'].item()
        img_name = dataset.dataset.annotations[image_id]['filename']
        new_img_path = os.path.join(save_folder, img_name)
        old_img_path = os.path.join(all_folder, img_name)
        # print('copy from: {}'.format(old_img_path))
        # print('       to: {}'.format(new_img_path))
        shutil.copyfile(old_img_path, new_img_path)


# set directories
root_dir = os.path.join('/home', 'dorian', 'Data', 'AOS_TussockDataset', 'Tussock_v1')
all_folder = os.path.join(root_dir, 'Images', 'All')
train_folder = os.path.join(root_dir, 'Images', 'Train')
val_folder = os.path.join(root_dir, 'Images', 'Validation')
test_folder = os.path.join(root_dir, 'Images', 'Test')

# set annotation file
annotation_dir = os.path.join(root_dir, 'Annotations')
annotation_master = os.path.join(annotation_dir, 'annotations_tussock_21032526_G507_combined_all.json')

# ensure that annotations file matches all images in all_folder (a requirement when doing random_split)
annotation_all = os.path.join(annotation_dir, 'annotations_tussock_21032526_G507_all.json')
sync_annotations_with_folder(all_folder, annotation_master, annotation_all)

# create dummy WD dataset and do random split on it
# TODO add rest of data augmentations directly to train folder
# tform_train = WD.Compose([WD.Rescale(rescale_size),
#                           WD.RandomBlur(5, (0.5, 2.0)),
#                           WD.ToTensor()])
wd = WD.WeedDataset(all_folder, annotation_all, transforms=None)
# wd_train = WD.WeedDataset(all_folder, annotation_all, transforms=tform_train)

# dataset lengths
# nimg = len(wd) - number of images in the all_folder
files = os.listdir(all_folder)
img_files = []
for f in files:
    if f.endswith('.png'):
        img_files.append(f)
nimg = len(img_files)
print('number of images in all_folder: {}'.format(nimg))

# define ratio:
# 70/20/10
# train/test/val
ratio = [0.7, 0.2]
ratio.append(1 - ratio[0] - ratio[1])

tr = int(round(nimg * ratio[0]))
te = int(round(nimg * ratio[1]))
va = int(round(nimg * ratio[2]))

print('ntrain {}'.format(tr))
print('ntest {}'.format(te))
print('nval {}'.format(va))

# do random split of image data
ds_train, ds_val, ds_test = torch.utils.data.random_split(wd, [tr, va, te])
# only apply tform_train to training dataset
ds_train.dataset = wd_train

# now, actually copy images from All folder to respective image folders
# dataset = ds_train
# save_folder = train_folder
copy_images_to_folder(ds_train, all_folder, train_folder)
copy_images_to_folder(ds_val, all_folder, val_folder)
copy_images_to_folder(ds_test, all_folder, test_folder)
print('copy images from all_folder to train/test/val_folder complete')

# now, call sync_annotations_with_imagefolder for each:
annotations_train = os.path.join(annotation_dir, 'annotations_tussock_21032526_G507_train.json')
annotations_val = os.path.join(annotation_dir, 'annotations_tussock_21032526_G507_val.json')
annotations_test = os.path.join(annotation_dir, 'annotations_tussock_21032526_G507_test.json')

# function calls for each folder
sync_annotations_with_folder(train_folder, annotation_all, annotations_train)
sync_annotations_with_folder(val_folder, annotation_all, annotations_val)
sync_annotations_with_folder(test_folder, annotation_all, annotations_test)
print('sync json with image folders complete')

import code
code.interact(local=dict(globals(), **locals()))
