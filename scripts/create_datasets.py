#! /usr/bin/env python

""" script to create dataset/dataloader objects using output from
PPT.split_image_data() """

import os
import weed_detection.WeedModel as WeedModel

# setup folder locations init object call object

# NOTE: copied/pasted output from PPT.split_iamge_data()
# dataset_name = 'Tussock_v3_augment'
# dataset_name = 'Tussock_v0_mini'
# dataset_name = 'Tussock_v3_neg_train_test'
dataset_name = '2021-03-25_MFS_Tussock'

root_dir = os.path.join('/home',
                        'dorian',
                        'Data',
                        'AOS_TussockDataset',
                        dataset_name)
train_folder = os.path.join(root_dir, 'images_train')
test_folder = os.path.join(root_dir, 'images_test')
val_folder = os.path.join(root_dir, 'images_validation')
img_folders = [train_folder, test_folder, val_folder]


# default mask folders
mask_folders = [os.path.join(root_dir, 'masks_train'),
               os.path.join(root_dir, 'masks_test'),
               os.path.join(root_dir, 'masks_validation')]
# all_mask_dir = os.path.join(mask_dir, 'All')
all_mask_dir = os.path.join(root_dir, 'masks')

ann_dir = os.path.join(root_dir, 'metadata')
ann_file = '2021-03-25_MFS_Tussock_ed20210909.json'
ann_path = os.path.join(ann_dir, ann_file)

ann_master_file = '2021-03-25_MFS_Tussock_ed20210909.json'  # we are using master file as allpoly, because it contains all images

# annotation files out
ann_train_file = ann_file[:-5] + '_train.json'
ann_test_file = ann_file[:-5] + '_test.json'
ann_val_file = ann_file[:-5] + '_val.json'

annotations_train = os.path.join(ann_dir, ann_train_file)
annotations_val = os.path.join(ann_dir, ann_val_file)
annotations_test = os.path.join(ann_dir, ann_test_file)
ann_files = [annotations_train, annotations_test, annotations_val]

# ann_files = [os.path.join(root_dir, 'Annotations', 'annotations_tussock_21032526_G507_train.json'),
#             os.path.join(root_dir, 'Annotations', 'annotations_tussock_21032526_G507_test_shortgrass.json'),
#             os.path.join(root_dir, 'Annotations', 'annotations_tussock_21032526_G507_val.json')]

# set hyper parameters of dataset
batch_size = 10
num_workers = 10
learning_rate = 0.005 # 0.002
momentum = 0.9 # 0.8
weight_decay = 0.0001
num_epochs = 100
step_size = round(num_epochs / 2)
shuffle = True
rescale_size = int(1024)

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

hp_train = hp
hp_test = hp
hp_test['shuffle'] = False

# dataset_name = 'Tussock_v2'

# init object
Tussock = WeedModel()
# save all datasets/dataloaders in a .pkl file
# dataset_name_save = dataset_name + '_shortgrass'
dataset_name_save = dataset_name
dataset_path = Tussock.create_train_test_val_datasets(img_folders,
                                                      ann_files,
                                                      hp,
                                                      dataset_name_save,
                                                      annotation_type='poly',
                                                      mask_folders=mask_folders)

# TODO open pkl file and confirm that datasets match image length, but some are
# "augmented"?
print('dataset_path = {}'.format(dataset_path))

# test forward pass
dso = Tussock.load_dataset_objects(dataset_path)
dataset = dso['ds_train']
dataloader = dso['dl_train']


import code
code.interact(local=dict(globals(), **locals()))