#! /usr/bin/env python

""" test forward method of maskrcnn pipeline """

import os
from weed_detection.PreProcessingToolbox import PreProcessingToolbox
from weed_detection.WeedDatasetPoly import WeedDatasetPoly as WDP
from weed_detection.WeedModel import WeedModel as WM

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

ann_files = [os.path.join(root_dir, 'Annotations', 'annotations_tussock_21032526_G507_train.json'),
            os.path.join(root_dir, 'Annotations', 'annotations_tussock_21032526_G507_test.json'),
            os.path.join(root_dir, 'Annotations', 'annotations_tussock_21032526_G507_val.json')]


# TODO split image data
# run split_image_data.py
# folder containing all images to be used for testing/training/validation
all_folder = os.path.join(root_dir, 'Images', 'All')
ann_master_file = 'via_project_29Apr2021_17h43m_json_bbox_poly_pt.json'
ann_all_file = 'via_project_29Apr2021_17h43m_json_bbox_poly_pt.json'

ProTool = PreProcessingToolbox()
img_folders, ann_files = ProTool.split_image_data(root_dir,
                                                  all_folder,
                                                  ann_master_file,
                                                  ann_all_file,
                                                  ann_files[0],
                                                  ann_files[2],
                                                  ann_files[1],
                                                  annotation_type='poly')

# TODO create dataset objects
# 
# set hyper parameters of dataset
batch_size = 10
num_workers = 10
learning_rate = 0.005
momentum = 0.9
weight_decay = 0.0001
num_epochs = 75
step_size = round(num_epochs / 2)
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
                                                      annotation_type='poly')


# test forward pass
dso = Tussock.load_dataset_objects(dataset_path)
dataset = dso['ds_train']
dataloader = dso['dl_train']

# for training
images, targets = next(iter(dataloader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]
output = Tussock.model(images, targets)
print(output)

# for inference
Tussock.model.eval()  # I don't think this works, but maybe
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = Tussock.model(x)
print(predictions)

# call WM.train
# Tussock.train(model_name=dataset_name,
#               dataset_path=dataset_path)



import code
code.interact(local=dict(globals(), **locals()))