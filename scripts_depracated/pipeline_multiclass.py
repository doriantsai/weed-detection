#! /usr/bin/env python

""" agkelpie multiclass weed pipeline,
    - create image folders from dataserver using symbolic links
    - create balanced image folders + annotation files
    - create masks
    - split image data
    - create dataset objects
    - call train model
    - generate pr curve

    - NOTE: so far, just can it work on multiclass data
    - TODO: balance multiple classes during preprocessing stage (as per LDROP)
    - TODO: display of results for multiple classes
"""

import os
from weed_detection.WeedModel import WeedModel
from weed_detection.PreProcessingToolbox import PreProcessingToolbox

# setup file/folder locations
dataserver_dir = os.path.join('/home/agkelpie/Data/03_Tagged')
dataset_name = '2021-03-26_MFS_Multiclass_v1'

# glob string patterns to find the images and metadata (annotations) files, respectively
img_dir_patterns=['/2021-03-26/*/images/']
ann_dir_patterns=['/2021-03-26/*/metadata/']

ppt = PreProcessingToolbox()


# ======================================================================================
# create symbolic links to image folders
# ======================================================================================
print('Generating symbolic links')
ann_dataset_path, root_dir = ppt.generate_symbolic_links(dataserver_dir,
                                                        dataset_name,
                                                        img_dir_patterns,
                                                        ann_dir_patterns)


# ======================================================================================
# create balanced image folder and annotation file from symbolic links
# ======================================================================================
print('Generating balanced image folder and annotation file')
img_bal_dir, ann_bal_path = ppt.generate_dataset_from_symbolic_links(root_dir,
                                                             ann_dataset_path)


# ======================================================================================
# create masks
# ======================================================================================
print('Creating masks')
model_type = 'poly'
mask_dir = os.path.join(root_dir, 'masks')
res, _ = ppt.create_masks_from_poly(img_bal_dir,
                                           ann_bal_path,
                                           mask_dir_out=mask_dir)


# ======================================================================================
# split image data
# ======================================================================================
# split into train/test/val folders w respective json files
print('Splitting image data into train/test/val')
ann_train_file = ann_bal_path[:-5] + '_train.json'
ann_test_file = ann_bal_path[:-5] + '_test.json'
ann_val_file = ann_bal_path[:-5] + '_val.json'

img_dirs, ann_files, mask_dirs = ppt.split_image_data(root_dir,
                                                    img_bal_dir,
                                                    ann_bal_path,
                                                    ann_bal_path,
                                                    ann_train_file,
                                                    ann_val_file,
                                                    ann_test_file,
                                                    annotation_type=model_type,
                                                    mask_folder=mask_dir,
                                                    ann_dir=False)


# ======================================================================================
# create dataset objects (pkl)
# ======================================================================================
# setting hyper parameters
# create datasets/dataloaders for training

# TODO use library argparse for setting parameters, or json reader to store parameters
# set hyper parameters of dataset
batch_size = 10
num_workers = 10
learning_rate = 0.005 # 0.002
momentum = 0.9 # 0.8
weight_decay = 0.0001
num_epochs = 10
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

# create dataset
print('Creating dataset objects')
MulticlassModel = WeedModel(model_name=dataset_name)
dataset_path = MulticlassModel.create_train_test_val_datasets(img_dirs,
                                                        ann_files,
                                                        hp,
                                                        dataset_name,
                                                        annotation_type=model_type,
                                                        mask_folders=mask_dirs)
# load dataset
dso = MulticlassModel.load_dataset_objects(dataset_path)


# ======================================================================================
# train model
# ======================================================================================
print('Training model')
model, model_save_path = MulticlassModel.train(model_name=dataset_name,
                                         dataset_path=dataset_path,
                                         model_name_suffix=True,
                                         num_classes=3) # num_classes = 2 + background = 3
print(f'finished training model: {model_save_path}')


# ======================================================================================
# demonstrate model on test set (infer)
# ======================================================================================
print('Model Demonstration on Test set')
pred = MulticlassModel.infer_dataset(dataset=dso['ds_test'],
                                     imsave=True)
print('Model inference complete')

# ======================================================================================
# generate pr curve
# ======================================================================================
# print('Computing PR curve')
# model_names = [MulticlassModel.get_model_name()]
# model_descriptions = ['Multiclass_MaskRCNN']
# model_types = [model_type]
# model_epochs = [5]  # TODO find min. of validation curve and use nearest
# models={'name': model_names,
#         'folder': model_names,
#         'description': model_descriptions,
#         'type': model_types,
#         'epoch': model_epochs}

# dataset_names = [dataset_name]
# datasets = [os.path.join('dataset_objects', d, d + '.pkl') for d in dataset_names]

# import code
# code.interact(local=dict(globals(), **locals()))

# TODO: need to update PR curve generation for multiclass object detction
# res = MulticlassModel.compare_models(models,
#                          datasets,
#                          load_prcurve=False,
#                          show_fig=True)

# python debug code
import code
code.interact(local=dict(globals(), **locals()))