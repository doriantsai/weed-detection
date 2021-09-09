#! /usr/bin/env python

""" script to infer from model for entire dataset """

import os
import pickle
from weed_detection.WeedModel import WeedModel as WM
from weed_detection.WeedDatasetPoly import Compose, Rescale, ToTensor

# init WM object
# load model
# call inference functions

# init WM object
Tussock = WM()

# load dataset objects
# dataset_name = 'Tussock_v0_mini'
# dataset_name = 'Tussock_v3_neg_train_test'
dataset_name = '2021-03-25_MFS_Tussock_MaskRCNN'

DATASET_FILE_EXISTS = True
if DATASET_FILE_EXISTS:
    dataset_file = os.path.join('dataset_objects',
                                dataset_name,
                                dataset_name + '.pkl')
    # load dataset files via unpacking the pkl file
    dso = Tussock.load_dataset_objects(dataset_file)

    # just choose which dataset object to use for this script
    ds_infer = dso['ds_train']
    dl_infer = dso['dl_train']

else:
    root_dir = os.path.join('/home',
                            'dorian',
                            'Data',
                            'AOS_TussockDataset',
                            dataset_name)
    img_dir = os.path.join(root_dir, 'Images', 'PolySubset')
    mask_dir = os.path.join(root_dir, 'Masks', 'PolySubset')
    ann_file = 'via_project_07Jul2021_08h00m_240_polysubset_bootstrap.json'
    ann_path = os.path.join(root_dir, 'Annotations', ann_file)

    # set hyper parameters of dataset
    batch_size = 10
    num_workers = 10
    learning_rate = 0.005 # 0.002
    momentum = 0.9 # 0.8
    weight_decay = 0.0001
    num_epochs = 50
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

    tform = Compose([Rescale(rescale_size),
                     ToTensor()])
    dataset, dataloader = Tussock.create_dataset_dataloader(root_dir=root_dir,
                                      json_file=ann_file,
                                      transforms=tform,
                                      hp=hp,
                                      annotation_type='poly',
                                      mask_dir=mask_dir,
                                      img_dir=img_dir)


    ds_infer = dataset

# load model
# model_name = 'tussock_test_2021-05-16_16_13'
# model_name = 'Tussock_v0_mini_2021-06-09_09_19'
# model_name = 'Tussock_v4_poly286_2021-07-15_11_08'
# model_name = '2021-03-25_MFS_Tussock_MaskRCNN_2021-08-31_19_33'
model_name = '2021-03-25_MFS_Tussock_FasterRCNN_2021-09-01_16_49'

# model_name = dataset_name
save_model_path = os.path.join('output',
                               model_name,
                               model_name + '.pth')
Tussock.load_model(save_model_path, annotation_type='box')
Tussock.set_model_name(model_name)
Tussock.set_model_path(save_model_path)
Tussock.set_model_folder(model_name)
Tussock.set_snapshot(20)

# run model inference on entire dataset
print('infering dataset')
pred = Tussock.infer_dataset(dso['ds_test'],
                             imsave=True,
                             save_subfolder='infer_dataset_test',
                             conf_thresh=0.42,
                             annotation_type='box')
    # image_out, pred = Tussock.infer_image(image,
    #                                       sample=sample,
    #                                       imshow=True,
    #                                       imsave=True)
    # print('{}: {}'.format(i, image_id))
    # print('   pred = {}'.format(pred))

import code
code.interact(local=dict(globals(), **locals()))