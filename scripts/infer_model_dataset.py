#! /usr/bin/env python

""" script to infer from model for entire dataset """

# overview of script:
# init WM object
# load model
# call inference functions

import os
from weed_detection.WeedModel import WeedModel as WM
from weed_detection.WeedDatasetPoly import Compose, Rescale, ToTensor

# init WM object
WeedModel = WM()

# load dataset objects
dataset_name = '2021-03-26_MFS_Multiclass_v1'

DATASET_FILE_EXISTS = True
if DATASET_FILE_EXISTS:
    dataset_file = os.path.join('dataset_objects',
                                dataset_name,
                                dataset_name + '.pkl')
    # load dataset files via unpacking the pkl file
    dso = WeedModel.load_dataset_objects(dataset_file)

else:
    root_dir = os.path.join('/home',
                            'agkelpie',
                            'Data',
                            dataset_name)
    img_dir = os.path.join(root_dir, 'images_test')
    mask_dir = os.path.join(root_dir, 'masks_test')
    ann_file = '2021-03-25_MFS_Multiclass_v0_balanced_test.json'
    ann_path = os.path.join(root_dir, 'metadata', ann_file)

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
model_name = '2021-03-26_MFS_Multiclass_v1_2022-08-17_16_19'

# model_name = dataset_name
save_model_path = os.path.join('output',
                               model_name,
                               model_name + '.pth')
WeedModel.load_model(save_model_path, annotation_type='poly', num_classes=3)
WeedModel.set_model_name(model_name)
WeedModel.set_model_path(save_model_path)
WeedModel.set_model_folder(model_name)
# WeedModel.set_snapshot(5)

# run model inference on entire dataset
print('infering dataset')
pred = WeedModel.infer_dataset(dso['ds_test'],
                             imsave=True,
                             save_subfolder='infer_dataset_test',
                             conf_thresh=0.5,
                             annotation_type='poly')
    # image_out, pred = Tussock.infer_image(image,
    #                                       sample=sample,
    #                                       imshow=True,
    #                                       imsave=True)
    # print('{}: {}'.format(i, image_id))
    # print('   pred = {}'.format(pred))

import code
code.interact(local=dict(globals(), **locals()))