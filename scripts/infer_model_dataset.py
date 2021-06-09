#! /usr/bin/env python

""" script to infer from model for entire dataset """

import os
import pickle
from weed_detection.WeedModel import WeedModel as WM

# init WM object
# load model
# call inference functions

# init WM object
Tussock = WM()

# load dataset objects
dataset_name = 'Tussock_v0_mini'
# dataset_name = 'Tussock_v3_neg_train_test'
dataset_file = os.path.join('dataset_objects',
                            dataset_name,
                            dataset_name + '.pkl')
# load dataset files via unpacking the pkl file
dso = Tussock.load_dataset_objects(dataset_file)


# load model
# model_name = 'tussock_test_2021-05-16_16_13'
model_name = 'Tussock_v0_mini_2021-06-09_09_19'
# model_name = dataset_name
save_model_path = os.path.join('output',
                               model_name,
                               model_name + '.pth')
Tussock.load_model(save_model_path)
Tussock.set_model_name(model_name)
Tussock.set_model_path(save_model_path)
Tussock.set_model_folder(model_name)

# run model inference on entire dataset
pred = Tussock.infer_dataset(dso['ds_test'], imsave=True)
    # image_out, pred = Tussock.infer_image(image,
    #                                       sample=sample,
    #                                       imshow=True,
    #                                       imsave=True)
    # print('{}: {}'.format(i, image_id))
    # print('   pred = {}'.format(pred))

import code
code.interact(local=dict(globals(), **locals()))