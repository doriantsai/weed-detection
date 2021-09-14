#! /usr/bin/env python

""" script to generate a single pr curve """

import os
import pickle
import numpy as np
from weed_detection.WeedModel import WeedModel as WM

# init WM object
# load model
# call prcurve function

# model_name = 'Tussock_v3_neg_train_test'
# model_folder = 'Tussock_v3_neg_train_test'
# dataset_name = 'Tussock_v3_neg_train_test'


model_names = ['2021-03-26_MFS_Horehound_FasterRCNN_2021-09-09_20_17']
model_descriptions = ['Hh_FasterRCNN']
model_types = ['box']
model_epochs = [25]
models={'name': model_names,
        'folder': model_names,
        'description': model_descriptions,
        'type': model_types,
        'epoch': model_epochs}
dataset_names = ['2021-03-26_MFS_Horehound']
datasets = [os.path.join('dataset_objects', d, d + '.pkl') for d in dataset_names]

Horehound = WM(model_name=model_names[0],
               model_folder=model_names[0])
Horehound.compare_models(models,
                         datasets,
                         load_prcurve=False,
                         show_fig=True)
# init WM object
# Tussock = WM(model_name=model_name,
#              model_folder=model_folder)

# save_model_path = os.path.join('output',
#                                model_name,
#                                model_name + '.pth')
# Tussock.load_model(save_model_path)
# Tussock.set_model_name(model_name)
# Tussock.set_model_path(save_model_path)
# Tussock.set_snapshot(25)

# # load dataset objects
# # dataset_name = 'Tussock_v1'
# # dataset_name = 'Tussock_v2'

# dataset_file = os.path.join('dataset_objects', dataset_name, dataset_name + '.pkl')
# # load dataset files via unpacking the pkl file
# dso = Tussock.load_dataset_objects(dataset_file)


# conf_thresh = np.linspace(0.99, 0.01, num=25, endpoint=True)
# # TODO for 0.0 and 1.0 confidence threshold, produces nans because no tp

# iou_thresh = 0.5
# save_prcurve_folder = os.path.join('output', model_name, 'prcurve')
# res = Tussock.get_prcurve(dso['ds_test'],
#                             confidence_thresh=conf_thresh,
#                             nms_iou_thresh=iou_thresh,
#                             decision_iou_thresh=iou_thresh,
#                             save_folder=save_prcurve_folder,
#                             imsave=True)
# print(res)
# with imasve=True, 0.30500981178548603 hrs
# with imsave=False, ? 0.2239314360751046 hrs


#  res = {'precision': p_final,
#            'recall': r_final,
#            'ap': ap,
#            'f1score': f1score,
#            'confidence': c_final}

import code
code.interact(local=dict(globals(), **locals()))