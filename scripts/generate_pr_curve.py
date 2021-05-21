#! /usr/bin/env python

""" script to generate a single pr curve """

import os
import pickle
import numpy as np
from weed_detection.WeedModel import WeedModel as WM

# init WM object
# load model
# call prcurve function


# init WM object
Tussock = WM()

# load dataset objects
# dataset_name = 'Tussock_v1'
# dataset_name = 'Tussock_v2'
dataset_name = 'Tussock_v3_neg_test'
dataset_file = os.path.join('dataset_objects', dataset_name, dataset_name + '.pkl')
# load dataset files via unpacking the pkl file
dso = Tussock.load_dataset_objects(dataset_file)

# import code
# code.interact(local=dict(globals(), **locals()))

# load model
# model_name = 'tussock_test_2021-05-16_16_13'
model_name = 'Tussock_v3_neg_train_test_2021-05-20_13_58'
save_model_path = os.path.join('output',
                               model_name,
                               model_name + '.pth')
Tussock.load_model(save_model_path)
Tussock.set_model_name(model_name)
Tussock.set_model_path(save_model_path)

conf_thresh = np.linspace(0.99, 0.01, num=101, endpoint=True)
# TODO for 0.0 and 1.0 confidence threshold, produces nans because no tp

iou_thresh = 0.5
save_prcurve_folder = os.path.join('output', model_name, 'prcurve')
res = Tussock.get_prcurve(dso['ds_test'],
                            confidence_thresh=conf_thresh,
                            nms_iou_thresh=iou_thresh,
                            decision_iou_thresh=iou_thresh,
                            save_folder=save_prcurve_folder,
                            imsave=True)

#  res = {'precision': p_final,
#            'recall': r_final,
#            'ap': ap,
#            'f1score': f1score,
#            'confidence': c_final}

import code
code.interact(local=dict(globals(), **locals()))