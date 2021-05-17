#! /usr/bin/env python

""" script to infer from model for video stream """

import os
import cv2 as cv
from weed_detection.WeedModel import WeedModel as WM

# call inference functions

# init WM object
Tussock = WM()

# load model
model_name = 'tussock_test_2021-05-16_16_13'
save_model_path = os.path.join('output',
                               model_name,
                               model_name + '.pth')
Tussock.load_model(save_model_path)
Tussock.set_model_name(model_name)
Tussock.set_model_path(save_model_path)

# run model inference on entire dataset
# pred = Tussock.infer_dataset(ds_test, imsave=True)

# create video capture object
vid_cap = cv.VideoCapture(0)
vid_out_path = Tussock.infer_video(vid_cap)



import code
code.interact(local=dict(globals(), **locals()))