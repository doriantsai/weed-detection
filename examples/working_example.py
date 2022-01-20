#! /usr/bin/env python

"""
Minimum working example for model inference
Author: Dorian Tsai
Date: 2021 October 20
Project: AOS Weed Detection

Given model filename and image, detect target weed in image,
output bounding contour and spraypoint (center of polygon)
"""

import os
import cv2 as cv
from weed_detection.WeedModel import WeedModel
from subprocess import call

# ---------------------------------------------------------------------------- #

# check if model is in folder. If so, use it. Otherwise, download it
model_path = os.path.join('models','detection_model.pth')
if not os.path.exists(model_path):
    # download model to folder/file-path specified by model_path
    os.makedirs('models', exist_ok=True)
    url = 'https://cloudstor.aarnet.edu.au/plus/s/ZJQAKiOZFDBxDJc/download'
    call(['wget', '-O', model_path, url])

# setup weed model
Tussock = WeedModel()
Tussock.load_model(model_path)

# image name, note img_path must be precise path from working_example.py
# img_path = os.path.join('../images', 'images_train', 'mako___2021-3-25___14-51-45-481.png')
img_path = os.path.join('./images', 'mako___2021-3-26___10-53-40-932.png')

# import image as a numpy array, BGR style
img = cv.imread(img_path, cv.IMREAD_COLOR)
# img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# infer image
img_out, pred = Tussock.infer_image(img,
                                    imsave=True,
                                    save_dir='output',
                                    image_name=os.path.basename(img_path)[:-4],
                                    image_color_format='BGR')

# print where image is
print(f'img_out type = {type(img_out)}')
print(f'img_out size = {img_out.shape}')
print(f'img_out path = output/{os.path.basename(img_path)}') # assuming save_dir, image_name same as above

# print predictions
# print polygon/contour output
print('polygons:')
for i, poly in enumerate(pred['polygons']):
    print(f'{i}: {len(poly)} points')

# print boxes (derived from the polygons)
print('boxes:')
for i, box in enumerate(pred['bin_boxes']):
    print(f'{i}: {box}')

# print spray point
print('spraypoints:')
for i, poly_cen in enumerate(pred['poly_centroids']):
    print(f'{i}: {poly_cen}')

# handy debug code to examine output
# import code
# code.interact(local=dict(globals(), **locals()))
