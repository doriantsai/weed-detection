#! /usr/bin/env python

""" test data augmentations """

import os
import numpy as np
import pickle
import json
import cv2 as cv
import torchvision
import torch

from PIL import Image
from SerratedTussockDataset import SerratedTussockDataset, RandomHorizontalFlip
from SerratedTussockDataset import Rescale, ToTensor, Blur, Compose
from SerratedTussockDataset import RandomVerticalFlip, RandomBrightness
from SerratedTussockDataset import RandomContrast, RandomHue, RandomSaturation

from inference import show_groundtruth_and_prediction_bbox, cv_imshow
torch.manual_seed(42)

dataset_name = 'Horehound_v0'
root_dir = os.path.join(os.sep, 'home', 'dorian', 'Data', 'agkelpie', 'Horehound_v0')
json_file = os.path.join('Annotations', 'annotations_horehound_210326_G507_location2.json')

dataset_lengths = (200, 1, 49)

rescale_size = 2056
prob = 1

tformtens = Compose([ToTensor()])
# tforms = Compose([Rescale(rescale_size), ToTensor()])
# tforms = Compose([Blur(11, 3), ToTensor()])
# tforms = Compose([RandomVerticalFlip(1), ToTensor()])
# tforms = Compose([RandomBrightness(1, 2), ToTensor()])
# tforms = Compose([RandomContrast(1, 2), ToTensor()])
# tforms = Compose([RandomHue(1, 0.5), ToTensor()])
# tforms = Compose([RandomSaturation(1, 1.5), ToTensor()])

# tforms = Compose([Rescale(rescale_size), ToTensor()])

ds = SerratedTussockDataset(root_dir, json_file, tformtens)
dst = SerratedTussockDataset(root_dir, json_file, tforms)

# grab one iteration of image and sample
i = 0
img_raw, smp_raw = ds[i]

img_tr, smp_tr = dst[i] # transformed, should be tensor and transformed

# show side-by-side comparison of two images, one should be transformed


img_raw_out = show_groundtruth_and_prediction_bbox(img_raw, smp_raw)
img_raw_name = os.path.join('output', 'img_raw.png')
cv.imwrite(img_raw_name, img_raw_out)
cv_imshow(img_raw_out, 'raw image')
# img_raw = np.array(img_raw)
# img_raw = cv.cvtColor(img_raw, cv.COLOR_RGB2BGR)
# cv.imwrite(img_raw_name, img_raw)
# wait_time = 2000 # ms
# winname = 'raw image'
# cv.namedWindow(winname, cv.WINDOW_NORMAL)
# cv.imshow(winname, img_raw)
# cv.waitKey(wait_time)
# cv.destroyWindow(winname)

# as a tensor, this image is a bit different
# img_tr
# img_tr = img_tr.numpy()
# img_tr = np.transpose(img_tr, (1, 2, 0))
# img_tr = cv.normalize(img_tr,
#                         None,
#                         alpha=0,
#                         beta=255,
#                         norm_type=cv.NORM_MINMAX,
#                         dtype=cv.CV_8U)

img_tr_out = show_groundtruth_and_prediction_bbox(img_tr, smp_tr)
img_tr_name = os.path.join('output', 'img_tr.png')
cv.imwrite(img_tr_name, img_tr_out)
cv_imshow(img_tr_out, 'transformed image')

import code
code.interact(local=dict(globals(), **locals()))