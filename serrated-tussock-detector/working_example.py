#! /usr/bin/env python

"""
Minimum working example for model inference
Author: Dorian Tsai
Date: 2021 April 20
Project: AOS Weed Detection

Given model weights, infer from an image
"""

import os
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

# TODO
# model file name
# read in model/build model
# have image name
# output from model input
# for more complete example, see inference.py


# setup device, use gpu if possible
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model file name
# NOTE: replace with own/local file name
model_name = 'Tussock_v0_8'
model_path = os.path.join('output', model_name, model_name + '.pth')

# import model structure from torchvision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
# NOTE: 2 classes - background and tussock

# import weights to model
model.load_state_dict(torch.load(model_path))

# image name
img_name = 'mako___2021-3-25___15-8-20-458.png'
img_path = os.path.join('/home',
                        'dorian',
                        'Data',
                        'AOS_TussockDataset',
                        'Tussock_v0',
                        'Images',
                        img_name)

# import image as a PIL image
img = Image.open(img_path)

# convert image to tensor
tform = transforms.ToTensor()
img_t = tform(img)

# NOTE: image may need to be recaled to specific size, eg, 800 pix in longest
# image dimension; however, at the moment, image is rescaled to original image
# size, so should not be necessary, otherwise, see ReScale transform in
# SerratedTussockDataset.py

# do not compute gradients, put model into evaluation mode
with torch.no_grad():
    model.eval()

    #  send model and image to device
    model.to(device)
    img_t = img_t.to(device)

    # model inference
    output = model([img_t])

# handy debug code to examine output
import code
code.interact(local=dict(globals(), **locals()))
