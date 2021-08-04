#! /usr/bin/env python

"""
Attempt to get featurees out of deep weeds classifier model using register hooks
"""

# file locations (images/labels/model)
# import model
# feature extractor
# run model forward pass

# from show_class_distribution import CLASS_COLOURS
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
# import torch.nn as nn

from torch import nn, Tensor
from torchvision.models import resnet, resnet50
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models

from classifier_deepweeds import DeepWeedsDataset as DWD
from classifier_deepweeds import Rescale, RandomCrop, ToTensor

from typing import Dict, Iterable, Callable

# fix random seeds, so that results are reproducable:
# https://learnopencv.com/t-sne-for-feature-visualization/
import random
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

from sklearn.manifold import TSNE

# from https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904

# class FeatureExtractor()

# wrapper that prints the output shapes of each layer's output
class VerboseExecution(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # Register a hook for each layer
        for name, layer in self.model.named_children():
            layer.__name__ = name
            layer.register_forward_hook(
                lambda layer, _, output: print(f"{layer.__name__}: {output.shape}")
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

## example code to run/print layer shapes of resnet50
# verbose_resnet = VerboseExecution(resnet50())
# dummy_input = torch.ones(10, 3, 224, 224)
# _ = verbose_resnet(dummy_input)

# wrapper that extracts features

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features


def scale_to_range(x):
    # compute distribution range
    value_range = (np.max(x) - np.min(x))
    
    # move distribution start zero
    starts_zero = x - np.min(x)

    # scale to 1
    return starts_zero / value_range




########### classes #############
CLASSES = (0, 1, 2, 3, 4, 5, 6, 7)
CLASS_NAMES = ('Chinee apple',
                'Lantana',
                'Parkinsonia',
                'Parthenium',
                'Prickly acacia',
                'Rubber vine',
                'Siam weed',
                'Snake weed')
CLASS_DICT = {i: CLASS_NAMES[i] for i in range(0, len(CLASSES))}
# set colours for histogram based on ones used in paper
    # RGB
pink = np.r_[255, 105,180]/255
blue = np.r_[0, 0, 255]/255
green = np.r_[0, 255, 0]/255
yellow = np.r_[255, 255, 0]/255
cyan = np.r_[0, 255, 255]/255
red = np.r_[255, 0, 0]/255
purple = np.r_[128, 0, 128]/255
orange = np.r_[255, 127, 80]/255
CLASS_COLOURS = [pink,
                blue,
                green,
                yellow,
                cyan,
                red,
                purple,
                orange]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load model
model_location = os.path.join('saved_model', 'training1')
model_name = 'dw_r50_s200_i0.pth'
model_path = os.path.join(model_location, model_name)

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(in_features=2048, out_features=len(CLASSES), bias=True)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# verbose_resnet = VerboseExecution(model())

# for now, run it on everything
img_dir = 'nonnegative_images'
img_list = os.listdir(img_dir)
lbl_dir = 'labels'
labels_file = os.path.join(lbl_dir, 'nonnegative_labels.csv')

# i = 0
# img_name = img_list[i]
# img_path = os.path.join(img_dir, img_name)

tforms = transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()])
dataset = DWD(labels_file, img_dir, tforms)
nimg = len(dataset)
print('dataset length =', nimg)

## example code to run/print feature of resenet50 layer4 and avgpool
# avgpool is where the features are
resnet_features = FeatureExtractor(model, layers=["avgpool"])

# run feature code over entire dataset
print('getting features from network')

features = []
img_ids = []
lbls = []
with torch.no_grad():
    for i in range(len(dataset)):
        sample = dataset[i]
        img, lbl, id = sample['image'], sample['label'], sample['image_id']

        img.to(device)
        img = img.unsqueeze(0)
        img = img.float().cuda()
        # import code
        # code.interact(local=dict(globals(), **locals()))
        # _ = verbose_resnet(img)
        output  = model(img)
        _, pred = torch.max(output, 1)
        # print(f'{i}: label = {lbl}, {CLASSES[lbl]}')
        # print('Predicted: ', ' '.join('%5s' % CLASSES[pred[j]] for j in range(nimg)))
        p = pred.item()
        # print(f'{i}: pred = {p}, {CLASSES[p]}')

        # _ = verbose_resnet(img)

        # import code
        # code.interact(local=dict(globals(), **locals()))

        avgpool = resnet_features(img)
        f = avgpool['avgpool'].cpu().numpy()
        f = np.squeeze(f)
        features.append(f)

        img_ids.append(id)
        lbls.append(lbl)

        if i % 1000 == 0:
            print(f'feature {i}')
        # if i == 5:
        #     break

# convert list to 2D array
features = np.array(features)

# get tsne
n_components = 2
tsne = TSNE(n_components=n_components).fit_transform(features)

# print({name: output.shape for name, output in features.items()})

if n_components == 2:
    TWO_DIM = True
    THREE_DIM = False
elif n_components == 3:
    TWO_DIM = False
    THREE_DIM = True
else:
    print('error: more n_components not yet implemented')

if TWO_DIM:
    # extract x, y coords representing image position on tsne plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # tx = scale_to_range(tx)
    # ty = scale_to_range(ty)

    # plot for every class, adding separate scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for cls in CLASSES:
        # find samples in current class
        idx = [i for i, l in enumerate(lbls) if l == cls]

        # extract coordinates of points for this class only
        current_tx = np.take(tx, idx)
        current_ty = np.take(ty, idx)

        # convert class color to matplotlib format
        colour = CLASS_COLOURS[cls]

        # add scatter plot
        ax.scatter(current_tx, current_ty, c=colour, label=CLASS_NAMES[cls])

    # build legend
    ax.legend(loc='best')

    # show
    plt.show()

elif THREE_DIM:
    # extract x, y coords representing image position on tsne plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tz = tsne[:, 2]

    tx = scale_to_range(tx)
    ty = scale_to_range(ty)
    tz = scale_to_range(tz)

    # plot for every class, adding separate scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for cls in CLASSES:
        # find samples in current class
        idx = [i for i, l in enumerate(lbls) if l == cls]

        # extract coordinates of points for this class only
        current_tx = np.take(tx, idx)
        current_ty = np.take(ty, idx)
        current_tz = np.take(tz, idx)

        # convert class color to matplotlib format
        colour = CLASS_COLOURS[cls]

        # add scatter plot
        ax.scatter(current_tx, current_ty, current_tz, c=colour, label=CLASS_NAMES[cls])

    # build legend
    ax.legend(loc='best')

    # show
    plt.show()

import code
code.interact(local=dict(globals(), **locals()))