#! /usr/bin/env python

"""
pr curves for classifier, between development and deployment conditions
"""

from sklearn.preprocessing import label_binarize
import numpy as np
import os
import pandas as pd
import pickle
from deepweeds_dataset import DeepWeedsDataset, Rescale, RandomCrop, ToTensor, CLASSES, CLASS_NAMES
from torchvision import models
import torch.nn as nn
import torch

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

########### classes #############
CLASS_NAMES = ('Chinee apple',
                'Lantana',
                'Parkinsonia',
                'Parthenium',
                'Prickly acacia',
                'Rubber vine',
                'Siam weed',
                'Snake weed',
                'Negative')
CLASSES = np.arange(0, len(CLASS_NAMES))
CLASS_DICT = {i: CLASS_NAMES[i] for i in range(0, len(CLASSES))}

# load model
model_file = 'dw_r50_s500_i0.pth'
model_path = os.path.join('saved_model', 'development_training1', model_file)
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(in_features=2048, out_features=len(CLASSES), bias=True)
model.load_state_dict(torch.load(model_path))
print(f'model loaded {model_path}')
# print(model)

# load data
data_file = 'development_labels_trim.pkl'
data_path = os.path.join('dataset_objects', 'development_labels_trim', data_file)

with open(data_path, 'rb') as file:
    td = pickle.load(file)
    vd = pickle.load(file)
    test_dataset = pickle.load(file)
    tdl = pickle.load(file)
    vdl = pickle.load(file)
    test_loader = pickle.load(file)


# print(td)

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# infer model over entire dataset,
model.to(device)

predicted_values = []
predicted_labels = []
actual_labels = []
with torch.no_grad():
    for step, sample_batch in enumerate(vdl):
        imgs, lbls = sample_batch['image'].float(), sample_batch['label']

        imgs = imgs.to(device)

        lbls = lbls.to(device)

        outputs = model(imgs)
        pred_val, pred_indx = torch.max(outputs.data, 1)

        predicted_values.extend(pred_val.tolist())
        predicted_labels.extend(pred_indx.tolist())
        actual_labels.extend(lbls.tolist())

        if step >= 2:
            break


# TODO run this code on pure negative images to see if major confusion


# TODO need to binarize then OneVsRest style?
# ap = average_precision_score(actual_labels, predicted_values)
# print('Average precision-recall score: {0:0.2f}'.format(
#       ap))

import code
code.interact(local=dict(globals(), **locals()))