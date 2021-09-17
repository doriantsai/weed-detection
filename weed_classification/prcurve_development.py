#! /usr/bin/env python

"""
pr curves for classifier between development and deployment conditions
using torchmetrics
"""

import os
import torch
import pickle as pkl
import code
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from itertools import cycle
from deepweeds_dataset import DeepWeedsDataset, Rescale, RandomCrop, ToTensor, CLASSES, CLASS_NAMES
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torchmetrics import PrecisionRecallCurve, AveragePrecision

np.random.seed(42)
torch.manual_seed(42)

# CLASS_NAMES = ('Chinee apple',
#                'Lantana',
#                'Parkinsonia',
#                'Parthenium',
#                'Prickly acacia',
#                'Rubber vine',
#                'Siam weed',
#                'Snake weed',
#                'Negative')
# CLASSES = np.arange(0, len(CLASS_NAMES))

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# load model
model_file = 'dw_r50_s500_i0.pth'
model_path = os.path.join('saved_model', 'development_training1', model_file)
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(in_features=2048, out_features=len(CLASSES), bias=True)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()


LOAD_DATA = False
if LOAD_DATA:
    lbls_file = 'development_labels_trim.pkl'
    data_path = os.path.join('dataset_objects',
                             'development_labels_trim',
                             lbls_file)
    with open(data_path, 'rb') as f:
        # TODO save full dataset to pkl file
        ted = pkl.load(f)
        vad = pkl.load(f)
        trd = pkl.load(f)
        trdl = pkl.load(f)
        vadl = pkl.load(f)
        tedl = pkl.load(f)

    # TODO sum/length of ted, vad, trd for len(full_data)
    dataloader_pr = tedl
else:
    lbl_dir = './labels'
    img_dir = './images'

    lbls_file = 'development_labels_trim.csv'
    lbls_file = os.path.join(lbl_dir, lbls_file)

    tforms = transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()])
    full_data = DeepWeedsDataset(lbls_file, img_dir, tforms)

    batch_size = 10
    num_workers = 10
    tedl = DataLoader(full_data,
                      batch_size=batch_size,
                      shuffle=False,
                      num_workers=num_workers)
    dataloader_pr = tedl


# infer model over entire dataloader

predicted_values = torch.empty(size=([0]), device=device)
predicted_labels = torch.empty(size=([0]), device=device)
actual_labels = torch.empty(size=([0]), device=device)
outputs = torch.empty(0, len(CLASSES), device=device)
with torch.no_grad():
    for sample in dataloader_pr:
        imgs, lbls = sample['image'].float(), sample['label']

        imgs = imgs.to(device)
        lbls = lbls.to(device)

        outs = model(imgs)

        # for single image/max scores
        pred_val, pred_indx = torch.max(outs.data, 1)
        # predicted_values.extend(pred_val.tolist())

        predicted_values = torch.cat((predicted_values, pred_val), dim=0)
        predicted_labels = torch.cat((predicted_labels, pred_indx), dim=0)
        actual_labels = torch.cat((actual_labels, lbls), dim=0)

        # for all scores of all classes for each image
        outputs = torch.cat((outputs, outs), dim=0)

# calculate pr curves using torchmetrics
pr_curve = PrecisionRecallCurve(num_classes=len(CLASSES))
precision, recall, thresholds = pr_curve(outputs, actual_labels)
ap_compute = AveragePrecision(num_classes=len(CLASSES))
ap = ap_compute(outputs, actual_labels)

# TODO calculate AP scores
colors = ['pink', 'blue', 'green', 'yellow', 'cyan', 'red', 'purple', 'orange', 'grey']

# f-score contours
plt.figure(figsize=(7, 8))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
lines.append(l)
labels.append('iso-f1 curves')


for i in range(len(CLASSES)):
    l, = plt.plot(recall[i].cpu().numpy(),
                    precision[i].cpu().numpy(),
                    color=colors[i],
                    lw=2)
    lines.append(l)
    labels.append('{0} (ap={1:0.2f})'.format(CLASS_NAMES[i], ap[i].cpu().numpy()))

fig = plt.gcf()
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('PR-Curve for Deep Weeds Classifier')
plt.legend(lines, labels)

base_name = os.path.basename(lbls_file)
save_img_path = os.path.join('output', base_name[:-4] + '_prcurves2.png')
# plt.show()
plt.savefig(save_img_path)

# end code
code.interact(local=dict(globals(), **locals()))


