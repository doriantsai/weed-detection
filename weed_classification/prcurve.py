#! /usr/bin/env python

"""
pr curves for classifier, between development and deployment conditions
"""

import code
from itertools import cycle
from sklearn.preprocessing import label_binarize
import numpy as np
import os
import pandas as pd
import pickle
from deepweeds_dataset import DeepWeedsDataset, Rescale, RandomCrop, ToTensor, CLASSES, CLASS_NAMES
from torchvision import models
import torch.nn as nn
import torch
from torchvision import transforms, utils, models
from torch.utils.data import Dataset, DataLoader


from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

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
LOAD_DATA = False
if LOAD_DATA:
    lbls_file = 'development_labels_trim.pkl'

    data_path = os.path.join(
        'dataset_objects', 'development_labels_trim', lbls_file)

    with open(data_path, 'rb') as file:
        td = pickle.load(file)
        vd = pickle.load(file)
        test_dataset = pickle.load(file)
        tdl = pickle.load(file)
        vdl = pickle.load(file)
        test_loader = pickle.load(file)

    dataloader_do = test_loader
else:
    labels_folder = './labels'
    images_folder = './images'

    # lbls_file = 'development_labels_trim.csv'
    lbls_file = 'deployment_labels.csv'
    labels_file = os.path.join(labels_folder, lbls_file)

    tforms = transforms.Compose([
        Rescale(256),
        RandomCrop(224),
        ToTensor()
    ])
    full_dataset = DeepWeedsDataset(csv_file=labels_file,
                                    root_dir=images_folder,
                                    transform=tforms)
    batch_size = 10
    num_workers = 10
    tdl = DataLoader(full_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers)
    dataloader_do = tdl
# print(td)


# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# infer model over entire dataset,
model.to(device)

predicted_values = []
predicted_labels = []
actual_labels = []
outputs = []
with torch.no_grad():
    for step, sample_batch in enumerate(dataloader_do):
        imgs, lbls = sample_batch['image'].float(), sample_batch['label']

        imgs = imgs.to(device)

        lbls = lbls.to(device)

        outs = model(imgs)

        # for single image/max scores
        pred_val, pred_indx = torch.max(outs.data, 1)
        predicted_values.extend(pred_val.tolist())
        predicted_labels.extend(pred_indx.tolist())
        actual_labels.extend(lbls.tolist())

        # for all scores of all classes for each image
        outputs.append(outs.cpu().numpy())

        # if step >= 2:
        #     break


# TODO run this code on pure negative images to see if major confusion


# ap = average_precision_score(actual_labels, predicted_values)
# print('Average precision-recall score: {0:0.2f}'.format(
#       ap))
# TODO label_binarize/sklearn requires negative labels be less than positive labels? makes sense, but hasn't been done in all  rest of code/deepweeds dataset

actual_labels = np.array(actual_labels)
bin_classes = label_binarize(actual_labels, classes=CLASSES)

# need to reshape outputs into a nx9 array
predictions = np.vstack(outputs)

# for each class, find p, r, ap:
# NOTE: ask David, why dictionary? what more freedom does it grant over list?
precision = dict()
recall = dict()
average_precision = dict()
for i in range(len(CLASSES)):
    precision[i], recall[i], _ = precision_recall_curve(
        bin_classes[:, i], predictions[:, i])
    average_precision[i] = average_precision_score(
        bin_classes[:, i], predictions[:, i])

precision['micro'], recall['micro'], _ = precision_recall_curve(
    bin_classes.ravel(), predictions.ravel())
average_precision['micro'] = average_precision_score(
    bin_classes, predictions, average="micro")

a = average_precision['micro']
print(f'ap score, micro-averaged over all classes {a:0.2f}')


# the plot!
# plt.figure()
# plt.step(recall['micro'], precision['micro'], where='post')
# plt.xlabel('recall')
# plt.ylabel('precision')
# plt.ylim([0, 1.05])
# plt.xlim([0, 1])
# plt.title('Average precision score, micro-avg over all classes: AP{0:0.2f}'.format(a))
# plt.show()

colors = cycle(['pink', 'blue', 'green', 'yellow',
                'cyan', 'red', 'purple', 'orange', 'grey'])
plt.figure(figsize=(7, 8))

# make f-score contours
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
l, plt.plot(recall['micro'], precision['micro'], color='gold', lw=2)
lines.append(l)
labels.append('micro-avg pr (area={0:0.2f})'.format(a))

for i, color in zip(range(len(CLASSES)), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('pr for class {0} (ap={1:0.2f}), {2}'.format(
        i, average_precision[i], CLASS_NAMES[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.4)
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('PR-Curve for Deep Weeds Classifier')
plt.legend(lines, labels, loc=(0.25, -.7), prop=dict(size=9))
# plt.show()

save_img_path = os.path.join('output', lbls_file[:-4] + '_prcurves.png')
plt.savefig(save_img_path)

code.interact(local=dict(globals(), **locals()))
