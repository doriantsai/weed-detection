#! /usr/bin/env python

"""
model evaluation/comparison
specify X models
get pr-curves + stats for each
combine output prcurves/ap
should be used to save "which one is better"
"""


import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

from find_file import find_file
from prcurve_singleiteration import get_prcurve
from train import build_model
from split_dataset import collate_fn  # TODO move this into SerratedTussockDataset


IMSHOW = False

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# list models to compare:
# we need full path names
# model_names = ['Tussock_v0_13',
#                'Tussock_v0_14']
model_names = ['Horehound_v0_0',
               'Horehound_v0_1']

model_folders = []
saved_model_names = []
saved_model_paths = []
for m in model_names:
    model_folder = os.path.join('output', m)
    saved_model_name = find_file('.pth', model_folder)
    saved_model_path = os.path.join(model_folder, saved_model_name[0])

    model_folders.append(model_folder)
    saved_model_names.append(saved_model_name)
    saved_model_paths.append(saved_model_path)

print(saved_model_paths)

# load the datasets
# dataset location
dataset_name = 'Tussock_v0'
data_save_path = os.path.join('output',
                            'dataset',
                            dataset_name,
                            dataset_name + '.pkl')
with open(data_save_path, 'rb') as f:
    dataset_tform_test = pickle.load(f)
    dataset_tform_train = pickle.load(f)
    dataset_train = pickle.load(f)
    dataset_val = pickle.load(f)
    dataset_test = pickle.load(f)
    dataloader_test = pickle.load(f)
    dataloader_train = pickle.load(f)
    dataloader_val = pickle.load(f)
    hp = pickle.load(f)

# set the thresholds
nms_iou_thresh = 0.5
decision_iou_thresh = 0.5
# DECISION_CONF_THRESH = 0.5
confidence_thresh = np.linspace(0.99, 0.01, num=101, endpoint=True)
confidence_thresh = np.array(confidence_thresh, ndmin=1)

res = []
for m in model_names:

    # load model
    model_folder = os.path.join('output', m)
    saved_model_name = find_file('.pth', model_folder)
    saved_model_path = os.path.join(model_folder, saved_model_name[0])

    model = build_model(num_classes=2)
    model.load_state_dict(torch.load(saved_model_path))
    print('loading model: {}'.format(saved_model_path))

    results = get_prcurve(model,
                          dataset_test,
                          confidence_thresh,
                          nms_iou_thresh,
                          decision_iou_thresh,
                          m,
                          device,
                          imsave=False)
    res.append(results)


# now do the plotting:
fig, ax = plt.subplots()
ap_list = []
for i, r in enumerate(res):
    prec = r['precision']
    rec = r['recall']
    ap = r['ap']
    f1score = r['f1score']
    c = r['confidence']

    m_str = 'm={}, ap={:.2f}'.format(model_names[i], ap)
    ax.plot(rec, prec, label=m_str)
    ap_list.append(ap)

ax.legend()
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('model comparison: PR curve')

mdl_names_str = "".join(model_names)
save_plot_name = os.path.join('output', 'model_compare_' +  mdl_names_str + '.png')
plt.savefig((save_plot_name))
if IMSHOW:
    plt.show()

print('model comparison complete')
for i, m in enumerate(model_names):
    print(str(i) + ' model: ' + m)

import code
code.interact(local=dict(globals(), **locals()))

