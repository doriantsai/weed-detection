#! /usr/bin/env python


""" script to train model after running split_image_data.py and create_datasets.py """

import os
from scripts.create_datasets import Tussock
# import time
# import pickle

# from weed_detection.WeedDataset import WeedDataset as WD
from weed_detection.WeedModel import WeedModel as WM
import matplotlib.pyplot as plt
import numpy as np


# create WM object
T3_test = WM()

# create datasets
dataset_names = ['Tussock_v3_neg_test',
                 'Tussock_v3_neg_train_test']

# provide a list of model names:
model_names = dataset_names

# iterate for each model_name:
WeedModelList = []
results = []
i = 0
for ds in dataset_names:

    root_dir = os.path.join('/home',
                            'dorian',
                            'Data',
                            'AOS_TussockDataset',
                            ds)
    img_folders = [os.path.join(root_dir, 'Images','Train'),
                os.path.join(root_dir, 'Images', 'Test'),
                os.path.join(root_dir, 'Images', 'Val')]

    ann_files = [os.path.join(root_dir, 'Annotations', 'annotations_tussock_21032526_G507_train.json'),
                os.path.join(root_dir, 'Annotations', 'annotations_tussock_21032526_G507_test.json'),
                os.path.join(root_dir, 'Annotations', 'annotations_tussock_21032526_G507_val.json')]


    # set hyper parameters of dataset
    batch_size = 10
    num_workers = 10
    learning_rate = 0.005
    momentum = 0.9
    weight_decay = 0.0001
    num_epochs = 100
    step_size = round(num_epochs / 2)
    shuffle = True
    rescale_size = 2056

    # make a hyperparameter dictionary
    hp={}
    hp['batch_size'] = batch_size
    hp['num_workers'] = num_workers
    hp['learning_rate'] = learning_rate
    hp['momentum'] = momentum
    hp['step_size'] = step_size
    hp['weight_decay'] = weight_decay
    hp['num_epochs'] = num_epochs
    hp['shuffle'] = shuffle
    hp['rescale_size'] = rescale_size

    hp_train = hp
    hp_test = hp
    hp_test['shuffle'] = False

    WeedModel = WM()
    dataset_path = WeedModel.create_train_test_val_datasets(img_folders,
                                                      ann_files,
                                                      hp,
                                                      ds)

    # dataset_file = os.path.join('dataset_objects', dataset_name0, dataset_name0 + '.pkl')

# load dataset files via unpacking the pkl file
    dso = WeedModel.load_dataset_objects(dataset_path)

# create other/model comparison
    WeedModel.train(model_name=ds,
                dataset_path=dataset_path,
                model_name_suffix=False)

    # --------------------------------------------------------------------------- #
    # set the thresholds
    # TODO call prcurve functions
    nms_iou_thresh = 0.5
    decision_iou_thresh = 0.5
    confidence_thresh = np.linspace(0.99, 0.01, num=100, endpoint=True)
    confidence_thresh = np.array(confidence_thresh, ndmin=1)
    # TODO for 0.0 and 1.0 confidence threshold, produces nans because no tp

    name = model_names[i] # technically, right now same as ds

    save_model_path = os.path.join('output', name, name + '.pth')
    WeedModel.load_model(save_model_path)
    WeedModel.set_model_name(name)
    WeedModel.set_model_path(save_model_path)

    save_prcurve_folder = os.path.join('output', name, 'prcurve')
    res = WeedModel.get_prcurve(dso['ds_test'],
                                confidence_thresh=confidence_thresh,
                                nms_iou_thresh=nms_iou_thresh,
                                decision_iou_thresh=decision_iou_thresh,
                                save_folder=save_prcurve_folder)

    results.append(res)
    WeedModelList.append(WeedModel)
    i += 1
    # end loop


#  res = {'precision': p_final,
#            'recall': r_final,
#            'ap': ap,
#            'f1score': f1score,
#            'confidence': c_final}

# now plot:
fig, ax = plt.subplots()
ap_list = []
for i, r in enumerate(results):
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
plt.show()

print('model comparison complete')
for i, m in enumerate(model_names):
    print(str(i) + ' model: ' + m)

# --------------------------------------------------------------------------_ #
import code
code.interact(local=dict(globals(), **locals()))