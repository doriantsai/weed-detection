#! /usr/bin/env python

""" script to do model comparison """

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from weed_detection.WeedModel import WeedModel as WM

# init WM object
# load model
# call prcurve function

IMSHOW = True

# load dataset objects
dataset_file = os.path.join('dataset_objects', 'Tussock_v1', 'Tussock_v1.pkl')
# load dataset files via unpacking the pkl file
if os.path.isfile(dataset_file):
    with open(dataset_file, 'rb') as f:
        ds_train = pickle.load(f)
        ds_test = pickle.load(f)
        ds_val = pickle.load(f)
        dl_train = pickle.load(f)
        dl_test = pickle.load(f)
        dl_val = pickle.load(f)
        hp_train = pickle.load(f)
        hp_test = pickle.load(f)
        dataset_name = pickle.load(f)

# set the thresholds
# TODO call prcurve functions
nms_iou_thresh = 0.5
decision_iou_thresh = 0.5
confidence_thresh = np.linspace(0.99, 0.01, num=100, endpoint=True)
confidence_thresh = np.array(confidence_thresh, ndmin=1)
# TODO for 0.0 and 1.0 confidence threshold, produces nans because no tp


# provide a list of model names:
model_names = ['tussock_test_2021-05-16_16_13',
               'blah2']

# iterate for each model_name:

results = []
WeedModelList = []
for name in model_names:
    WeedModel = WM()
    save_model_path = os.path.join('output', name, name + '.pth')
    WeedModel.load_model(save_model_path)
    WeedModel.set_model_name(name)
    WeedModel.set_model_path(save_model_path)

    save_prcurve_folder = os.path.join('output', name, 'prcurve')
    res = WeedModel.get_prcurve(ds_val,
                                confidence_thresh=confidence_thresh,
                                nms_iou_thresh=nms_iou_thresh,
                                decision_iou_thresh=decision_iou_thresh,
                                save_folder=save_prcurve_folder))

    results.append(res)
    WeedModelList.append(WeedModel)

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
if IMSHOW:
    plt.show()

print('model comparison complete')
for i, m in enumerate(model_names):
    print(str(i) + ' model: ' + m)

# TODO add runtime notes?

import code
code.interact(local=dict(globals(), **locals()))