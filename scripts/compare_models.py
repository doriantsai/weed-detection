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

WeedTemp = WM()
# load dataset objects
dataset_names = ['Tussock_v2',
                 'Tussock_v3_neg_test',
                 'Tussock_v3_neg_train_test']


# set the thresholds
# TODO call prcurve functions
nms_iou_thresh = 0.5
decision_iou_thresh = 0.5
confidence_thresh = np.linspace(0.99, 0.01, num=3, endpoint=True)
confidence_thresh = np.array(confidence_thresh, ndmin=1)
# TODO for 0.0 and 1.0 confidence threshold, produces nans because no tp


# provide a list of model names:
model_names = ['Tussock_v2',
               'Tussock_v2',
               'Tussock_v3_neg_train_test']

# iterate for each model_name:

results = []
WeedModelList = []
i = 0
for name in model_names:

    dataset_file = os.path.join('dataset_objects', dataset_names[i], dataset_names[i] + '.pkl')
    dso = WeedTemp.load_dataset_objects(dataset_file)

    WeedModel = WM()
    save_model_path = os.path.join('output', name, name + '.pth')
    WeedModel.load_model(save_model_path)
    WeedModel.set_model_name(name)
    WeedModel.set_model_path(save_model_path)

    # HACK for negative testing images
    if i == 1:
        name = 'Tussock_v3_neg_test'

    save_prcurve_folder = os.path.join('output', name, 'prcurve')
    res = WeedModel.get_prcurve(dso['ds_test'],
                                confidence_thresh=confidence_thresh,
                                nms_iou_thresh=nms_iou_thresh,
                                decision_iou_thresh=decision_iou_thresh,
                                save_folder=save_prcurve_folder,
                                imsave=True)

    results.append(res)
    WeedModelList.append(WeedModel)
    i += 1

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