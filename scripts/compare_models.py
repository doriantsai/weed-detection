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
# dataset_names = ['Tussock_v2',
#                  'Tussock_v3_neg_test',
#                  'Tussock_v3_neg_train_test']

dataset_names = ['2021-03-25_MFS_Tussock',
                 '2021-03-25_MFS_Tussock']
                #  'Tussock_v3_neg_train

# set the thresholds
# TODO call prcurve functions
nms_iou_thresh = 0.5
decision_iou_thresh = 0.5
confidence_thresh = np.linspace(0.99, 0.01, num=25, endpoint=True)
confidence_thresh = np.array(confidence_thresh, ndmin=1)
# TODO for 0.0 and 1.0 confidence threshold, produces nans because no tp


# provide a list of model names:
# model_names = ['Tussock_v2 epoch 100',
#                'Tussock_v2 epoch 20']
            #    'Tussock_v3_neg_train_test']
model_names = ['2021-03-25_MFS_Tussock_FasterRCNN_2021-09-01_16_49',
               '2021-03-25_MFS_Tussock_MaskRCNN_2021-08-31_19_33']

# where to store the results
# model_folders = [dataset_names[0],
#                  dataset_names[1]]
model_folders = model_names

dataset_object_names = ['2021-03-25_MFS_Tussock',
                        '2021-03-25_MFS_Tussock']
legend_names = ['FasterRCNN', 'MaskRCNN']

ann_types = ['box', 'poly']
# iterate for each model_name:

results = []
WeedModelList = []
for i, name in enumerate(model_names):

    # dataset_file = os.path.join('dataset_objects', dataset_names[i], dataset_names[i] + '.pkl')
    dataset_file = os.path.join('dataset_objects', dataset_object_names[i], dataset_object_names[i] + '.pkl')
    dso = WeedTemp.load_dataset_objects(dataset_file)

    WeedModel = WM(model_name=name, model_folder=model_folders[i])
    save_model_path = os.path.join('output', model_folders[i], model_folders[i] + '.pth')
    # import code
    # code.interact(local=dict(globals(), **locals()))
    WeedModel.load_model(save_model_path, annotation_type=ann_types[i])
    WeedModel.set_model_name(name)
    WeedModel.set_model_path(save_model_path)

    # TEMP changing epoch for "early stopping"
    if i == 0:
        WeedModel.set_snapshot(25)
    elif i == 1:
        WeedModel.set_snapshot(20)

    # import code
    # code.interact(local=dict(globals(), **locals()))

    save_prcurve_folder = os.path.join('output', name, 'prcurve')
    res = WeedModel.get_prcurve(dso['ds_test'],
                                confidence_thresh=confidence_thresh,
                                nms_iou_thresh=nms_iou_thresh,
                                decision_iou_thresh=decision_iou_thresh,
                                save_folder=save_prcurve_folder,
                                imsave=True,
                                annotation_type=ann_types[i])

    results.append(res)
    WeedModelList.append(WeedModel)
    # i += 1

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

    # m_str = 'm={}, ap={:.2f}'.format(model_names[i], ap)
    m_str = 'm={}, ap={:.2f}'.format(legend_names[i], ap)
    ax.plot(rec, prec, label=m_str)
    ap_list.append(ap)

ax.legend()
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('model comparison: PR curve')

mdl_names_str = "".join(legend_names)
save_plot_name = os.path.join('output', 'model_compare_' +  mdl_names_str + '.png')
plt.savefig((save_plot_name))
if IMSHOW:
    plt.show()

print('model comparison complete')
for i, m in enumerate(model_names):
    print(str(i) + ' model: ' + m)
    print(str(i) + ' name: ' + legend_names[i])

# TODO add runtime notes?

import code
code.interact(local=dict(globals(), **locals()))