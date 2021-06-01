#! /usr/bin/env python

""" script to test image size """

import os
from weed_detection.WeedModel import WeedModel as WM
from weed_detection.PreProcessingToolbox import PreProcessingToolbox as PT
import numpy as np
import matplotlib.pyplot as plt
import time


# boolean to control if we need to train models or not
TRAIN = True
IMSHOW = False

# PT object
ProTool = PT()

# setup parameters/folders:
# dataset_folder = 'Tussock_v2_mini'
dataset_folder = 'Tussock_v2'
root_dir = os.path.join('/home', 'dorian','Data','AOS_TussockDataset', dataset_folder)

ann_master = os.path.join(root_dir, 'Annotations', 'annotations_tussock_21032526_G507_master.json')

ann_files = [os.path.join(root_dir, 'Annotations', 'annotations_tussock_21032526_G507_train.json'),
            os.path.join(root_dir, 'Annotations', 'annotations_tussock_21032526_G507_test.json'),
            os.path.join(root_dir, 'Annotations', 'annotations_tussock_21032526_G507_val.json')]

img_folders = [os.path.join(root_dir, 'Images','Train'),
               os.path.join(root_dir, 'Images', 'Test'),
               os.path.join(root_dir, 'Images', 'Validation')]


# sync ann_files with respective image folders
ProTool = PT()
ann_files_out = []
for i in range(len(img_folders)):
    ann_files_out.append(ProTool.sync_annotations(img_folders[i], ann_master, ann_files[i]))

# setup model for rescale sizes of:
image_sizes = [256, 512, 1024, 2056]
snapshot_epoch = [30, 30, 35, 65] # NOTE need to retrain several times

model_names = []
for i in range(len(image_sizes)):
    model_names.append(dataset_folder + '_' + str(image_sizes[i]))

dataset_names = model_names

print(dataset_names)


# set hyper parameters of dataset
batch_size = 10
num_workers = 10
learning_rate = 0.005
momentum = 0.9
weight_decay = 0.0001
num_epochs = 100
step_size = round(num_epochs / 2)
shuffle = True

# set the thresholds
nms_iou_thresh = 0.5
decision_iou_thresh = 0.5
confidence_thresh = np.linspace(0.99, 0.01, num=51, endpoint=True)
confidence_thresh = np.array(confidence_thresh, ndmin=1)

results = []
for i in range(len(image_sizes)):
    rescale_size = image_sizes[i]

    if TRAIN:
        #  make a hyperparameter dictionary
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

        # create dataset, which defines the hyperparameters for rescale size
        WeedModel = WM(model_name=model_names[i])
        ds_path = WeedModel.create_train_test_val_datasets(img_folders, ann_files, hp, dataset_names[i])

        # train model
        WeedModel.train(model_name=model_names[i],
                        dataset_path=ds_path,
                        model_name_suffix=False)

        # # delete weed model, because of memory limitations?
        # del(WeedModel)
    else:
        # load model from relevant model_names folder
        # set datapath to evaluate on
        # run pr_curve/model_compare code on the X number of models
        WeedModel = WM(model_name=model_names[i])



        save_model_path = os.path.join('output', model_names[i], model_names[i] + '.pth')
        WeedModel.load_model(save_model_path)
        WeedModel.set_model_name(model_names[i])
        WeedModel.set_model_path(save_model_path)
        WeedModel.set_snapshot(snapshot_epoch[i])

    dataset_file = os.path.join('dataset_objects', dataset_names[i], dataset_names[i] + '.pkl')
    dso = WeedModel.load_dataset_objects(dataset_file)

    save_prcurve_folder = os.path.join('output', model_names[i], 'prcurve')
    res = WeedModel.get_prcurve(dso['ds_test'],
                            confidence_thresh=confidence_thresh,
                            nms_iou_thresh=nms_iou_thresh,
                            decision_iou_thresh=decision_iou_thresh,
                            save_folder=save_prcurve_folder,
                            imsave=True)
    results.append(res)
        # WeedModelList.append(WeedModel)

# train model with rescale size set for each scale index
# do model comparison on each model with the same datasets (including both positive and negative images)

# get PR curve for all

# now plot:
fig, ax = plt.subplots()
ap_list = []
for i, r in enumerate(results):
    prec = r['precision']
    rec = r['recall']
    ap = r['ap']
    f1score = r['f1score']
    c = r['confidence']

    m_str = 'm={}, ap={:.3f}'.format(model_names[i], ap)
    ax.plot(rec, prec, label=m_str)
    ap_list.append(ap)

ax.legend()
plt.xlabel('recall')
plt.ylabel('precision')
plt.grid(True)
plt.title('model comparison: PR curve')

mdl_names_str = "".join(model_names)
save_plot_name = os.path.join('output', 'model_compare_' +  mdl_names_str + '.png')
plt.savefig((save_plot_name))
if IMSHOW:
    plt.show()

print('model comparison complete')
for i, m in enumerate(model_names):
    print(str(i) + ' model: ' + m)

# also, get a model inference times, (JUST model inference)
# plot on a graph vs time to compute for a single image

# also, create a mini test set of 100 images for training faster!

import code
code.interact(local=dict(globals(), **locals()))

# ============================== #
# now code to plot computation time vs image size

# CPU = False
# # image_sizes
# for i in range(len(image_sizes)):
#     # load model
#     WeedModel = WM(model_name=model_names[i])

#     dataset_file = os.path.join('dataset_objects', dataset_names[i], dataset_names[i] + '.pkl')
#     dso = WeedModel.load_dataset_objects(dataset_file)

#     save_model_path = os.path.join('output', model_names[i], model_names[i] + '.pth')
#     WeedModel.load_model(save_model_path)
#     WeedModel.set_model_name(model_names[i])
#     WeedModel.set_model_path(save_model_path)
#     WeedModel.set_snapshot(snapshot_epoch[i])

#     # get image:
#     dataset = dso['test']
#     image, sample = next(iter(dataset))
#     # start time
#     start_time = time.time()

#     if CPU:
#         WeedModel.device('cpu')
#     output = WeedModel.model([image])