#! /usr/bin/env python

""" get time plot """

import os
from weed_detection.WeedModel import WeedModel as WM
from weed_detection.PreProcessingToolbox import PreProcessingToolbox as PT
import numpy as np
import matplotlib.pyplot as plt
import time
import torch



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

# setup model for rescale sizes of: NOTE the first time the model is called, it
# is much slower as it seems something like cuda is loading the graph onto the
# GPU, I could not find any concrete explanation
# so work-around for compute times is any after the first
image_sizes = [256, 256, 512, 1024, 2056]
# import code
# code.interact(local=dict(globals(), **locals()))
# HACK temp, try reversing image_sixsizes order
# image_sizes = image_sizes[::-1]
# image_sizes = [512, 256, 2056, 1024]
snapshot_epoch = [30, 30, 30, 35, 65] # NOTE need to retrain several times

model_names = []
for i in range(len(image_sizes)):
    model_names.append(dataset_folder + '_' + str(image_sizes[i]))

dataset_names = model_names


CPU = False
if CPU:
    device=torch.device('cpu')
else:
    # gpu is default, but do so explicity
    device=torch.device('cuda')

# image_sizes
NUM_IMAGES = 100
model_time = []
for i in range(len(image_sizes)):
    # load model
    WeedModel = WM(model_name=model_names[i], device=device)

    dataset_file = os.path.join('dataset_objects', dataset_names[i], dataset_names[i] + '.pkl')
    dso = WeedModel.load_dataset_objects(dataset_file)

    save_model_path = os.path.join('output', model_names[i], model_names[i] + '.pth')
    WeedModel.load_model(save_model_path, map_location="cuda:0")
    WeedModel.set_model_name(model_names[i])
    WeedModel.set_model_path(save_model_path)
    WeedModel.set_snapshot(snapshot_epoch[i])



    # get image:
    dataset = dso['ds_test']
    # image, sample = next(iter(dataset))
    # import code
    # code.interact(local=dict(globals(), **locals()))

    # start time

    with torch.no_grad():

        image_time = []
        for j in range(NUM_IMAGES):
            image, sample = dataset[j]
            image = image.to(device)
            model = WeedModel.model.to(device)
            # WeedModel.model.eval()
            model.eval()
            start_time = time.time()
            # output = WeedModel.model([image])
            output = model([image])
            end_time = time.time()
            sec = end_time - start_time
            image_time.append(sec)

    # average image_time by NUM_IMAGES
    mean_time = np.mean(np.array(image_time))
    print('mean inference time: {} sec'.format(mean_time))

    model_time.append(mean_time)

model_time = np.array(model_time)

# plot image_sizes vs modeL_time

fig, ax = plt.subplots()
ax.plot(image_sizes[1::], model_time[1::], 'o-')
plt.xlabel('image sizes [pix]')
plt.ylabel('model inference times [s]')
plt.grid(True)
mdl_names_str = "".join(model_names)
save_plot_name = os.path.join('output', 'model_times_' +  mdl_names_str + '.png')
plt.savefig((save_plot_name))
plt.show()

import code
code.interact(local=dict(globals(), **locals()))