#! /usr/bin/env python

""" script to infer from model for single image/batch """

import os
import pickle
from weed_detection.WeedModel import WeedModel as WM
from weed_detection.WeedDatasetPoly import Compose, Rescale, ToTensor
import time
import matplotlib.pyplot as plt
# init WM object
# load model
# call inference functions

# init WM object
Tussock = WM()

# load dataset objects
# dataset_name = 'Tussock_v0_mini'
dataset_name = 'Tussock_v4_poly286'

DATASET_FILE_EXISTS = False
if DATASET_FILE_EXISTS:
    dataset_file = os.path.join('dataset_objects',
                                dataset_name,
                                dataset_name + '.pkl')
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

    # just choose which dataset object to use for this script
    ds_infer = ds_train

else:
    root_dir = os.path.join('/home',
                            'dorian',
                            'Data',
                            'AOS_TussockDataset',
                            dataset_name)
    img_dir = os.path.join(root_dir, 'Images', 'PolySubset')
    mask_dir = os.path.join(root_dir, 'Masks', 'PolySubset')
    ann_file = 'via_project_07Jul2021_08h00m_240_polysubset_bootstrap.json'
    ann_path = os.path.join(root_dir, 'Annotations', ann_file)

    # set hyper parameters of dataset
    batch_size = 10
    num_workers = 10
    learning_rate = 0.005 # 0.002
    momentum = 0.9 # 0.8
    weight_decay = 0.0001
    num_epochs = 50
    step_size = round(num_epochs / 2)
    shuffle = True
    rescale_size = int(1024)

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
    
    tform = Compose([Rescale(rescale_size),
                     ToTensor()])
    dataset, dataloader = Tussock.create_dataset_dataloader(root_dir=root_dir,
                                      json_file=ann_file,
                                      transforms=tform,
                                      hp=hp,
                                      annotation_type='poly',
                                      mask_dir=mask_dir,
                                      img_dir=img_dir)


    ds_infer = dataset



# load model
# model_name = 'tussock_test_2021-05-16_16_13'
# model_name = 'Tussock_v0_mini_2021-06-14_13_25'
# model_name = 'Tussock_v4_poly286_2021-07-14_10_24'
model_name = 'Tussock_v4_poly286_2021-07-15_11_08'
save_model_path = os.path.join('output',
                               model_name,
                               model_name + '.pth')
Tussock.load_model(save_model_path)
Tussock.set_model_name(model_name)
Tussock.set_model_path(save_model_path)
Tussock.set_model_folder(model_name)
Tussock.set_snapshot(20)



# images, targets = next(iter(dl_train))
# images = list(image for image in images)
# targets = [{k: v for k, v in t.items()} for t in targets]
# model = Tussock.model
# model.eval()
# model.cpu()
# import code
# code.interact(local=dict(globals(), **locals()))

# output = model(images, targets)

# import code
# code.interact(local=dict(globals(), **locals()))

# model = Tussock.build_maskrcnn_model(num_classes=2)



# run model inference on single image batch
images, samples = next(iter(dataloader))
bs = 1 # hp_test['batch_size']
for i in range(bs):
    # import code
    # code.interact(local=dict(globals(), **locals()))
    image = images[i]
    sample = samples[i]
    image_id = sample['image_id']
    start_time = time.time()
    image_out, pred = Tussock.infer_image(image,
                                          sample=sample,
                                          imshow=True,
                                          imsave=True,
                                          conf_thresh=0.2)
    print('{}: {}'.format(i, image_id))
    print('   pred = {}'.format(pred))
    end_time = time.time()
    sec = end_time - start_time
    print('cycle time: {} sec'.format(sec))

import code
code.interact(local=dict(globals(), **locals()))