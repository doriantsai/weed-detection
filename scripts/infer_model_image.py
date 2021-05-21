#! /usr/bin/env python

""" script to infer from model for single image/batch """

import os
import pickle
from weed_detection.WeedModel import WeedModel as WM
import time

# init WM object
# load model
# call inference functions

# load dataset objects
dataset_file = os.path.join('dataset_objects',
                            'Tussock_v2',
                            'Tussock_v2.pkl')
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

# init WM object
Tussock = WM()

# load model
model_name = 'tussock_test_2021-05-16_16_13'
save_model_path = os.path.join('output',
                               model_name,
                               model_name + '.pth')
Tussock.load_model(save_model_path)
Tussock.set_model_name(model_name)
Tussock.set_model_path(save_model_path)

# run model inference on single image batch
images, samples = next(iter(dl_test))
bs = 1 # hp_test['batch_size']
for i in range(bs):
    image = images[i]
    sample = samples[i]
    image_id = sample['image_id']
    start_time = time.time()
    image_out, pred = Tussock.infer_image(image,
                                          sample=sample,
                                          imshow=False,
                                          imsave=True)
    print('{}: {}'.format(i, image_id))
    print('   pred = {}'.format(pred))
    end_time = time.time()
    sec = end_time - start_time
    print('cycle time: {} sec'.format(sec))

import code
code.interact(local=dict(globals(), **locals()))