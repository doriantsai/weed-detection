#! /usr/bin/env python

""" script to infer from model for single image/batch """

import os
import pickle
from weed_detection.WeedModel import WeedModel as WM
import time
import matplotlib.pyplot as plt
# init WM object
# load model
# call inference functions

# load dataset objects
dataset_name = 'Tussock_v4_poly286'
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

# init WM object
Tussock = WM()

# load model
# model_name = 'tussock_test_2021-05-16_16_13'
# model_name = 'Tussock_v0_mini_2021-06-14_13_25'
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


import numpy as np
import matplotlib.pyplot as plt
import random as rng
import cv2 as cv

# run model inference on single image batch
images, samples = next(iter(dl_train))
bs = 1 # hp_test['batch_size']
for i in range(bs):
    image = images[i]
    sample = samples[i]
    image_id = sample['image_id']
    start_time = time.time()
    image_out, pred = Tussock.infer_image(image,
                                          sample=sample,
                                          imshow=False,
                                          imsave=True,
                                          conf_thresh=0.2)
    print('{}: {}'.format(i, image_id))
    print('   pred = {}'.format(pred))
    end_time = time.time()
    sec = end_time - start_time
    print('cycle time: {} sec'.format(sec))

    masks = pred['masks']
    polygons = []
    if len(masks) > 0:
        for mask in masks:
            mask = np.transpose(mask, (1,2,0))
            thresh = 0.5
            mask_bin, contours, hierarchy, polygon = Tussock.binarize_confidence_mask(mask, thresh)

            plt.imshow(mask)
            plt.title('confidence mask')
            # plt.show()
            plt.close()

            plt.imshow(mask_bin)
            plt.title('binary mask')
            # plt.show()
            plt.close()

            # plot contours
            drawing = np.zeros((mask_bin.shape[0], mask_bin.shape[1], 3), dtype=np.uint8)
            for j in range(len(contours)):
                colour = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
                cv.drawContours(drawing, contours, j, colour, 2, cv.LINE_8, hierarchy, 0)
            plt.imshow(drawing)
            plt.title('contours')
            # plt.show()
            plt.close()



            
            polygons.append(polygon)

            import code
            code.interact(local=dict(globals(), **locals()))

    # TODO convert contours into polygon (x, y points), save them to the json file


    print('done x_binarize_image.py')

    
    