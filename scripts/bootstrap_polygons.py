#! /usr/bin/env python

"""
bootstrap data labelling process for maskrcnn
on round1 data
given: pre-trained model on subset of data+annotations
label the rest of the data in a given folder
"""

import os
import json
import numpy as np
from weed_detection.WeedDatasetPoly import Rescale
from torchvision.transforms import functional as tvtransfunc
from PIL import Image
import torch

from weed_detection.PreProcessingToolbox import PreProcessingToolbox as PPT
from weed_detection.WeedDatasetPoly import WeedDatasetPoly as WDP
from weed_detection.WeedModel import WeedModel as WM

# establish files/folders
# folder/path of dataset, dataset name, image directory, annotation directory
# annotation file with existing polygon annotations
# annotation file with all image annotaions (eg, master)
# get list of all images with polygon annotations
# get list of all images without polygon annotations (set exclusion thing)
# import model from given folder/file
# run model inference on set of all images without polygon annotations
# convert mask rcnn output (mask from 0-1) into binary mask,
# export

# dataset name:
dataset_name = 'Tussock_v4_poly286'
dataset_path = os.path.join('/home', 'dorian', 'Data', 'AOS_TussockDataset', dataset_name)
img_dir = os.path.join(dataset_path, 'Images', 'All_Unlabelled_v2')
ann_dir = os.path.join(dataset_path, 'Annotations')

# all annotations (positive images only):
ann_all = 'annotations_tussock_21032526_G507_all.json'
ann_all_path = os.path.join(ann_dir, ann_all)

# current annotations file with polygons:
ann_poly = 'via_project_07Jul2021_08h00m_240_test_allpoly.json'
ann_poly_path = os.path.join(ann_dir, ann_poly)

# annotation out file with all the predictions:
ann_poly_out = 'via_project_07Jul2021_08h00m_240_test_bootstrap.json'
ann_poly_out_path = os.path.join(ann_dir, ann_poly_out)


# load annotations
ann_poly_dict = json.load(open(ann_poly_path))
ann_poly_list = list(ann_poly_dict.values())

ann_all_dict = json.load(open(ann_all_path))
ann_all_list = list(ann_all_dict.values())

print(f'num images in ann_poly_list: {len(ann_poly_list)}')
print('should be 286')
print(f'num images in ann_all_list: {len(ann_all_list)}')
print('should be 570')

# take these lists of dictionaries, and just get list of file names?
img_names_poly = [s['filename'] for s in ann_poly_list]
img_names_all = [s['filename'] for s in ann_all_list]
# find set of images in ann_all_list not already in ann_poly_list

img_names_rem = np.setdiff1d(img_names_all, img_names_poly)
# rem = list(set(all) - set(part)) # note: unordered
print(f'num images in img_names_rem {len(img_names_rem)}')
print(f'should be 570 - 286 = {570-286}')



# first, check all filenames are contained within the names of all images
# in the image folder
img_dir_names = os.listdir(img_dir)

int_set = set(img_dir_names) & set(img_names_rem)
print(f'intersecting set: should match img_names_rem: {len(int_set)}')



# now I need to use these file names to do model inference on them

# setup device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# load model
model_name = 'Tussock_v4_poly286_2021-07-15_11_08'
model_folder = os.path.join( model_name)
model_path = os.path.join(model_folder, model_name + '.pth')
boot_model = WM(model_path=model_path, device=device)
boot_model.set_model_name(model_name)
boot_model.set_model_folder(model_folder)
boot_model.set_snapshot(20)


resize_img = Rescale(1024) # specific to what image was originally sized for
conf_thresh = 0.5

for i, img_name in enumerate(img_names_rem):
    img_path = os.path.join(img_dir, img_name)
    img = Image.open(img_path)

    img = resize_img(img)
    img = tvtransfunc.to_tensor(img)
    img.to(device)

    pred, keep = boot_model.infer_image(img,
                                        imshow=True,
                                        imsave=True,
                                        conf_thresh=conf_thresh,
                                        annotation_type='poly')
    print(f'image name: {img_name}')
    print(f'     pred = {pred}')

    if i >= 1:
        print('hit max number of images for testing purposes')
        break



# for each name in list, do model inference
# save annotated image
# save annotation/remake dictionary

# TODO need to unscale images, create function unscale_polygon(img_orig, img_resc, pred_resc), return pred_out

import code
code.interact(local=dict(globals(), **locals()))

