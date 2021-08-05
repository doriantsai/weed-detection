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

def unscale_polygon(polygon, output_size, input_size):
    """ unscale polygon """
    h, w = output_size
    if isinstance(input_size, int):
        if h > w:
            new_h, new_w = input_size * h / w, input_size
        else:
            new_h, new_w = input_size, input_size * w / h
    else:
        new_h, new_w = input_size

    new_h, new_w = int(new_h), int(new_w)
    xChange = float(new_w) / float(w)
    yChange = float(new_h) / float(h)

    p_out = polygon.copy()
    if len(polygon) > 0:
        new_x = np.array(p_out['all_points_x'], np.float32) / xChange
        new_y = np.array(p_out['all_points_y'], np.float32) / yChange
        # p_out['all_points_x'] = list(np.int32(new_x)) # because of pointers/referencing, not sure if this line is needed
        # p_out['all_points_y'] = list(np.int32(new_y))
        p_out['all_points_x'] = [round(new_x[i]) for i in range(len(new_x))] # must be in "int" for json
        p_out['all_points_y'] = [round(new_y[i]) for i in range(len(new_y))]
        

    return p_out


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

# this is the file we want to do the bootstrapping on!
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

rescale_size = 1024
resize_img = Rescale(rescale_size) # specific to what image was originally sized for
conf_thresh = 0.5

original_size = [2056, 2464] # pixels, h, w


# polygons = []
for i, img_name in enumerate(img_names_rem):
    img_path = os.path.join(img_dir, img_name)
    img = Image.open(img_path)

    img = resize_img(img)
    img = tvtransfunc.to_tensor(img)
    img.to(device)

    img_out, pred = boot_model.infer_image(img,
                                        imshow=False,
                                        imsave=False,
                                        conf_thresh=conf_thresh,
                                        annotation_type='poly')
    print(f'image name: {img_name}')
    # print(f'     pred = {pred}')

    masks = pred['masks']
    polygons_per_img = []

    # find index of img_name in img_names_all
    idx = img_names_all.index(img_name)
    if len(masks) > 0:
        for j, mask in enumerate(masks):
            mask = np.transpose(mask, (1,2,0))
            thresh = 0.5
            mask_bin, ctr, hier, poly = boot_model.binarize_confidence_mask(mask, thresh)

            # TODO need to unscale images, create function unscale_polygon(img_orig, img_resc, pred_resc), return pred_out
            # TODO consider Shapeply
            # https://shapely.readthedocs.io/en/stable/manual.html
            # poly = Polygon(contour)
            # poly = poly.simplify(1.0, preserve_topology=False)


            # unscale polygon to original image size
            poly_uns = unscale_polygon(poly, original_size, rescale_size)
            
            # need to add the polygon region/annotation 
            poly_region = {'shape_attributes': poly_uns, 'region_attributes': {}}
            # find out how many regions there already are
            nreg = len(ann_all_list[idx]['regions'])
            # create dictionary with key "nreg", signifying the region number
            # don't have to "add one", because increments from 0
            add_poly_region = {str(nreg): poly_region} 

            ann_all_list[idx]['regions'].update(add_poly_region)

            polygons_per_img.append(poly)

    # polygons.append(polygons_per_img)
    # put x_binarize_image.py into a function, call it here on the predicted mask

    
    
    # if i >= 1:
    #     print('hit max number of images for testing purposes')
    #     break

# now we have a list of names
# and a list of corresponding polygons
# for each name, we want to add this to ann

# for each name in list, do model inference
# save annotated image
# save annotation/remake dictionary
# get all keys, values from dict into lists (need to recreate dict later on)
# TODO could replace this for-loop with a list comprehension
keys = []
values = []
for k in ann_all_dict:
    keys.append(k)
    values.append(ann_all_dict[k])

ann_out = {}
for i, k in enumerate(keys):
    ann_out = {**ann_out, k: ann_all_list[i]}

import code
code.interact(local=dict(globals(), **locals()))



# ann_poly_out = 'via_project_07Jul2021_08h00m_240_test_bootstrap.json'
# ann_poly_out_path = os.path.join(ann_dir, ann_poly_out)
with open(ann_poly_out_path, 'w') as af:
    json.dump(ann_out, af, indent=4)




# TODO need to unscale images, create function unscale_polygon(img_orig, img_resc, pred_resc), return pred_out
# TODO convert unscaled polygon to mask, based on some threshold
# NOTE: probably more efficient to scale polygon and then just convert polygon to mask at appropriate size
# but for now, we'll just resize the output mask and threshold it
# def scale_polygon(poly, current_size, output_size):
#     # current_size should be height, width tuple
#     # h, w = image.size[:2]
#     h, w = current_size
#     if isinstance(output_size, int):
#         if h > w: 
#             new_h, new_w = output_size * h / w, output_size
#         else:
#             new_h, new_w = output_size, output_size * w / h
#     else:
#         new_h, new_w = output_size
    
#     new_h, new_w = int(new_h), int(new_w)

#     # scale polygon
#     p_x = poly[:, 0]
#     p_y = poly[:, 1]


import code
code.interact(local=dict(globals(), **locals()))

