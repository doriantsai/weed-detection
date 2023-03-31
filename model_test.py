#! /usr/bin/env python3

"""
Example script on how to use the detector
Run detector. For a given image folder with annotations, show the detections
overlayed with original annotations
"""

import os
from weed_detection.TestModel import TestModel


# root_dir of dataset
root_dir = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_clarkefield31'
# root_dir = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_yellanglo29'
# root_dir = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_yellanglo32'

# code_base dir
code_dir = '/home/agkelpie/Code/agkelpie_weed_detection/weed-detection'

# annotation file, correesponding image directory and mask directory
annotation_data = {'annotation_file': os.path.join(root_dir, 'dataset.json'),
                    'image_dir': os.path.join(root_dir, 'annotated_images'),
                    'mask_dir': os.path.join(root_dir, 'masks')}

# output directory of saved images with predictions and annotations overlayed
output_dir = os.path.join(root_dir, 'images_test')

# model file of trained network weights
model_file = os.path.join(code_dir, 'model/Clarkefield31/model_best.pth')

# thresholds for predictions/detections
confidence_thresh = 0.5
nms_thresh = 0.5

# names of possible species that the model has been trained for/are in the dataset
species_file = os.path.join(code_dir, 'model/names_clarkefield31.txt')

# text file that lists the images reserved for testing (that the model hasn't seen yet)
# imagelist_file = os.path.join(root_dir, 'metadata/train.txt')
# imagelist_file = os.path.join(root_dir, 'metadata/val.txt')
imagelist_file = os.path.join(root_dir, 'metadata/test.txt')

# initiate TestModel class
Test = TestModel(model_file = model_file,
                 annotation_data = annotation_data,
                 output_dir = output_dir,
                 confidence_threshold = confidence_thresh,
                 nms_threshold = nms_thresh,
                 names_file = species_file)

# grab list of images from image_dir
# img_list = sorted(os.listdir(annotation_data['image_dir']))

# grab list of images from imagelist_file (reach line and remove trailing /n)
with open(imagelist_file, 'r') as f:
    imgs_str = f.readlines()
img_list = [img.strip() for img in imgs_str]

# specify max num images for debugging
max_img = len(img_list)
for i, img_name in enumerate(img_list):
    if i > max_img:
        print(f'reached max images: {max_img}')
        break
    print(f'{i}: {os.path.basename(img_name)}')
    # plot/save image detections and annotations overlayed on original image
    Test.show_image_test(os.path.join(annotation_data['image_dir'], img_name), POLY=True)

print(f'saved images in {Test.output_dir}')
# done
