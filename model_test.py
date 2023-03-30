#! /usr/bin/env python3

"""
Example script on how to use the detector
Run detector. For a given image folder with annotations, show the detections
overlayed with original annotations
"""

import os
from weed_detection.TestModel import TestModel

annotation_data = {'annotation_file': '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_clarkefield31/dataset.json',
                   'image_dir': '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_clarkefield31/annotated_images',
                   'mask_dir': '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_clarkefield31/masks'}

output_dir = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_clarkefield31/images_test'

model_file = os.path.join('/home/agkelpie/Code/agkelpie_weed_detection/weed-detection/model/Clarkefield31/model_best.pth')

confidence_thresh = 0.5
nms_thresh = 0.5
species_file = os.path.join(os.getcwd(), 'model/names_clarkefield31.txt')

Test = TestModel(model_file = model_file,
                 annotation_data = annotation_data,
                 output_dir = output_dir,
                 confidence_threshold = confidence_thresh,
                 nms_threshold = nms_thresh,
                 names_file = species_file)

img_list = os.listdir(annotation_data['image_dir'])
    
max_img = 5
for i, img_name in enumerate(img_list):
    if i > max_img:
        print(f'reached max images: {max_img}')
        break
    print(f'{i}: {os.path.basename(img_name)}')
    Test.show_image_test(os.path.join(annotation_data['image_dir'], img_name), POLY=True)

print(f'saved images in {Test.output_dir}')
# done
