#! /usr/bin/env python

"""
script to check image files match json file
"""


import os
import json

dataset_dir = os.path.join('/home',
                           'dorian',
                           'Data',
                           'AOS_TussockDataset')

case = 3
if case == 0:
    root_dir = os.path.join(dataset_dir, 'Tussock_210325_G507_location1_positivetags')
    annotations_file = 'Thursday_25-03-21_G507_location1_positive-tags_labels.json'
elif case == 1:
    root_dir = os.path.join(dataset_dir, 'Tussock_210325_G507_location2_positivetags')
    annotations_file = 'Thursday_25-03-21_G507_location2_positive-tags_labels.json'
elif case == 2:
    root_dir = os.path.join(dataset_dir, 'Tussock_210326_G507_location1_positivetags')
    annotations_file = 'Friday_26-03-21_G507_location1_positive-tags_labels.json'
elif case == 3:
    root_dir = os.path.join(dataset_dir, 'Horehound_210326_G507_location2_positivetags')
    annotations_file = 'Friday_26-03-21_G507_location2_positive-tags_labels.json'
else:
    print('Warning: not valid case number')

image_dir = os.path.join(root_dir, 'positive-tags')
unsorted_annotations_dir = os.path.join(root_dir, 'Annotations')

# read in json - raw json file from AOS
annotations = json.load(open(os.path.join(unsorted_annotations_dir,
                                      annotations_file)))
annotations = list(annotations.values())

print('number of images in unsorted annotation file: {}'.format(len(annotations)))

# get list of image names in json file:
img_names = []
for s in annotations:
    img_name = s['filename']
    img_names.append(img_name)

# TEST
# find list of all images in unsorted_image_dir, make sure they match all
# entries in the json file

# get list of files in unsorted_image_dir
files = os.listdir(image_dir)

print('img_names from annotation json file: {}'.format(len(img_names)))
print('files from unsorted_image_dir: {}'.format(len(files)))
intersection_set = set(img_names) & set(files)
print('intersection of the two sets: {}'.format(len(intersection_set)))
print('the above should all be equal!')

# import code
# code.interact(local=dict(globals(), **locals()))

