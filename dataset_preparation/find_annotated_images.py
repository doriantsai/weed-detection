#! /usr/bin/env python

"""
code to identify the images annotated with serrated tussock from the "positive tags"
of images
"""

# TODO
# set directories
#   Images
#   root_dir
#   location of Annotations file
#   location of unsorted images
# read in all images, read in json
# find all images that have bounding boxes in them
# save those images
# copy them over to a new folder - Images
# save the relevant json into the Annotations folder

import os
import json
import shutil

dataset_dir = os.path.join('/home',
                           'dorian',
                           'Data',
                           'AOS_TussockDataset')

# annotation file/image folder pairs
case = 3
if case == 0:
    root_dir = os.path.join(dataset_dir, 'Tussock_210325_G507_location1_positivetags')
    annotations_file_in = 'Thursday_25-03-21_G507_location1_positive-tags_labels.json'
    annotations_file_out = 'annotations_tussock_210325_G507_location1.json'
elif case == 1:
    root_dir = os.path.join(dataset_dir, 'Tussock_210325_G507_location2_positivetags')
    annotations_file_in = 'Thursday_25-03-21_G507_location2_positive-tags_labels.json'
    annotations_file_out = 'annotations_tussock_210325_G507_location2.json'
elif case == 2:
    root_dir = os.path.join(dataset_dir, 'Tussock_210326_G507_location1_positivetags')
    annotations_file_in = 'Friday_26-03-21_G507_location1_positive-tags_labels.json'
    annotations_file_out = 'annotations_tussock_210326_G507_location1.json'
elif case == 3:
    root_dir = os.path.join(dataset_dir, 'Horehound_210326_G507_location2_positivetags')
    annotations_file_in = 'Friday_26-03-21_G507_location2_positive-tags_labels.json'
    annotations_file_out = 'annotations_horehound_210326_G507_location2.json'
else:
    print('Warning: not valid case number')

unsorted_image_dir = os.path.join(root_dir, 'positive-tags')
annotations_dir = os.path.join(root_dir, 'Annotations')
image_dir = os.path.join(root_dir, 'Images')
if not os.path.isdir(image_dir):
    print('image_dir does not yet exist. making image_dir')
    os.mkdir(image_dir)

# read in json
annotations_uns_dict = json.load(open(os.path.join(annotations_dir, annotations_file_in)))
annotations_uns = list(annotations_uns_dict.values())

# sort through json file, find all images with non-zero regions
positive_img_list = []
ann_dict = {}
for i, sample in enumerate(annotations_uns):

    img_name = annotations_uns[i]['filename']
    nregions = len(annotations_uns[i]['regions'])

    if nregions > 0:
        # if we have any bounding boxes, we want to save this image name in a list
        positive_img_list.append(img_name)

        # immediately coopy image over to image_dir
        shutil.copyfile(os.path.join(unsorted_image_dir, img_name),
                        os.path.join(image_dir, img_name))

        # save sample info to annotations_file_out
        # I think I need to remake the dictionary
        file_ref = sample['fileref']
        file_size = sample['size']
        file_name = sample['filename']
        imgdata = sample['base64_img_data']
        file_att = sample['file_attributes']
        regions = sample['regions']

        ann_dict[file_name + str(file_size)] = {
            'fileref': file_ref,
            'size': file_size,
            'filename': file_name,
            'base64_img_data': imgdata,
            'file_attributes': file_att,
            'regions': regions
        }

print('image copy to image_dir complete')

# save filtered annotation_file
with open(os.path.join(annotations_dir, annotations_file_out), 'w') as ann_file:
    json.dump(ann_dict, ann_file, indent=4)

print('filtered annotation file complete')

# for f in positive_img_list:
#     print(f)


# TODO check:
# read number of images in images folder
# compare them to original number of nonzero json entries in the unsorted annotations folder
print('number of images with tussock in them from original annotations file: {}'.format(len(positive_img_list)))
image_files = os.listdir(image_dir)
print('number of images in image_dir: {}'.format(len(image_files)))
print('the two above numbers should match')

import code
code.interact(local=dict(globals(), **locals()))


# files = os.listdir(unsorted_image_dir)
# i = 0
# for f in files:
#     # check file extension
#     if f.endswith('.png'):
#         # copy/move image into root_dir/Images folder

#         i +=1


