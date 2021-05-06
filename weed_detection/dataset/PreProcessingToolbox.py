#! /usr/bin/env python

"""
preprocessing class for dataset
converting images
moving images around
synchronizing annotation files
a collection of functions for the above
"""

import os
import json
import pickle
import numpy as np
from PIL import Image


class PreProcessingToolbox:
    """ collection of functions to preprocessing the dataset """

    def __init__(self, image_dir, annotations_file):
        self.image_dir = image_dir
        self.annotations_file = annotations_file


    def check_images_match_annotations(self,
                                       img_dir=None,
                                       ann_file=None):
        """ check if images match the annotations file """
        if img_dir is None:
            img_dir = self.image_dir

        if ann_file is None:
            ann_file = self.annotations_file

        print('ann_file to check: ' + ann_file)
        print('img_dir to check: ' + img_dir)

        # read in json file
        ann = json.load(open(ann_file))
        ann = list(ann.values())

        print('num images in ann file: {}'.format(len(ann)))

        # get list of image names in annotations file
        # img_names = []
        # for s in ann:
        #     img_name = s['filename']
        #     img_names.append(img_name)
        img_names = [s['filename'] for s in ann]

        # get list of files in image_directory
        files = os.listdir(img_dir)

        # find intersection of the two lists (turned into sets)
        intersection_set = set(img_names) & set(files)

        n_img = len(img_names)
        n_files = len(files)
        n_int = len(intersection_set)
        print('n img_names from annotation file: {}'.format(n_img))
        print('n files from img_dir: {}'.format(n_files))

        print('intersection of the two sets: {}'.format(n_int))
        print('the above should all be equal!')

        if (n_img == n_int) and (n_files == n_int):
            return True
        else:
            return False


    def combine_annotations(self, ann_files, ann_dir, ann_out=None):
        """ combine annotation files
        ann_files is a list of annotation files (absolute path) ann_dir =
        annotations file directory, if None then we need absolute filepath or
        assume local otherwise, ann_files relative to ann_dir
        """

        # TODO check valid input
        ann = []
        if ann_dir is None:
            ann_dir = '.'
        #     # absolute filepath for ann_files
        #     for i in range(len(ann_files)):
        #         ann.append(json.load(open(ann_files[i])))
        # else:

        for i in range(len(ann_files)):
            ann.append(json.load(open(os.path.join(ann_dir, ann_files[i]))))

        # for now, assume unique key-value pairs, but should probably check length
        ann_all = {}
        n_img_all = []
        for ann_i in ann:
            ann_all = {**ann_all, **ann_i}
            n_img_all.append(len(ann_i))

        n_img_all = np.array(n_img_all)
        n_img_sum = np.sum(n_img_all)
        # TODO do check
        # len(ann_all) vs sum(len(ann_i))

        with open(os.path.join(ann_dir, ann_out), 'w') as ann_file:
            json.dump(ann_all, ann_file, indent=4)

        n_all = len(ann_all)
        if np_img_sum == n_all:
            return True
        else:
            return False


    def convert_images(self, folder, file_pattern='.png'):
        """ convert all iamges to "filepattern" type in a given folder """

        files = os.listdir(folder)
        for f in files:
            if f.endswith('.jpg') or f.endswith('.tif') or f.endswith('.bmp') or f.endswith('.jpeg'):
                if f.endswith('.jpeg'):
                    backspace = 5
                else:
                    backspace = 4
                # convert to png by opening then saving (might be a more
                # efficient way of doing this)
                img = Image.open(os.path.join(folder, f))
                g = f[:-backspace] + file_pattern
                print('saving as {}/{}'.format(folder, g))
                img.save(os.path.join(folder, g))

        return True


    def find_positive_images(self, folder_in, root_dir, ann_in, ann_out):
        """ find images with positive weed annotations in an image
        folder/annotations_in file pair that have both negative and positive
        images of weeds
        also copies positive images to root_dir/Images
        and negative images to root_dir/Negative_Images
        and outputs a corresponding annotations file for all the positive Images
        """

        # setup directories
        img_dir = os.path.join(root_dir, 'Images')
        neg_img_dir = os.path.join(root_dir, 'Negative_Images')
        ann_dir = os.path.join(root_dir, 'Annotations')

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(neg_img_dir, exist_ok=True)

        # read in annotations
        ann_uns = json.load(open(os.path.join(ann_dir, ann_in)))
