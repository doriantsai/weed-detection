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
from posix import ST_SYNCHRONOUS
import numpy as np
from PIL import Image
import shutil
from subprocess import call
from weed_detection.WeedDataset import WeedDataset
import torch


class PreProcessingToolbox:
    """ collection of functions to preprocessing the dataset """

    def __init__(self, image_dir=None, annotations_file=None):
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
        if n_img_sum == n_all:
            return True
        else:
            return False


    def sync_annotations(self, image_dir, ann_master_file, ann_out_file):
        """ synchronise annotations file (ann_out) based on what images are
        in image_dir and ann_master """

        # read in annotations master
        # create list out of dictionary values, so we have indexing
        ann_master = json.load(open(ann_master_file))
        ann_master = list(ann_master)

        # find all files in img_dir
        img_list = os.listdir(image_dir)

        # find all dictionary entries that match img_dir
        master_filename_list = [s['filename'] for s in ann_master]

        # create dictionary with keys: list entries, values: indices
        ind_dict = dict((k, i) for i, k in enumerate(master_filename_list))

        # find intersection of master and local ann files:
        inter = set(ind_dict).intersection(img_list)

        # compile list of indices of the intersection
        indices = [ind_dict[x] for x in inter]

        # for each index, we take the sample from ann_master and make a new dict
        ann_dict = {}
        for i in indices:
            sample = ann_master[i]

            # save/create new annotations file
            ann_dict = self.sample_dict(ann_dict, sample)

        # create annotations_out_file:
        with open(ann_out_file, 'w') as ann_file:
            json.dump(ann_dict, ann_file, indent=4)

        return True


    def sample_dict(self, ann_dict, sample):
        """ helper function to build annotation dictionary """
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
        # return ann_dict might not be necessary due to pointers
        return ann_dict


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
        ann_uns = list(ann_uns.values())

        # sort through unsorted annotations and find all images with nonzero regions
        pos_img_list = []
        ann_dict = {}
        for i, sample in enumerate(ann_uns):

            img_name = ann_uns[i]['filename']
            n_regions = len(ann_uns[i]['regions'])

            if n_regions > 0:
                # if we have any bounding boxes, we want to save this image name in a list
                pos_img_list.append(img_name)

                # copy over image to img_dir
                shutil.copyfile(os.path.join(folder_in, img_name),
                                os.path.join(img_dir, img_name))

                # save annotations by remaking the dictionary
                # file_ref = sample['fileref']
                # file_size = sample['size']
                # file_name = sample['filename']
                # img_data = sample['base54_img_data']
                # file_att = sample['file_attributes']
                # regions = sample['regions']

                # ann_dict[file_name + str(file_size)] = {
                #     'fileref': file_ref,
                #     'size': file_size,
                #     'filename': file_name,
                #     'base64_img_data': img_data,
                #     'file_attributes': file_att,
                #     'regions': regions
                # }
                ann_dict = self.sample_dict(ann_dict, sample)

            else:
                # copy negative images
                shutil.copyfile(os.path.join(folder_in, img_name),
                                os.path.join(neg_img_dir, img_name))

        print('image copy to image_dir complete')
        # save annotation file
        with open(os.path.join(ann_dir, ann_out), 'w') as ann_file:
            json.dump(ann_dict, ann_file, indent=4)

        print('filtered annotation file saved')

        # check:
        n_in = len(ann_uns)
        print('n img in original ann_file: {}'.format(n_in))
        n_out = len(ann_dict)
        print('n img positive in out ann_file: {}'.format(n_out))

        return True


    def rename_images(self, img_dir, str_pattern='.png'):
        """ rename all files that match str_pattern in a given folder """

        files = os.listdir(img_dir)
        for i, f in enumerate(files):

            # check file extension
            if f.endswith(str_pattern):
                # rename to st-###.png
                img_name = 'st' + str(i).zfill(3) + str_pattern
                print(f + ' --> ' + img_name)
                os.rename(os.path.join(img_dir, f),
                          os.path.join(img_dir, img_name))

        return True


    def copy_images(self, dataset, all_folder, save_folder):
        """ copy images specified in dataset from all_folder to save_folder """
        for image, sample in dataset:
            image_id = sample['image_id'].item()
            img_name = dataset.dataset.annotations[image_id]['filename']
            new_img_path = os.path.join(save_folder, img_name)
            old_img_path = os.path.join(all_folder, img_name)
            # print('copy from: {}'.format(old_img_path))
            # print('       to: {}'.format(new_img_path))
            shutil.copyfile(old_img_path, new_img_path)


    def split_image_data(self,
                         root_dir,
                         all_folder,
                         ann_master_file,
                         ann_all_file,
                         ann_train_file,
                         ann_val_file,
                         ann_test_file,
                         ratio_train_test=None):
        """ prepare dataset/dataloader objects by randomly taking images from all_folder,
        and splitting them randomly into Train/Test/Val with respecctive annotation files
        """

        # setup folders
        train_folder = os.path.join(root_dir, 'Images', 'Train')
        test_folder = os.path.join(root_dir, 'Images','Test')
        val_folder = os.path.join(root_dir, 'Images', 'Validation')

        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)

        ann_dir = os.path.join(root_dir, 'Annotations')
        # NOTE I don't think I'm entirely consistent with use of annotation file names
        # TODO check for consistency
        ann_master = os.path.join(ann_dir, ann_master_file)
        ann_all = os.path.join(ann_dir, ann_all_file)

        # ensure annotations file matches all images in all_folder
        # a requirement for doing random_split
        self.sync_annotations(all_folder, ann_master, ann_all)

        # create dummy weed dataset object to do random split
        wd = WeedDataset(all_folder, ann_all, transforms=None)

        # dataset lengths
        files = os.listdir(all_folder)
        # img_files = [f for f in files if f.endswith('.png')]  # I think this works
        # TODO check
        img_files = []
        for f in files:
            if f.endswith('.png'):
                img_files.append(f)
        n_img = len(img_files)
        print('number of images in all_folder: {}'.format(n_img))

        # define ratio for training and testing data
        if ratio_train_test is None:
            ratio_train_test = [0.7, 0.2]
        # compute validation ratio from remainder
        ratio_train_test.append(1 - ratio_train_test[0] - ratio_train_test[1])

        tr = int(round(n_img * ratio_train_test[0]))
        te = int(round(n_img * ratio_train_test[1]))
        va = int(round(n_img * ratio_train_test[2]))

        print('n_train {}'.format(tr))
        print('n_test {}'.format(te))
        print('n_val {}'.format(va))

        # do random split of image data
        ds_train, ds_val, ds_test = torch.utils.data.random_split(wd, [tr, va, te])

        # now, actually copy images from All folder to respective image folders
        # dataset = ds_train
        # save_folder = train_folder
        self.copy_images(ds_train, all_folder, train_folder)
        self.copy_images(ds_val, all_folder, val_folder)
        self.copy_images(ds_test, all_folder, test_folder)
        print('copy images from all_folder to train/test/val_folder complete')

        # now, call sync_annotations_with_imagefolder for each:
        annotations_train = os.path.join(ann_dir, ann_train_file)
        annotations_val = os.path.join(ann_dir, ann_val_file)
        annotations_test = os.path.join(ann_dir, ann_test_file)

        # function calls for each folder
        self.sync_annotations(train_folder, ann_all, annotations_train)
        self.sync_annotations(val_folder, ann_all, annotations_val)
        self.sync_annotations(test_folder, ann_all, annotations_test)
        print('sync json with image folders complete')

        # package output
        img_folders = [train_folder, test_folder, val_folder]
        ann_files = [annotations_train, annotations_test, annotations_val]
        return img_folders, ann_files