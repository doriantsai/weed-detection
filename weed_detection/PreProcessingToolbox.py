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
from weed_detection.WeedDatasetPoly import WeedDatasetPoly
import cv2 as cv
from posix import ST_SYNCHRONOUS
import numpy as np
from PIL import Image
import shutil
from subprocess import call
# from torch._C import namedtuple_solution_cloned_coefficient
from weed_detection.WeedDataset import WeedDataset, Compose, \
    RandomBlur, RandomVerticalFlip, RandomHorizontalFlip, \
    RandomBrightness, RandomContrast, ToTensor
import torch
import random
import glob

import matplotlib.pyplot as plt # somewhat redundany w cv, but provides better image/plot analysis/gui tools
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


class PreProcessingToolbox:
    """ collection of functions to preprocessing the dataset """

    def __init__(self, image_dir=None, annotations_file=None):
        self.image_dir = image_dir
        self.annotations_file = annotations_file


    def check_images_match_annotations(self,
                                       img_dir=None,
                                       ann_file=None):
        """ check if images match the annotations file """
        # NOTE: depracated due to is_processed tag in images, should compare to count of images with is_processed == 1

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
        ann_list = []
        if ann_dir is None:
            ann_dir = '.'

        for ann in ann_files:

            if ann_dir is False:
                # absolute pathing
                ann_open = ann
            else:
                ann_open = os.path.join(ann_dir, ann)

            ann_list.append(json.load(open(ann_open)))

        

        # for now, assume unique key-value pairs, but should probably check length
        ann_all = {}
        n_img_all = []
        for ann_i in ann_list:
            # handle via project file (rather than just pure annotations file)
            if '_via_settings' in ann_i:
                # only grab the via img metadata
                ann_i = ann_i['_via_img_metadata']
            ann_all = {**ann_all, **ann_i}
            n_img_all.append(len(ann_i))

        n_img_all = np.array(n_img_all)
        n_img_sum = np.sum(n_img_all)
        # TODO do check
        # len(ann_all) vs sum(len(ann_i))

        if ann_dir is False:
            self.make_annfile_from_dict(ann_all, ann_out)
        else:
            self.make_annfile_from_dict(ann_all, os.path.join(ann_dir, ann_out))

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
        # access img metadata
        # NOTE: previous datasets only saved the img_metadata
        if '_via_settings' in ann_master:
            # only grab the via img metadata
            ann_master = ann_master['_via_img_metadata']
        # ann_master = ann_master["_via_img_metadata"] 
        ann_master = list(ann_master.values())

        # find all files in img_dir
        img_list = os.listdir(image_dir)

        # find all dictionary entries that match img_dir
        # master_filename_list = [s['filename'] for s in ann_master ]
        # update: filter annotation list for images that have "1" for is_processed flag
        # NOTE already happens in filter_processed images, so should be unnecessary
        master_filename_list = [s['filename'] for s in ann_master if s['file_attributes']['is_processed'] == str(1) ]

        # import code
        # code.interact(local=dict(globals(), **locals()))

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
        self.make_annfile_from_dict(ann_dict, ann_out_file)

        return ann_out_file


    def sample_dict(self, ann_dict, sample):
        """ helper function to build annotation dictionary """
        # file_ref = sample['fileref']
        file_size = sample['size']
        file_name = sample['filename']
        # imgdata = sample['base64_img_data']
        file_att = sample['file_attributes']
        regions = sample['regions']

        ann_dict[file_name + str(file_size)] = {
            # 'fileref': file_ref,
            'size': file_size,
            'filename': file_name,
            # 'base64_img_data': imgdata,
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


    def split_pos_neg_images(self, folder_in, root_dir, ann_in, ann_out):
        """ find images with positive weed annotations in an image
        folder/annotations_in file pair that have both negative and positive
        images of weeds
        also copies positive images to root_dir/Images
        and negative images to root_dir/Negative_Images
        and outputs a corresponding annotations file for all the positive Images
        """

        # setup directories
        img_dir = os.path.join(root_dir, 'images')
        neg_img_dir = os.path.join(root_dir, 'images_neg')
        ann_dir = os.path.join(root_dir, 'metadata')

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
                ann_dict = self.sample_dict(ann_dict, sample)

            else:
                # copy negative images
                shutil.copyfile(os.path.join(folder_in, img_name),
                                os.path.join(neg_img_dir, img_name))

        print('image copy to image_dir complete')
        # save annotation file
        self.make_annfile_from_dict(ann_dict, os.path.join(ann_dir, ann_out))

        print('filtered annotation file saved')

        # check:
        n_in = len(ann_uns)
        print('n img in original ann_file: {}'.format(n_in))
        n_out = len(ann_dict)
        print('n img positive in out ann_file: {}'.format(n_out))

        return True

    def count_pos_neg_images(self, ann_list):
        """ count positive/negative images in json file, returns tuple (# pos, # neg) """
        """ also can split pos/neg into dictionaries, return those dictionaries """

        # ann_dict = json.load(open(ann_path))
        # ann_list = list(ann_dict.values())

        idx_pos = []
        idx_neg = []
        ann_pos = {}
        ann_neg = {}
        for i, ann in enumerate(ann_list):
            reg = ann['regions']
            # if regions is not empty, we have a positive image, otherwise, negative image
            # NOTE: this applies for any classes, does not ensure balanced classes
            if bool(reg):
                idx_pos.append(i)
            else:
                idx_neg.append(i)

        print(f'pos images: {len(idx_pos)}')
        print(f'neg images: {len(idx_neg)}')
        # print('neg images: ' + len(idx_neg))
        return (idx_pos, idx_neg)



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


    def copy_images(self, dataset, all_folder, save_folder, mask=False, symlink=True):
        """ copy images specified in dataset from all_folder to save_folder """
        for image, sample in dataset:
            image_id = sample['image_id'].item()
            if mask:
                img_name = dataset.dataset.annotations[image_id]['filename'][:-4] + '_mask.png'
            else:
                img_name = dataset.dataset.annotations[image_id]['filename']

            new_img_path = os.path.join(save_folder, img_name)
            old_img_path = os.path.join(all_folder, img_name)

            # if mask:
            #     print('copy from: {}'.format(old_img_path))
            #     print('       to: {}'.format(new_img_path))
            #     import code
            #     code.interact(local=dict(globals(), **locals()))
            if symlink:
                if os.path.exists(new_img_path):
                    os.unlink(new_img_path)
                os.symlink(old_img_path, new_img_path)
            else:
                # TODO possible solution to faster copyfiles
                # https://stackoverflow.com/questions/22078621/how-to-copy-files-fast
                shutil.copyfile(old_img_path, new_img_path)


    def split_image_data(self,
                         root_dir,
                         all_folder,
                         ann_master_file,
                         ann_all_file,
                         ann_train_file,
                         ann_val_file,
                         ann_test_file,
                         ratio_train_test=[0.7, 0.2],
                         clear_image_folders=True,
                         annotation_type='poly',
                         mask_folder=None,
                         ann_dir=None):
        """ prepare dataset/dataloader objects by randomly taking images from all_folder,
        and splitting them randomly into Train/Test/Val with respecctive annotation files
        """
        # TODO should just return generator as object, so can reproduce with just generator, rather than splitting/making new data

        # TODO make sub-function to iterate for one type of image folder/ann_file
        # TODO then call sub-function 3x for each ann_file

        # setup folders
        train_folder = os.path.join(root_dir, 'images_train')
        test_folder = os.path.join(root_dir, 'images_test')
        val_folder = os.path.join(root_dir, 'images_validation')

        if clear_image_folders:
            if os.path.isdir(train_folder):
                print(f'WARNING: removing all files in folder {train_folder}')
                shutil.rmtree(train_folder)
            if os.path.isdir(test_folder):
                print(f'WARNING: removing all files in folder {test_folder}')
                shutil.rmtree(test_folder)
            if os.path.isdir(val_folder):
                print(f'WARNING: removing all files in folder {val_folder}')
                shutil.rmtree(val_folder)

        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)

        if mask_folder is None:
            mask_folder = os.path.join(root_dir, 'masks')

        if annotation_type == 'poly':
            print('making mask folders')
            mask_train_folder = os.path.join(root_dir, 'masks_train')
            mask_test_folder = os.path.join(root_dir, 'masks_test')
            mask_val_folder = os.path.join(root_dir, 'masks_validation')

            if clear_image_folders:
                if os.path.isdir(mask_train_folder):
                    print(f'WARNING: removing all files in folder {mask_train_folder}')
                    shutil.rmtree(mask_train_folder)
                if os.path.isdir(mask_test_folder):
                    print(f'WARNING: removing all files in folder {mask_test_folder}')
                    shutil.rmtree(mask_test_folder)
                if os.path.isdir(mask_val_folder):
                    print(f'WARNING: removing all files in folder {mask_val_folder}')
                    shutil.rmtree(mask_val_folder)

            os.makedirs(mask_train_folder, exist_ok=True)
            os.makedirs(mask_test_folder, exist_ok=True)
            os.makedirs(mask_val_folder, exist_ok=True)


        # already in train/test/val folders
        if ann_dir is None:
            ann_dir = os.path.join(root_dir, 'metadata')
        # NOTE I don't think I'm entirely consistent with use of annotation file names
        # TODO check for consistency
        if ann_dir is False:
            ann_master = os.path.join(ann_master_file)
            ann_all = os.path.join(ann_all_file)
        else:
            ann_master = os.path.join(ann_dir, ann_master_file)
            ann_all = os.path.join(ann_dir, ann_all_file)

        # ensure annotations file matches all images in all_folder
        # a requirement for doing random_split
        print('syncing annotations file with master file')
        self.sync_annotations(all_folder, ann_master, ann_all)

        # import code
        # code.interact(local=dict(globals(), **locals()))

        # create dummy weed dataset object to do random split
        if annotation_type == 'poly':
            wd = WeedDatasetPoly(root_dir, ann_all, transforms=None, mask_dir=mask_folder)
        else:
            # bounding boxes
            wd = WeedDataset(root_dir, ann_all, transforms=None)

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
        # if ratio_train_test is None:
        #     ratio_train_test = [0.7, 0.2]
        # compute validation ratio from remainder
        ratio_train_test.append(1 - ratio_train_test[0] - ratio_train_test[1])

        tr = int(round(n_img * ratio_train_test[0]))
        te = int(round(n_img * ratio_train_test[1]))
        va = n_img - tr - te

        print('n_train {}'.format(tr))
        print('n_test {}'.format(te))
        print('n_val {}'.format(va))

        # import code
        # code.interact(local=dict(globals(), **locals()))

        # do random split of image data
        ds_train, ds_val, ds_test = torch.utils.data.random_split(wd, [tr, va, te])

        # print('ds_train {}'.format(len(ds_train)))
        # print('ds_test {}'.format(len(ds_test)))
        # print('ds_val {}'.format(len(ds_val)))

        # now, actually copy images from All folder to respective image folders
        # dataset = ds_train
        # save_folder = train_folder
        print('copying images from all to train/test/val...')
        self.copy_images(ds_train, all_folder, train_folder)
        self.copy_images(ds_val, all_folder, val_folder)
        self.copy_images(ds_test, all_folder, test_folder)
        print('copy images from all_folder to train/test/val_folder complete')

        # now, call sync_annotations_with_imagefolder for each:

        if not ann_dir is False:
            ann_train_file = os.path.join(ann_dir, ann_train_file)
            ann_val_file = os.path.join(ann_dir, ann_val_file)
            ann_test_file = os.path.join(ann_dir, ann_test_file)

        # function calls for each folder
        print('syncing json files with image folders...')
        self.sync_annotations(train_folder, ann_all, ann_train_file)
        self.sync_annotations(val_folder, ann_all, ann_val_file)
        self.sync_annotations(test_folder, ann_all, ann_test_file)
        print('sync json with image folders complete')

        # copy masks to corresponding folders as well!
        if annotation_type == 'poly':
            print('copying image masks')
            self.copy_images(ds_train, mask_folder, mask_train_folder, mask=True)
            self.copy_images(ds_val, mask_folder, mask_val_folder, mask=True)
            self.copy_images(ds_test, mask_folder, mask_test_folder, mask=True)

        # package output
        img_folders = [train_folder, test_folder, val_folder]
        ann_files = [ann_train_file, ann_test_file, ann_val_file]
        mask_folders = []
        if annotation_type == 'poly':
            mask_folders = [mask_train_folder, mask_test_folder, mask_val_folder]
        # TODO check, as in split_image_data.py

        return img_folders, ann_files, mask_folders


    def augment_training_data(self,
                              root_dir,
                              img_dir,
                              ann_in,
                              tform_select,
                              tform_param=None,
                              ann_out=None,
                              ann_transform=None,
                              ann_append=False,
                              imshow=False,
                              rm_folder=False):

        # given image_folder and corresponding annotations file
        # choose a transform with certain parameters
        # apply transform to all images in image_folder
        # save each transformed iamge with a unique name
        # into image folder
        # return with updated annotations file

        # assume image_folder is synced with annotations file in
        ann_dir = os.path.join(root_dir, 'metadata')
        if ann_transform is None:
            # ann_transform = os.path.join(ann_dir, 'annotations_transform.json')
            ann_transform = 'annotations_transform.json'

        ann_transform_path = os.path.join(ann_dir,ann_transform)
        # select transform
        prob = 1.0
        # probability of transform happening


        if tform_select == 0:
            tform = Compose([RandomVerticalFlip(prob), ToTensor()])
            name_suffix = 'vertFlip'

        elif tform_select == 1:
            tform = Compose([RandomHorizontalFlip(prob), ToTensor()])
            name_suffix = 'horzFlip'
        elif tform_select == 2:
            # random kernel size
            # create an array of reasonable kernel sizes, and randint an index
            k_size = [5, 7, 9, 11, 13, 15, 17]
            idx = random.randint(0, len(k_size) - 1)
            # kernel_size = 5; # odd number from 3 - 21
            # sigma = () * sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1 + 0.8)
            tform = Compose([RandomBlur(kernel_size=k_size[idx]), ToTensor()])
            name_suffix = 'blur'
        elif tform_select == 3:
            tform = Compose([RandomBrightness(prob), ToTensor()])
            name_suffix = 'bright'
        elif tform_select == 4:
            tform = Compose([RandomContrast(prob), ToTensor()])
            name_suffix = 'contrast'
        elif tform_select == 5:
            tform = Compose([RandomHue(prob), ToTensor()])
            name_suffix = 'hue'
        elif tform_select == 6:
            tform = Compose([RandomSaturation(prob), ToTensor()])
            name_suffix = 'saturatation'
        else:
            # nothing/no transform
            tform = Compose([ToTensor()])
            name_suffix = 'none'

        print(f'transform selected: {tform_select} = {name_suffix}')

        # normal dataset no tform (for comparison
        # dataset = WeedDataset(root_dir=root_dir,
        #                        json_file=ann_in,
        #                        transforms=Compose([ToTensor()]))
        # create weed dataset object with appropriate transform
        dataset = WeedDataset(root_dir=root_dir,
                               json_file=ann_in,
                               transforms=tform,
                               img_dir=img_dir)

        save_folder = os.path.join(root_dir, 'images_augmented')
        if rm_folder and os.path.isdir(save_folder):
            # remove all files in folder
            print(f'WARNING: removing all files in folder {save_folder}')
            shutil.rmtree(save_folder)
        os.makedirs(save_folder, exist_ok=True)


        region_definition = ['__background__', 'Tussock']

        # for each image in dataset, apply transform, save
        # note: we are not using a model
        image_path_list = []
        image_name_list = []
        sample_list = []
        ann_dict = {}
        for image, sample in dataset:

            # transform should be automatically applied, so should
            # just be able to save the image
            image_out = image.numpy()
            image_out = np.transpose(image_out, (1,2,0))
            image_out = cv.normalize(image_out, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

            # image name + transform name
            image_id = sample['image_id'].item()
            image_name = dataset.annotations[image_id]['filename'][:-4]
            image_name = image_name + '_' + name_suffix

            save_image_path = os.path.join(save_folder, image_name + '.png')
            image_out = cv.cvtColor(image_out, cv.COLOR_RGB2BGR)
            cv.imwrite(save_image_path, image_out)

            if imshow:
                print('TODO show image')

            # save name and sample
            image_path_list.append(save_image_path)
            image_name_list.append(image_name)
            sample_list.append(sample)

            # now create new annotations for dataset and save
            # create new ann_dict
            # for each index, we take the sample from ann_master and make a new dict
            # save/create new annotations file


            # TODO take converted/transformed target and convert to annotation
            # style from json file
            # ann_dict = self.
            # ann_dict[file_name + str(file_size)]
            boxes = sample['boxes'].tolist()
            labels = sample['labels'].tolist()
            # image_id = sample['image_id'].tolist()
            # area = sample['area'].tolist()
            # iscrowd = sample['iscrowd'].tolist()

            regions = {}
            for i in range(len(boxes)):
                name = 'rect' # bounding boxes
                x = boxes[i][0]
                y = boxes[i][1]
                width = boxes[i][2] - boxes[i][0]
                height = boxes[i][3] - boxes[i][1]

                shape_attributes = {'name': name,
                                    'x': int(x),
                                    'y': int(y),
                                    'width': int(width),
                                    'height': int(height)}

                region_attributes = {'Weed': region_definition[labels[i]]}
                attributes = {'shape_attributes': shape_attributes,
                            'region_attributes': region_attributes}
                regions[str(i)] = attributes

            ann_orig = dataset.annotations
            file_ref = ann_orig[image_id]['fileref']
            file_size = ann_orig[image_id]['size'] # file size might actually change
            file_name = ann_orig[image_id]['filename']
            base_img_data = ann_orig[image_id]['base64_img_data']
            file_attributes = ann_orig[image_id]['file_attributes']
            # region_attributes = ann_orig[image_id]['region_attributes']
            # regions = ann_orig[image_id]['regions'] # we replace this w transformed


            # ann_dict[file_name + str(file_size)] = {
            ann_sample = {'fileref': file_ref,
                'size': file_size,
                'filename': image_name + '.png', # replace with new image_name
                'base64_img_data': base_img_data,
                'file_attributes': file_attributes,
                'regions': regions,
                'region_attributes': region_attributes}

            ann_dict[image_name + '.png' + str(file_size)] = ann_sample
            # ann_dict = self.sample_dict(ann_dict, target)

        # create annotations_out_file:

        if ann_append:
            # tried to just use 'a+' for dumping json file, but doesn't play
            # well with multiple objects. Must combine into a single object if

            # ann_transform_path already exists, combine insert ann_dict into
            # ann_transform
            # so first load existing ann_transform_path, then append ann_dict
            if os.path.isfile(ann_transform_path):
                # if already a file, we load the file
                ann_old = json.load(open(ann_transform_path))
                ann_all = {**ann_old, **ann_dict}
                self.make_annfile_from_dict(ann_all, ann_transform_path)
            else:
                # ann_transform_path is not a file, so shouldn't cause any
                # problems
                self.make_annfile_from_dict(ann_dict, ann_transform_path)
        else:
            self.make_annfile_from_dict(ann_dict, ann_transform_path)

        # append/merge with ann_in + ann_out
        ann_files = [ann_in, ann_transform]

        self.combine_annotations(ann_files, ann_dir, ann_out=ann_out)

        return ann_out, save_folder


    def get_polyons_image(self, annotations, idx):
        """ extract x, y coordinates if available """
        # all_points is a list of x,y points per index (ie, for a single image)
        # eg, all_points[0] = [x_pts0, y_pts0] for 0'th polygon
        #     all_points[1] = [x_pts1, y_pts1]

        # try:
        reg = annotations[idx]['regions']
        poly = []
        # find which regions are polygons
        # TODO should be able to do this in one line w/ list comprehension
        if len(reg) > 0:
            for i in range(len(reg)):
                if reg[i]['shape_attributes']['name'] == 'polygon':
                    poly.append(reg[i])

        all_points = []
        if len(poly) > 0:
            for j in range(len(poly)):
                x_points = poly[j]['shape_attributes']['all_points_x']
                y_points = poly[j]['shape_attributes']['all_points_y']
                all_points.append([x_points, y_points])
            # x_points = annotations[idx]['regions'][count]['shape_attributes']['all_points_x']
            # y_points = annotations[idx]['regions'][count]['shape_attributes']['all_points_y']
        # except:
        #     print('No polygon. Skipping', image_id)
        #     return

        # n_poly = len(poly)

        return all_points


    # https://towardsdatascience.com/generating-image-segmentation-masks-the-easy-way-dd4d3656dbd1
    def create_masks_from_poly(self,
                               img_dir_in,
                               ann_file_path,
                               mask_dir_out=None):
        """ create binary masks for given folder img_dir_in and ann_file, output
        to img_dir_out """
        # TODO to reduce compute, a flag for positive/negative image status
        # could be used so no mask need be generated for negative images, though
        # depending on how mask images are handled later on down the pipeline
        # (eg, training), one would need to further use the pos/neg image flag.

        # grab one image to get image width/height
        img_files = os.listdir(img_dir_in)
        try:
            # assume it is an image file

            image = Image.open(os.path.join(img_dir_in, img_files[0]))
            IMAGE_WIDTH = image.size[0] # could be the wrong syntax TODO
            IMAGE_HEIGHT = image.size[1]
        except:
            print(f'could not open image from {os.path.join(img_dir_in, img_files[0])}')
            return

        # read in the ann_file
        # load it
        # sort through all regions with "polygon" type
        # get all x, y points
        print(ann_file_path)
        with open(ann_file_path) as f:
            ann_dict = json.load(f)

        annotations = list(ann_dict.values())

        # goal is each image has a mask, with each polygon masked with a different number
        image_poly = {}
        image_ids = []
        for i in range(len(annotations)):

            filename = annotations[i]['filename']
            image_id = filename[:-4]
            # count_masks_per_image = 0  # count of masks/single groundtruth image

            image_poly[image_id] = self.get_polyons_image(annotations, i)
            image_ids.append(image_id)

        # print(f'dictionary size: {len(image_poly)}')

        # create folder for masks
        if mask_dir_out is None:
            mask_dir_out = os.path.join('masks')
        os.makedirs(mask_dir_out, exist_ok=True)

        # import code
        # code.interact(local=dict(globals(), **locals()))

        # for each image/iteration in annotations, generate mask and save image
        # each mask's values are incremented
        # NOTE we expect no more than 256 masks in a single image
        image_poly_list = list(image_poly.values())
        for i, polygons in enumerate(image_poly_list):
            # num_masks = len(polygons)
            mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), np.int32)

            # TODO check valid polygon?
            # might not be closed, etc
            # if there are any polygons, fill them in onto mask
            # with unique label

            # import code
            # code.interact(local=dict(globals(), **locals()))
            if len(polygons) > 0:
                count_polygons = 0
                for poly in polygons:
                    poly_pts = np.array(poly, np.int32).transpose()
                    count_polygons += 1
                    # import code
                    # code.interact(local=dict(globals(), **locals()))
                    cv.fillPoly(mask, [poly_pts], color=(count_polygons))

            # save mask
            mask_name = image_ids[i] + "_mask.png"
            # print(mask_name)
            mask_filepath = os.path.join(mask_dir_out, mask_name)
            cv.imwrite(mask_filepath, mask)


            # show image
            SHOW = False
            SAVE = False
                # mask_out = cv.normalize(mask, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
                # # mask_out = cv.cvtColor(mask_out, cv.COLOR_RGB2BGR)
                # win_name = mask_name
                # wait_time=0
                # cv.namedWindow(win_name, cv.WINDOW_GUI_NORMAL)
                # cv.imshow(win_name, mask_out)
                # cv.waitKey(wait_time)
                # cv.destroyWindow(win_name)
                # plt.imshow(mask)
                # plt.title(mask_name)
                # plt.show()



            if SHOW or SAVE:
                # import code
                # code.interact(local=dict(globals(), **locals()))
                print(f'image shape: {image.size}')
                print(f'mask shape: {mask.shape}')
                img_filename = os.path.join(img_dir_in, annotations[i]['filename'])
                image = Image.open(img_filename)
                patches = []
                if len(polygons) > 0:
                    for poly in polygons:
                        polt_pts = np.array(poly, np.int32).transpose()
                        patches.append(Polygon(polt_pts, closed=True))

                colors = 100 * np.random.rand(len(patches))
                p = PatchCollection(patches, alpha=0.4)
                p.set_array(colors)

                fig, ax = plt.subplots(2, 1)
                ax[0].imshow(image)
                # plt.imshow(image)
                # plot ontop the polygon

                ax[0].add_collection(p)

                # compare polygon to original image and annotation?
                ax[1].imshow(mask)
                if SHOW:
                    plt.show()

        # check
        img_list = os.listdir(img_dir_in)
        mask_list = os.listdir(mask_dir_out)
        print(f'number of images: {len(img_list)}')
        print(f'number of masks: {len(mask_list)}')
        if len(img_list) == len(mask_list):
            return True, mask_dir_out
        else:
            return False, mask_dir_out


    def unscale_polygon(self, polygon, output_size, input_size):
        """ unscale single polygon"""

        # determine new size multipliers based on input size
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

        # apply scale changes to all points in polygon
        p_out = polygon.copy()
        if len(polygon) > 0:
            new_x = np.array(p_out['all_points_x'], np.float32) / xChange
            new_y = np.array(p_out['all_points_y'], np.float32) / yChange
            p_out['all_points_x'] = [round(new_x[i]) for i in range(len(new_x))] # must be in "int" for json
            p_out['all_points_y'] = [round(new_y[i]) for i in range(len(new_y))]

        return p_out



    def generate_symbolic_links(self,
                          data_server_dir,
                          dataset_name,
                          img_dir_patterns = None,
                          ann_dir_patterns = None,
                          img_pattern = '*.png',
                          ann_pattern = '*.json',
                          data_dir_out = None,
                          ann_file_out = None,
                          default_dir = '/media/david/storage_device/Datasets/AOS_Kelpie',
                          unwanted_anns = None): # TODO make this general
        """ generate symlinks from original data server directory
        given specific string patterns for image directory, annotation directories, based on GLOB patterns
        output symbolic links in data_dir_out and ann_file_out """

        # given folder locations
        # given types of files to look for (glob) --> images/annotations
        # glob all relevant files
        # combine into singular json file with folder locations (might require new tag added)
        # output all photos into single folder via symlinks

        # TODO check for valid input

        # glob for all images
        glob_imgs = []
        for img_dir in img_dir_patterns:
            print(str(data_server_dir + img_dir + img_pattern))
            glob_img = glob.glob(str(data_server_dir + img_dir + img_pattern), recursive=True)
            glob_imgs.extend(glob_img)

        # check globs
        print('Found Image Globs:')
        # for g in glob_img:
        #     print(g)
        print(len(glob_imgs))

        # glob for all annotation files
        glob_anns = []
        for ann_dir in ann_dir_patterns:
            # print(str(data_server_dir + ann_dir_patterns + ann_pattern))
            glob_ann = glob.glob(str(data_server_dir + ann_dir + ann_pattern), recursive=True)
            # print('glob_ann = ' + str(data_server_dir + ann_dir + ann_pattern))
            glob_anns.extend(glob_ann)

        print('Found Annotation Globs:')
        print(glob_anns)
        print(len(glob_anns))
        for g in glob_anns:
            print(g)

        # create folder for symlinks
        if data_dir_out is None:
            data_dir_out = os.path.join(default_dir, dataset_name)
        img_out_dir = os.path.join(data_dir_out, 'images')
        img_raw_dir = os.path.join(data_dir_out, 'images_raw')
        ann_out_dir = os.path.join(data_dir_out, 'metadata')

        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(img_raw_dir, exist_ok=True)
        os.makedirs(ann_out_dir, exist_ok=True)

        # we only want to create symlinks of the images that have been processed
        
        

        # create symlinks
        print('Creating symbolic links')
        for img_path in glob_imgs:
            # print(f'{i+1} / {len(glob_img)}')
            img_name = os.path.basename(img_path)

            dst_file = os.path.join(img_raw_dir, img_name)
            # symlink fails if link already exists, so we unlink any existing links first
            # also, os.path.exists returns false if symlink is already broken
            if os.path.exists(dst_file) or os.path.lexists(dst_file):
                os.unlink(dst_file)
            # print(os.path.exists(dst_file))
            # print(dst_file)
            os.symlink(img_path, dst_file)

        # check number of symlinks in folder:
        print(f'num images in glob = {len(glob_imgs)}')
        img_list = os.listdir(img_raw_dir)
        print(f'num imgs in out_dir = {len(img_list)}')
        print('should be the same')

        # merge annotations to one:
        if ann_file_out is None:
            ann_raw_file_out = dataset_name + '_raw.json'

        ann_raw_path = os.path.join(ann_out_dir, ann_raw_file_out)
        res = self.combine_annotations(ann_files=glob_anns,
                                        ann_dir=False,
                                        ann_out=ann_raw_path)

        if ann_file_out is None:
            ann_file_out = dataset_name + '.json'

        ann_out_path = os.path.join(ann_out_dir, ann_file_out)
        # remove unprocessed images
        ann_out_path, img_dir_out = self.filter_processed_images_annotations(ann_path = ann_raw_path,
                                                                             ann_out_path=ann_out_path,
                                                                             img_dir= img_raw_dir, 
                                                                             img_out_dir= img_out_dir,
                                                                             unwanted_anns=unwanted_anns)

        # check number of entries in annotation out file
        ann_check = json.load(open(ann_out_path))
        print(f'num entries in ann_out = {len(ann_check)}')

        return ann_out_path, data_dir_out


    def trim_list(self, ann_dict, n_delete, img_dir, img_trim_dir, ann_file):
        """ helper function to trim images from annotation list """

        ann_list = list(ann_dict.values())
        to_delete = set(random.sample(range(len(ann_list)), n_delete))
        ann_trim_list = [x for k, x in enumerate(ann_list) if not k in to_delete]
        print(f'len ann_neg_trim = {len(ann_trim_list)}')

        # convert ann_trim_list to dictionary
        ann_dict_trim = {}
        for ann in ann_trim_list:
            ann_dict_trim = self.sample_dict(ann_dict_trim, ann)

        # clear out the relevant image trim directory
        # img_trim_path = os.path.join(root_dir, img_trim_dir)
        if os.path.isdir(img_trim_dir):
            shutil.rmtree(img_trim_dir)
        self.copy_symlinks_from_dict(ann_dict_trim, img_dir, img_trim_dir)

        ann_file_trim_out = os.path.join(ann_file[:-5] + '_trim.json')
        self.make_annfile_from_dict(ann_dict_trim, ann_file_trim_out)

        return ann_dict_trim, ann_file_trim_out


    def copy_symlinks_from_dict(self, ann_dict_in, img_src_dir, img_dst_dir):
        """ helper function to generate_dataset_from_symlink """
        os.makedirs(img_dst_dir, exist_ok=True)
        print(img_dst_dir)
        ann_list = list(ann_dict_in.values())
        for ann in ann_list:
            img_name = ann['filename'] # looping over dictionary gives just the key?
            img_path = os.path.join(img_src_dir, img_name)
            dst_file = os.path.join(img_dst_dir, img_name)
            if os.path.exists(dst_file) or os.path.lexists(dst_file):
                os.unlink(dst_file)
            # create sym link
            os.symlink(img_path, dst_file)
        return True


    def make_annfile_from_dict(self, ann_dict, ann_path):
        """ create annotations file from dictionary """
        ann_path = os.path.join(ann_path)
        ann_dir = os.path.dirname(ann_path)
        os.makedirs(ann_dir, exist_ok=True)
        print(ann_path)
        with open(ann_path, 'w') as f:
            json.dump(ann_dict, f, indent=4)

        return True


    def convert_project2annotations(self, proj_path, ann_path=None):
        """ convert VIA project file to annotations file """

        proj_path = os.path.join(proj_path)
        proj_dict = json.load(open(proj_path))

        if ann_path is None:
            proj_dir =  os.path.dirname(proj_path)
            proj_name = os.path.basename(proj_path)
            ann_name = proj_name + '_annotations.json'
            ann_path = os.path.join(proj_dir, ann_name)

        ann_dict = proj_dict['_via_img_metadata']

        self.make_annfile_from_dict(ann_dict, ann_path)

        return ann_path


    def sort_imgs_by_index(self, ann_list, idx, img_in_dir, img_out_dir, ann_out_path):
        """ sort images in annotation list by index, output symlinks to img_out_dir, and annotations to ann_out_path"""

        # create dictionary from idx
        ann_dict = {}
        ann_idx = [ann_list[i] for i in idx]
        for ann in ann_idx:
            ann_dict = self.sample_dict(ann_dict, ann)

        # ann_out_dir = os.path.dirname(ann_out_path)
        self.copy_symlinks_from_dict(ann_dict, img_in_dir, img_out_dir)

        # make dictionary, save output
        self.make_annfile_from_dict(ann_dict, ann_out_path)

        # check:
        print(f'num images in ann_dict = {len(ann_dict)}')
        img_list = os.listdir(img_out_dir)
        print(f'num imgs in img_out_dir = {len(img_list)}')
        if len(ann_dict) == len(img_list):
            print('check passed - images same in ann_file and img_out_dir')
            return True, ann_dict
        else:
            print('check failed - not equal')
            print(ann_out_path)
            print(img_out_dir)
            return False, ann_dict


    def filter_processed_images_annotations(self,
                                            ann_path,
                                            img_dir,
                                            ann_out_path=None,
                                            img_out_dir=None,
                                            unwanted_anns=None):
        """filter out processed images and annotations - that only have the is_processed flag

        Args:
            ann_path (_type_): _description_
            img_dir (_type_): _description_
            ann_out_path (_type_, optional): _description_. Defaults to None.
            img_out_dir (_type_, optional): _description_. Defaults to None.
        """
        # load ann_file
        ann_dict = json.load(open(ann_path))
        ann_list = list(ann_dict.values())
        if '_via_settings' in ann_list:
                # only grab the via img metadata
                ann_list = ann_list['_via_img_metadata']
        ann_out = [a for a in ann_list if int(a['file_attributes']['is_processed']) == 1]

        print(f'Out of {len(ann_list)} images in img_dir, {len(ann_out)} images are is_processed==1')
        if unwanted_anns is not None:
            # Keep the annotation if there are no annotations
            ann_out = [a for a in ann_out if len(a['regions']) == 0 or 
            # or if no annotated regions are of undesired species
                       len([1 for region in a['regions'] if 'species' not in region['region_attributes'] or region['region_attributes']['species'] in unwanted_anns]) == 0]
            print(f'Removing unwanted annotations {unwanted_anns} leaves {len(ann_out)} images in dataset')

        # output ann_processed as ann_out_path (first convert back to dict)
        ann_dict_out = {}
        for a in ann_out:
            ann_dict_out = self.sample_dict(ann_dict_out, a)
        
        # create annotations out file:
        self.make_annfile_from_dict(ann_dict_out, ann_out_path)
        
        # then take all those image names, and copy them over to new folder
        # img_names = [a['filename'] for a in ann_out]
        self.copy_symlinks_from_dict(ann_dict_out, img_dir, img_out_dir)
        
        return ann_out_path, img_out_dir


    def generate_dataset_from_symbolic_links(self,
                                             root_dir,
                                             ann_path,
                                             ann_out_file=None,
                                             pos_neg_img_ratio=1.0,
                                             img_out_dir=None):
        """ generate dataset from symbolic link folders """
        # load annotations file
        # from images folder that has a mix of pos/neg images
        # create folder for symlinks into positive and negative
        # take X pos images based on take_pos_imgs ratio
        # take Y neg images, based on pos_neg_img_ratio
        # save this as a trim negative set
        # combine pos, neg image directories and respective json files into balanced json file/folder\
        # TODO what if insufficient negative/positive images?

        img_dir_in = os.path.join(root_dir, 'images')

        # load annotations file
        ann_dict = json.load(open(ann_path))
        ann_list = list(ann_dict.values())

        

        # TODO check if len(img_dir_in) == len(ann_dict)
        # we assume they match, otherwise, might be problems

        # first, only select the "is-processed" images
        # ann_processed = [a for a in ann_list if int(a['file_attributes']['is_processed']) == 1]

        # make a new image directory for processed images:
        # img_dir_in

        # get indices of positive, negative images  (wrt ann_path)
        idx_pos, idx_neg = self.count_pos_neg_images(ann_list)

        # create folders for symlinks
        # ann_out_dir = os.path.dirname(ann_path)
        img_neg_out_dir = os.path.join(root_dir, 'images_neg')
        img_pos_out_dir = os.path.join(root_dir, 'images_pos')

        ann_pos_out = ann_path[:-5] + '_pos.json'
        ann_neg_out = ann_path[:-5] + '_neg.json'

        # sort images based on index, create corresponding symlink image folder and annotation file
        # do this for both pos, negative image sets
        res, ann_pos = self.sort_imgs_by_index(ann_list, idx_pos, img_dir_in, img_pos_out_dir, ann_pos_out)
        res, ann_neg = self.sort_imgs_by_index(ann_list, idx_neg, img_dir_in, img_neg_out_dir, ann_neg_out)

        # "trim" negative images based on ratios
        # TODO could make this a method: self.trim_negative_images()
        # want to par down negative images to X:Y of num img_pos: num img_neg
        n_pos = len(ann_pos)
        n_neg = len(ann_neg)
        n_neg_goal = round(n_pos * pos_neg_img_ratio)

        # import code
        # code.interact(local=dict(globals(), **locals()))

        img_neg_trim_dir = os.path.join(root_dir, 'images_neg_trim')
        img_pos_trim_dir = os.path.join(root_dir, 'images_pos_trim')


        n_delete = n_neg - n_neg_goal
        if n_delete <= 0:
            print('no trimming of negative images, we need to trim positive images')
            print(f'n_delete = {n_delete}')

            # TODO call trim_list with -ndelete, ann_pos
            ann_dict_pos, ann_file_pos = self.trim_list(ann_dict=ann_pos,
                                                        n_delete=-n_delete,
                                                        img_dir=img_dir_in,
                                                        img_trim_dir=img_pos_trim_dir,
                                                        ann_file=ann_pos_out)
            ann_dict_neg = ann_neg
            ann_file_neg = ann_neg_out
            # copy all images from img_neg to img_neg_trim
            self.copy_symlinks_from_dict(ann_neg, img_dir_in, img_neg_trim_dir)
            
        else:
            print('trimming negative images')
            # randomly sample which elements to delete, based on index
            # to_delete = set(random.sample(range(len(ann_neg)), n_delete))
            # ann_neg_list = list(ann_neg.values())
            # ann_neg_trim_list = [x for k, x in enumerate(ann_neg_list) if not k in to_delete]
            # print(f'len ann_neg_trim = {len(ann_neg_trim_list)}')
            ann_dict_neg, ann_file_neg = self.trim_list(ann_dict=ann_neg,
                                                        n_delete=n_delete,
                                                        img_dir=img_dir_in,
                                                        img_trim_dir=img_neg_trim_dir,
                                                        ann_file=ann_neg_out)
            ann_dict_pos = ann_pos
            ann_file_pos = ann_pos_out
            self.copy_symlinks_from_dict(ann_pos, img_dir_in, img_pos_trim_dir)

        # convert ann_neg_trim_list to dictionary
        # ann_neg_trim = {}
        # for ann in ann_neg_trim_list:
        #     ann_neg_trim = self.sample_dict(ann_neg_trim, ann)

        # # need to clear out img_neg_trim_dir before loading it up
        # img_neg_trim_dir = os.path.join(root_dir, 'images_neg_trim')
        # if os.path.isdir(img_neg_trim_dir):
        #     shutil.rmtree(img_neg_trim_dir)
        # self.copy_symlinks_from_dict(ann_neg_trim, img_dir_in, img_neg_trim_dir)

        # # make dictionary, save output
        # ann_neg_trim_out = os.path.join(ann_neg_out[:-5] + '_trim.json')
        # self.make_annfile_from_dict(ann_neg_trim, ann_neg_trim_out)

        # check:
        print(f'num annotations in ann_pos_trim = {len(ann_dict_pos)}')
        img_pos_list = os.listdir(img_pos_trim_dir)
        print(f'num imgs in img_pos_trim_dir = {len(img_pos_list)}')
        print('should be the same')
        
        print(f'num images in ann_neg_trim = {len(ann_dict_neg)}')
        img_neg_list = os.listdir(img_neg_trim_dir)
        print(f'num imgs in img_neg_trim_dir = {len(img_neg_list)}')
        print('should be the same')

        # import code
        # code.interact(local=dict(globals(), **locals()))

        # findally, combine pos, neg img directories and respective json files
        if img_out_dir is None:
            img_out_dir = os.path.join(root_dir, 'images_balanced')
        # clear img_out_dir first
        if os.path.isdir(img_out_dir):
            shutil.rmtree(img_out_dir)

        # copy images over as symlinks to img_out_dir
        self.copy_symlinks_from_dict(ann_dict_pos, img_pos_trim_dir, img_out_dir)
        self.copy_symlinks_from_dict(ann_dict_neg, img_neg_trim_dir, img_out_dir)


        # combine annotation files
        ann_combine = [ann_file_pos, ann_file_neg]
        if ann_out_file is None:
            ann_out_file = ann_path[:-5] + '_balanced.json'
        res = self.combine_annotations(ann_combine, ann_dir=False, ann_out=ann_out_file)

        # check
        img_out_list = os.listdir(img_out_dir)
        img_pos_list = os.listdir(img_pos_trim_dir)
        img_neg_list = os.listdir(img_neg_trim_dir)
        if (len(img_pos_list) + len(img_neg_list)) != len(img_out_list):
            print('warning: number of images in symbolically linked folder does not = num positive images + num negative images')
            print(f'img_pos_list +  img_neg_list = {len(img_pos_list) + len(img_neg_list)}')
            print(f'img_out_list = {len(img_out_list)}')
            # TODO should replace this with an assert or error statement

        ann_out_dict = json.load(open(ann_out_file))
        print(f'len of ann_out_dict = {len(ann_out_dict)}')

        # import code
        # code.interact(local=dict(globals(), **locals()))
        return img_out_dir, ann_out_file

# =========================================================================== #

if __name__ == "__main__":

    print('PreProcessingToolbox.py')

    """ testing create_masks_from_poly """

    ppt = PreProcessingToolbox()
    # db_name = 'Tussock_v0_mini'
    # root_dir = os.path.join('/home', 'dorian', 'Data', 'agkelpie',
    #                           db_name)
    # img_dir_in = os.path.join(root_dir, 'Images', 'All')
    # ann_file_name = 'via_project_29Apr2021_17h43m_json_bbox_poly_pt.json'
    # ann_file_path = os.path.join(root_dir, 'Annotations', ann_file_name)
    # img_dir_out = os.path.join(root_dir, 'Masks', 'All')
    # ppt.create_masks_from_poly(img_dir_in, ann_file_path, img_dir_out)

    # testing: generate_symbolic_links
    res = ppt.generate_symbolic_links(data_server_dir='/home/agkelpie/Data/03_Tagged',
                                      dataset_name='2021-03-25_MFS_Tussock',
                                      img_dir_patterns=['/2021-03-25/*/images/', '/2021-03-26/Location_1/images/'],
                                      ann_dir_patterns=['/2021-03-25/*/metadata/', '/2021-03-26/Location_1/metadata/'])
    print(res)

    # import code
    # code.interact(local=dict(globals(), **locals()))
