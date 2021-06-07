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
import cv2 as cv
from posix import ST_SYNCHRONOUS
import numpy as np
from PIL import Image
import shutil
from subprocess import call
# from torch._C import namedtuple_solution_cloned_coefficient
from weed_detection.WeedDataset import WeedDataset, Compose, \
    RandomBlur, RandomVerticalFlip, RandomHorizontalFlip, \
    RandomBrightness, RandomContrast, RandomHue, \
    RandomSaturation, ToTensor
import torch
import random


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

            ann_open = os.path.join(ann_dir, ann_files[i])
            # try:
            # print(i)
            # print(ann_open)
            # import code
            # code.interact(local=dict(globals(), **locals()))
            ann.append(json.load(open(ann_open)))
            # except:
            #     print(f'ERROR: failed to open {ann_open}')
            #     break

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
        ann_master = list(ann_master.values())

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
                         ratio_train_test=[0.7, 0.2],
                         clear_image_folders=True):
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

        # TODO if clear_image_folders is True, then delete/clear all image files
        # already in train/test/val folders

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
        # if ratio_train_test is None:
        #     ratio_train_test = [0.7, 0.2]
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

        # print('ds_train {}'.format(len(ds_train)))
        # print('ds_test {}'.format(len(ds_test)))
        # print('ds_val {}'.format(len(ds_val)))

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
        ann_dir = os.path.join(root_dir, 'Annotations')
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

        save_folder = os.path.join(root_dir, 'Images','Augmented')
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
                with open(ann_transform_path, 'w') as ann_file:
                    json.dump(ann_all, ann_file, indent=4)
            else:
                # ann_transform_path is not a file, so shouldn't cause any
                # problems
                with open(ann_transform_path, 'w') as ann_file:
                    json.dump(ann_dict, ann_file, indent=4)
        else:
            with open(ann_transform_path, 'w') as ann_file:
                json.dump(ann_dict, ann_file, indent=4)

        # append/merge with ann_in + ann_out
        ann_files = [ann_in, ann_transform]

        self.combine_annotations(ann_files, ann_dir, ann_out=ann_out)

        return ann_out, save_folder


    def get_polyons_image(annotations, idx):
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
                
        n_poly = len(poly)

        return all_points, n_poly


    # https://towardsdatascience.com/generating-image-segmentation-masks-the-easy-way-dd4d3656dbd1
    def create_masks_from_poly(self,
                               img_dir_in,
                               ann_file_path,
                               mask_dir_out=None):
        """ create binary masks for given folder img_dir_in and ann_file, output
        to img_dir_out """

        # grab one image to get image width/height
        img_files = os.listdir(img_dir_in)
        try:
            # assume it is an image file
            image = Image.open(img_files[0])
            IMAGE_WIDTH = image.shape[0] # could be the wrong syntax TODO
            IMAGE_HEIGHT = image.shape[1]
        except:
            print(f'could not open image from {img_files[0]}')
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
        for itr in annotations:
            filename = annotations[itr]['filename']
            image_id = filename[:-4]
            # count_masks_per_image = 0  # count of masks/single groundtruth image

            image_poly[image_id] = self.get_polyons_image(annotations, itr)
            image_ids.append(image_id)

        # print(f'dictionary size: {len(image_poly)}')

        # create folder for masks
        if mask_dir_out is None:
            mask_dir_out = os.path.join('masks')
        os.makedirs(img_dir_out, exist_ok=True)

        # for each image/iteration in annotations, generate mask and save image
        # each mask's values are incremented 
        # NOTE we expect no more than 256 masks in a single image
        for i, polygons in enumerate(image_poly):
            # num_masks = len(polygons)
            mask = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT))

            # TODO check valid polygon? 
            # might not be closed, etc
            # if there are any polygons, fill them in onto mask
            # with unique label
            count_polygons = 1
            if len(polygons) > 0:
                for poly in polygons:
                    p = np.array(poly)
                    count_polygons += 1
                    cv.fillPoly(mask, [p], color=(count_polygons))

            # save mask
            mask_name = image_ids[i] + "_mask.png"
            print(mask_name)
            mask_filepath = os.path.join(mask_dir_out, mask_name)
            cv.imwrite(mask_filepath, mask)
            


        import code
        code.interact(local=dict(globals(), **locals()))




# =========================================================================== #

if __name__ == "__main__":

    print('PreProcessingToolbox.py')

    """ testing create_masks_from_poly """

    ppt = PreProcessingToolbox()
    db_name = 'Tussock_v0_mini'
    root_dir = os.path.join('/home', 'dorian', 'Data', 'AOS_TussockDataset',
                              db_name)
    img_dir_in = os.path.join(root_dir, 'Images')
    ann_file_name = 'via_project_29Apr2021_17h43m_json_bbox_poly_pt.json'
    ann_file_path = os.path.join(root_dir, 'Annotations', ann_file_name)
    img_dir_out = os.path.join(root_dir, 'temp')
    ppt.create_masks_from_poly(img_dir_in, ann_file_path, img_dir_out)