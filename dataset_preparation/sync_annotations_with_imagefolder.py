#! /usr/bin/env python

"""
script to sync images folder with json file
read all image files in given folder
take from "master" annotations file
create annotations file from all images in given folder
TODO turn this into a function, and call for given folder
"""

import os
import json


def sync_annotations_with_folder(image_dir, annotations_master, annotations_out):

    done = False
    print('sync from images to json file')
    print('image_dir = ' + image_dir)
    print('annotations_master = ' + annotations_master)
    print('annotations_out = ' + annotations_out)

    # read in annotations_master
    annotations_master = json.load(open(annotations_master))

    # find all files in image_dir
    # assume only .png files inside folder
    img_list = os.listdir(image_dir)

    # find all dictionary entries that match files in image_dir
    # first, create list out of dictionary values, so we have indexing
    annotations_master_list = list(annotations_master.values())
    # then create a list out of all the filenames
    master_filename_list = []
    for s in annotations_master_list:
        master_filename_list.append(s['filename'])

    # create dictionary with keys: list entries, values: indices
    ind_dict = dict((k, i) for i, k in enumerate(master_filename_list))
    # find intersection of master and local:
    inter = set(ind_dict).intersection(img_list)
    # compile list of indices of the intersection
    indices = [ind_dict[x] for x in inter]

    # for each index, we take the sample from annotations_master_list and make a new dictionary
    ann_dict = {}
    for i in indices:
        sample = annotations_master_list[i]

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

    # create annotations_out file
    # save filtered annotation_file
    with open(annotations_out, 'w') as ann_file:
        json.dump(ann_dict, ann_file, indent=4)
    done = True
    return done


if __name__ == "__main__":

    TRAIN_VAL_TEST_FOLDERS = True

    # if test and train are in separate image folders
    if TRAIN_VAL_TEST_FOLDERS:
        # set directories
        root_dir = os.path.join('/home', 'dorian', 'Data', 'AOS_TussockDataset', 'Tussock_v1')
        train_folder = os.path.join(root_dir, 'Images', 'Train')
        val_folder = os.path.join(root_dir, 'Images', 'Validation')
        test_folder = os.path.join(root_dir, 'Images', 'Test')

        # set annotation names/files
        annotation_dir = os.path.join(root_dir, 'Annotations')
        annotations_master = os.path.join(annotation_dir, 'annotations_tussock_21032526_G507_combined_all.json')
        annotations_train = os.path.join(annotation_dir, 'annotations_tussock_21032526_G507_train.json')
        annotations_val = os.path.join(annotation_dir, 'annotations_tussock_21032526_G507_val.json')
        annotations_test = os.path.join(annotation_dir, 'annotations_tussock_21032526_G507_test.json')

        # function calls for each folder
        sync_annotations_with_folder(train_folder, annotations_master, annotations_train)
        sync_annotations_with_folder(val_folder, annotations_master, annotations_val)
        sync_annotations_with_folder(test_folder, annotations_master, annotations_test)

    else:
        # single image folder

        # set directories
        # root_dir = os.path.join('/home', 'dorian', 'Data', 'AOS_TussockDataset', 'Tussock_v0', 'Occluded_Cases')
        root_dir = os.path.join('/home', 'dorian', 'Data', 'AOS_TussockDataset', 'Tussock_v0_mini')

        image_dir = os.path.join(root_dir, 'Images')
        annotation_dir = os.path.join(root_dir, 'Annotations')

        # set files
        # annotations_master includes all the dictionary-entries from all positive/negative images
        # ie all images in Tusock**positivetags folders
        # TODO combine all json files from positivetags folders (to include negative images)
        annotations_master = os.path.join(annotation_dir, 'annotations_tussock_21032526_G507_combined_all.json')
        annotations_out = os.path.join(annotation_dir, 'annotations_tussock_21032526_G507_mini.json')

        sync_annotations_with_folder(image_dir, annotations_master, annotations_out)

    print('filtered annotation file complete')

    import code
    code.interact(local=dict(globals(), **locals()))