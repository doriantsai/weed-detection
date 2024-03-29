#! /usr/bin/env python

"""
# script to sort through polygon/point annotations and apply labels
# STS and STP, respectively

# file locations
# export json file from prject file (need to do this manually)
# setup file locations
# import json
# load json file as dictionary, then convert it to ordered list


# since this is an iterative process, find all bboxes that have a corresponding polygon/point pair

# sort through each region attributes
# for each region, if polygon, set label as STP
#                  if point, set label as STS
# also replaces the original "Weed" key with "plant", since not all annotations will be "weed" in the future

# finally, check to make sure polygon/point has a corresponding box and is labelled
# report: how many boxes have been converted
"""

import os
import json
from weed_detection.PreProcessingToolbox import PreProcessingToolbox

# dataset location
root_dir = os.path.join('/home',  'agkelpie', 'Data', '03_Tagged', '2021-03-26', 'Location_2')
ann_dir = os.path.join(root_dir, 'metadata')

# location of json file + name
json_name = 'Friday_26-03-21_G507_location2_positive-tags_labels_polygons.json'
json_path = os.path.join(ann_dir, json_name)

json_name_save = 'Friday_26-03-21_G507_location2_positive-tags_labels_polygons_v1.json'

# load json as a list
ann_dict = json.load(open(json_path))
ann_list = list(ann_dict.values())

# for every image index, find the region attributes
for i, ann in enumerate(ann_list):
    print(i)
    img_name = ann['filename']
    reg = ann['regions']
    # each region is an annotation, either a box, poly or pt
    if len(reg) > 0:
        for r in reg:
            # import code
            # code.interact(local=dict(globals(), **locals()))
            name = r['shape_attributes']['name']
            if name == 'rect':
                print('we have box')
                # set region_attributes: STB
                # TODO r['region_attributes'] = dict{'Weed': 'STB'} # should 'Weed'?
                r['region_attributes'] = {'species': 'Horehound'}
            elif name == 'polygon':
                print('we have polygon')
                r['region_attributes'] = {'species': 'Horehound'}
                # set region_attributes: STP
            elif name == 'point':
                print('we have point')
                r['region_attributes'] = {'species': 'Horehound'}
                # set region_attributes: STS
            else:
                print('uh oh, not a valid name, print name')

        # TODO now update ann['regions'], maybe this happens automatically thanks to Python pointers?
        ann['regions'] = reg

    else:
        # what do we do when no region properties?
        # could just be an image with no region properties, should just skip
        print(f'no annotations: {img_name}')

PPT = PreProcessingToolbox()
ann_dict_out = {}
for i, ann in enumerate(ann_list):
    sample = ann_list[i]
    ann_dict_out = PPT.sample_dict(ann_dict_out, sample)

# save json:
PPT.make_annfile_from_dict(ann_dict_out, os.path.join(ann_dir, json_name_save))


import code
code.interact(local=dict(globals(), **locals()))

# TODO will probably need to manually go through and identify the occluded cases
# TODO check if box is unmatched eg, no box is covered by a polygon or point?
# end code