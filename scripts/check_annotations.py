#! /usr/bin/env python

"""
script to check through annotations and note any that do not match-up/have a box/polygon/point triplet
based on nearest centroid
"""

# load json file
# for each annotation
#   sort through each region
#   check if number of box/poly/point makes three
#   else return warning/note/print which file
#   regardless, for each region, compute centroid
#   compute distances to each centroid
#   based on some threshold, declare correspondence

import os
import json

# dataset location
folder = os.path.join('/home', 'dt-cronus', 'Dropbox', 'QUT_WeedImaging')
dataset_name = 'Tussock_v2'
root_dir = os.path.join(folder, dataset_name)
ann_dir = os.path.join(root_dir, 'Annotations')
img_dir = os.path.join(root_dir, 'Images', 'All')

# location of json file + name
json_name = 'via_project_07Jul2021_08h00m_240_test.json'
json_path = os.path.join(ann_dir, json_name)

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
            name = r['shape_attributes']['name']
            if name == 'rect':
                print('we have box')
                # set region_attributes: STB
                # TODO r['region_attributes'] = dict{'Weed': 'STB'} # should 'Weed'?
                # r['region_attributes'] = {'plant': 'STB'}
                rect = r['shape_attributes']
                x_topleft = rect['x'] # top left corner of the bbox wrt image coordinates, 0,0 being topleft corner
                y_topleft = rect['y']
                w = rect['width']
                h = rect['height']
                cx = x_topleft + w / 2.0
                cy = y_topleft + h / 2.0

            elif name == 'polygon':
                print('we have polygon')
                # r['region_attributes'] = {'plant': 'STP'}
                # set region_attributes: STP
                # https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
                # https://pyhyd.blogspot.com/2017/08/calculate-centroid-of-polygon-with.html
                # TODO implement the above as a function
            elif name == 'point':
                print('we have point')
                # r['region_attributes'] = {'plant': 'STS'}
                # set region_attributes: STS
                pt = r['shape_attributes']
                cx = pt['cx']
                cy = pt['cy']
            else:
                print('uh oh, not a valid name, print name')
    else:
        # what do we do when no region properties?
        # could just be an image with no region properties, should just skip
        print(f'no annotations: {img_name}')

import code
code.interact(local=dict(globals(), **locals()))
