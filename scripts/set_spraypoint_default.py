#! /usr/bin/env python

"""
script to access annotation json (note: exported VIA annotations file, not the saved VIA project file)
TODO probably should be updated to work on VIA project file
author: Dorian Tsai
date: 02 Aug 2021
"""

# general idea behind code:
# set folder/file locations
# set spraypoint default
# open ann file, find all point annotations
# if not specified (eg, empty), then set spraypoint default

import os
import json

# folders and file names
ann_dir = os.path.join('/home', 'agkelpie', 'Data', 'Horehound_v0', 'Annotations')
ann_file = '20210803-agkelpie-MFS-02-030_json.json'
ann_filepath = os.path.join(ann_dir, ann_file)

ann_file_out = '20210803-agkelpie-MFS-02-030_json_out.json'

# spraypoint default (not: dictionary entry)
spray_default = {"occluded": "0"}

# load ann_filepath as unordered dictionary
print(ann_filepath)
with open(ann_filepath) as f:
    ann_dict = json.load(f)

# get all keys, values from dict into lists (need to recreate dict later on)
keys = []
values = []
for k in ann_dict:
    keys.append(k)
    values.append(ann_dict[k])

# convert to list
# ann = list(ann_dict.values())
ann = values

# iterate over annotations
print('iterating over annotations')
for i, a in enumerate(ann):
    # extract filename, useful for checking on image
    fname = a['filename']
    image_id = fname[:-4]

    # regions
    reg = a['regions']
    if len(reg) > 0:
        for j, r in enumerate(reg):
            if r['shape_attributes']['name'] == 'point':
                # set default by merging with existing dict
                # r['region_attributes'].update(spray_default)


                print(f'found point: img {i}, ann {j}')
                # print(r['region_attributes'])
                # NOTE not sure if this works to update the original annotation list
                # TODO if occluded is not there/spray default is not defined, update dictionary
                ann[i]['regions'][j]['region_attributes'].update(spray_default)
                # print(ann[i]['regions'][j]['region_attributes'])

# convert ann from list back to dict
ann_out = {}
for i, k in enumerate(keys):
    ann_out = {**ann_out, k: ann[i]}

# for ai in ann:
#     ann_out = {**ann_out, **ai}

# pauses python script at this point, allowing user to check data interactively through terminal
import code
code.interact(local=dict(globals(), **locals()))

# save output
ann_out_filepath = os.path.join(ann_dir, ann_file_out)
with open(ann_out_filepath, 'w') as af:
    json.dump(ann_out, af, indent=4)


# check?
# TODO do automatically, for now, open file manually to see
