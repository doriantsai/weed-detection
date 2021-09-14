#! /usr/bin/env python

"""
change region dictionary regions to to lists in json file
VIA outputs regions as lists, but somehow
annotations_tussock_21032526_G507_all.json outputs them as a dictionary
"""

import os
import json

from numpy import empty
from weed_detection.PreProcessingToolbox import PreProcessingToolbox

dataset_name = '2021-03-26_MFS_Horehound'

# folder locations and file names
root_dir = os.path.join('/home',
                        'dorian',
                        'Data',
                        'agkelpie',
                        dataset_name)



# corresponding annotations file to Images/All
ann_file = '2021-03-26_MFS_Horehound.json'

# annotation files Master (contains all images - we don't touch this file, just
# use it as a reference/check)
ann_dir = os.path.join(root_dir, 'metadata')
ann_path = os.path.join(ann_dir, ann_file)

# annotation files out
# ann_file_out = 'annotations_tussock_21032526_G507_all_regions_list.json'
# ann_file_out = ann_file[:-5] + '_reglist.json'
ann_file_out = ann_file
ann_path_out = os.path.join(ann_dir, ann_file_out)


# TODO load json file, sort through dict, change each regions to list
ann_dict = json.load(open(ann_path))
ann_list = list(ann_dict.values())

for i, ann in enumerate(ann_list):
    # print(i)
    # print(ann)
    regions = ann['regions']

    # print(type(regions))

    nreg = len(regions)
    if isinstance(regions, dict):
        nreg = len(regions)
        reg_list = []

        if nreg > 0:
            # iterate through region dictionary, creating l ist
            for r in range(nreg):
                reg_data = regions[str(r)]
                reg_list.append(reg_data)

        # change regions dictionary to list
        ann_list[i]['regions'] = reg_list
    elif len(regions) <= 0:
        ann_list[i]['regions'] = []
    # import code
    # code.interact(local=dict(globals(), **locals()))

    # if i > 2:
    #     break


# now, need to save ann_list as dictionary
keys = []
# values = []
for k in ann_dict:
    keys.append(k)
    # values.append(ann_dict[k])

ann_out = {}
for i, k in enumerate(keys):
    ann_out = {**ann_out, k: ann_list[i]}


with open(ann_path_out, 'w') as af:
    json.dump(ann_out, af, indent=4)

# check
# ann_dict = json.load(open(ann_path_out))

# n_ann_file = len(ann_dict)
# print(f'ann file count = {n_ann_file}')

print('reach end of code')

import code
code.interact(local=dict(globals(), **locals()))
