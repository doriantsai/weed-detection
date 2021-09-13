#! /usr/bin/env python

"""
script to convert VIA project file to annotations file
"""
# TODO add this as a function into PPT
# in VIA project file, the "export annotations" button just takes the _via_img_metadata
# tag and exports that

import os
import json

project_file = '20210907-agkelpie-MFS-02-positive_tags_labels_polygons_projectfile.json'
# ann_file_out = '20210819-MFS-01-bootprogress-570-occlusion24_test.json'
actual_ann_out = '20210907-agkelpie-MFS-02-positive_tags_labels_polygons.json'

ann_dir = '/home/dorian/Data/AOS_TussockDataset/03_Tagged/2021-03-26/Location_2/metadata'
proj_path = os.path.join(ann_dir, project_file)
ann_path = os.path.join(ann_dir, actual_ann_out)
# goal is to get ann_file_out == actual_ann_out

proj_dict = json.load(open(proj_path))
ann_dict = proj_dict['_via_img_metadata']

# actual_ann = json.load(open(ann_path))

# if ann_dict == actual_ann:
#     print('matches')
# else:
#     print('time to dig')

# save ann_path
with open(ann_path, 'w') as f:
    json.dump(ann_dict, f, indent=4)

import code
code.interact(local=dict(globals(), **locals()))


