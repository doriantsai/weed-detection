#! /usr/bin/env python

"""
script to convert VIA project file to annotations file
"""
# TODO add this as a function into PPT
# in VIA project file, the "export annotations" button just takes the _via_img_metadata
# tag and exports that

import os
import json

project_file = '20210819-MFS-01-bootprogress-570-occlusion24.json'
ann_file_out = '20210819-MFS-01-bootprogress-570-occlusion24_test.json'
actual_ann_out = '20210819-MFS-01-bootprogress-570-occlusion24_json.json'

# goal is to get ann_file_out == actual_ann_out

proj_dict = json.load(open(project_file))
ann_dict = proj_dict['_via_img_metadata']

actual_ann = json.load(open(actual_ann_out))

if ann_dict == actual_ann:
    print('matches')
else:
    print('time to dig')

import code
code.interact(local=dict(globals(), **locals()))


