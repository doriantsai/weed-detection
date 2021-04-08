#! /usr/bin/env python

"""
combine annotation files
"""

import os
import json

# annotation file locations
annotations_dir = '/home/dorian/Data/AOS_TussockDataset/Tussock_v0/Annotations'

a1 = os.path.join(annotations_dir, 'annotations_tussock_210325_G507_location1.json')
a2 = os.path.join(annotations_dir, 'annotations_tussock_210326_G507_location1.json')
a3 = os.path.join(annotations_dir, 'annotations_tussock_210325_G507_location2.json')

aout_name = 'annotations_tussock_21032526_G507_combined.json'

# load json files as dictionaries
a1 = json.load(open(a1))
a2 = json.load(open(a2))
a3 = json.load(open(a3))

# for now, assume unique key-value pairs, but
# TODO should check to make sure pairs are unique
# can do simple check - check length after-the-fact
a_comb = {**a1, **a2, **a3}

la1 = len(a1)
la2 = len(a2)
la3 = len(a3)
lall = la1 + la2 + la3

print(len(a1))
print(len(a2))
print(len(a3))
print(lall)
print(len(a_comb))
# last two numbers should be the same

# save a_comb
with open(os.path.join(annotations_dir, aout_name), 'w') as ann_file:
    json.dump(a_comb, ann_file, indent=4)

import code
code.interact(local=dict(globals(), **locals()))
