#! /usr/bin/env python

"""
script to check images match
"""

import os
from weed_detection import PreProcessingToolbox

pt = PreProcessingToolbox()

root_dir = os.path.join('/home', 'dorian','Data','AOS_TussockDataset')
img_dir = os.path.join(root_dir, 'positive-tags')
ann_file = os.path.join(root_dir, 'Annotations', 'Thursday_25-03-21_G507_location1_positive-tags_labels.json')

res = pt.check_images_match_annotations(img_dir, ann_file)

print(res)