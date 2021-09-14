#! /usr/bin/env python

"""
script to check images match
"""

import os
from weed_detection import PreProcessingToolbox

ppt = PreProcessingToolbox()

root_dir = os.path.join('/home', 'dorian','Data','agkelpie',
    '2021-03-26_MFS_Horehound')
img_dir = os.path.join(root_dir, 'images')
ann_file = os.path.join(root_dir, 'metadata', '2021-03-26_MFS_Horehound.json')

res = ppt.check_images_match_annotations(img_dir, ann_file)
print(res)

if res:
    print('Yay, images match annotations!')
else:
    print('Error: images do not match annotations!')
