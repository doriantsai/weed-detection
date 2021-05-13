#! /usr/bin/env python

import os
from PIL import Image

# convert all images to png in a given folder

filepath = '/home/dorian/Data/SerratedTussockDataset_v1/Images'

files = os.listdir(filepath)

for f in files:
    if f.endswith('.jpg') or f.endswith('.tif') or f.endswith('.bmp') or f.endswith('.jpeg'):
        # convert to png by opening then saving:
        img = Image.open(os.path.join(filepath, f))
        g = f[:-4] + '.png'
        print('saving as {}/{}'.format(filepath, g))
        img.save(os.path.join(filepath, g))
