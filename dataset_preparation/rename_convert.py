#! /usr/bin/env python

# python script to rename all the files to a desired string pattern
# because I could not figure out the damn perl expression for use with the rename 
# function

import os
from subprocess import call
from PIL import Image

# read in all files in given folder
# find all matching given pattern
# rename all files of given pattern according to a different pattern

filepath = '/home/dorian/Data/SerratedTussockDataset_v1/Images'

files = os.listdir(filepath)
i = 0
for f in files:


    # check file extension
    # if png, just rename
    if f.endswith('.png'):
        # rename file to st-###.jpg
        # pad with 3 zeros, incrementing i
        strpatt = 'st' + str(i).zfill(3) + '.png'
        print(f + ' --> ' + strpatt)
        os.rename(os.path.join(filepath, f), os.path.join(filepath, strpatt))
        # increment i
        i += 1

        # delete old file
        # os.remove(os.path.join(filepath, f))

    # if other image type, save as png and rename
    if f.endswith('.jpg') or f.endswith('.tif') or f.endswith('.bmp'):

        strpatt = 'st' + str(i).zfill(3) + '.png'

        # convert to png by opening then saving:
        img = Image.open(os.path.join(filepath, f))
        # g = f[:-4] + '.png'
        print('saving as {}/{}'.format(filepath, strpatt))
        img.save(os.path.join(filepath, strpatt))
        i += 1

        # delete old file   
        os.remove(os.path.join(filepath, f))




    

