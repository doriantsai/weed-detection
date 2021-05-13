#! /usr/bin/env python

# python script to rename all the files to a desired string pattern
# because I could not figure out the damn perl expression for use with the rename 
# function

import os
from subprocess import call

# read in all files in given folder
# find all matching given pattern
# rename all files of given pattern according to a different pattern

filepath = '/home/dorian/Data/SerratedTussockDataset_v1/Images'

files = os.listdir(filepath)
i = 0
for f in files:
    # check file extension
    if f.endswith('.png'):
        # rename file to st-###.jpg
        # pad with 3 zeros, incrementing i
        strpatt = 'lalalala' + str(i).zfill(3) + '.png'
        print(f + ' --> ' + strpatt)
        os.rename(os.path.join(filepath, f), os.path.join(filepath, strpatt))

        # imagemagick to convert from jpg to png

        # call(["magick "])
        # increment i
        i +=1


    

