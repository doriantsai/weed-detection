#! usr/bin/env python

""" augment training data with image transforms """

# to compensate for low data, we can augment our training data with original images transformed
# from the classes defined in WeedDataset.py

# TODO

# make a new copy of the dataset images (eg Tussock_v2)
# split image data randomly

# set image directories
# set annotation files
# init WeedDataset transforms
# for each transform
    # apply to image
    # append to image_name the transform/degree of effect
    # save image into Image/Train folder
    # add info to annotations file

#
# do model comparison with two datasets