#! /usr/bin/env python

"""
script to add days together to cumulatively get ~800 images/class into development, and ~200 images/class into deployment
we will manually specify each day to add to each set of images
"""

import os
import matplotlib.pyplot as plt
from numpy.lib.npyio import save
import pandas as pd
import numpy as np
import glob
import random

# file/folder locations
# get per-day distributions from relevant .csv files
# specify which days go into which set
#   probably do this by... a list/an array?
# then combine all days into set, show set distribution
# save as single .csv file

# TODO also, add in negative images!
# in all, pos:neg, 1:2, try to maintain same scenario
# TODO curious to see if features would be more spread apart with negative images added in

def process_label_set(label_files, set_indices, lbl_dir='labels'):
    """ input all labels, indices for desired set, output array of set labels """

    set_lbl_days = [label_files[i] for i in set_indices]

    set_lbl_paths = [os.path.join(lbl_dir, d) for d in set_lbl_days]

    # for each set, combine day labels
    set_dataframe = pd.concat(map(pd.read_csv, set_lbl_paths), ignore_index=True)

    # for each set, load dataframe, extract img_list, lbl_list
    # dev_img_list = list(dev_dataframe.iloc[:, 0])
    set_lbl_list = list(set_dataframe.iloc[:, 1])

    set_lbl_arr = np.array(set_lbl_list)

    return set_lbl_arr, set_dataframe

########### classes #############
CLASS_NAMES = ('Chinee apple',
                'Lantana',
                'Parkinsonia',
                'Parthenium',
                'Prickly acacia',
                'Rubber vine',
                'Siam weed',
                'Snake weed',
                'Negative')
CLASSES = np.arange(0, len(CLASS_NAMES))
CLASS_DICT = {i: CLASS_NAMES[i] for i in range(0, len(CLASSES))}

# set colours for histogram based on ones used in paper
    # RGB
pink = np.r_[255, 105,180]/255
blue = np.r_[0, 0, 255]/255
green = np.r_[0, 255, 0]/255
yellow = np.r_[255, 255, 0]/255
cyan = np.r_[0, 255, 255]/255
red = np.r_[255, 0, 0]/255
purple = np.r_[135, 0, 135]/255
orange = np.r_[255, 127, 80]/255
grey = np.r_[100, 100, 100]/255
CLASS_COLOURS = [pink,
                blue,
                green,
                yellow,
                cyan,
                red,
                purple,
                orange,
                grey]


# find label files
lbl_dir = 'labels'
# glob all label files with "nonneg_"
# lbl_files = glob.glob('labels/nonneg_*.csv', recursive=False)
lbl_files = glob.glob('labels/labels_day*.csv', recursive=False)
lbl_files.sort()

# check which lbl_files have been picked up
for lbl in lbl_files:
    print(lbl)

# remove labels/ str from lbl_files:
# this is because we want to use lbl_files to name our output figures
for i, lbl in enumerate(lbl_files):
    lbl_files[i] = lbl[7:]

# specify which lbl files go into which set
# TODO  ensure there is no overlap
# --------------------------------------------------------------------------------
# nonnegative day set
# dev_days = [2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 19,
#             21, 24, 25, 27, 28, 29, 31,
#             32, 33, 34, 35, 37, 38, 41, 42]
# --------------------------------------------------------------------------------
# dep_days = [0, 1, 8, 14, 16, 17, 18, 20, 22, 23, 26, 30, 36, 39, 40, 43, 44, 45]
# --------------------------------------------------------------------------------
# negative day set without many negative days
dev_days = [2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            34, 35, 36, 38, 39, 41, 42, 43]
# --------------------------------------------------------------------------------
dep_days = [0, 1, 6, 8, 14, 15, 16, 17, 18, 32, 33, 37, 40, 44, 45]


# process each label set (create dataframe, provide array of the days/classes)
dev_arr, dev_df = process_label_set(lbl_files, dev_days)
dep_arr, dep_df = process_label_set(lbl_files, dep_days)

# # for each set, show histogram in side-by-side mode?
fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)
binboundary = np.arange(0, len(CLASSES)+1)
N1, bins1, patches1 = ax1.hist(dev_arr, bins=binboundary)
N2, bins2, patches2 = ax2.hist(dep_arr, bins=binboundary)

for i, thispatch in enumerate(patches1):
    thispatch.set_facecolor(CLASS_COLOURS[i])

ax1.set_xlabel('classes')
ax1.set_ylabel('image count')
ax1.set_title('class distribution: development')

for i, thispatch in enumerate(patches2):
    thispatch.set_facecolor(CLASS_COLOURS[i])
ax2.set_xlabel('classes')
ax2.set_ylabel('image count')
ax2.set_title('class distribution: deployment')

save_folder = os.path.join('output')
os.makedirs(save_folder, exist_ok=True)
save_img_name = 'dev_vs_dep_days_combined_class_histogram_before_trim_noneg.png'
save_img_path = os.path.join(save_folder, save_img_name)
plt.savefig(save_img_path)
# plt.show()
plt.close(fig)

# TODO add rest of negative days - might need to randomly distribute them
# we don't know which day was from where, so take all negative class data and distribute them randomly/evenly to both dev/dep
# compute the percentage of class images in dev vs dep, use this for the negative image split between dev dep
# randomly split negative images for dev and dep, then append those images to dev/dep

# get number of images in entire dataset
lbl_all = os.path.join(lbl_dir, 'labels.csv')
lbl_all_df = pd.read_csv(lbl_all)
nlbl_all = len(lbl_all_df)

lbl_pos = os.path.join(lbl_dir, 'nonnegative_labels.csv')
lbl_pos_df = pd.read_csv(lbl_pos)
npos = len(lbl_pos_df)
ndev = len(dev_df)
ndep = len(dep_df)

lbl_neg = os.path.join(lbl_dir, 'negative_labels.csv')
lbl_neg_df = pd.read_csv(lbl_neg)
nneg = len(lbl_neg_df)

nneg_dev = round((ndev / npos) * nneg)
nneg_dep = nneg - nneg_dev

print(f'num of images in all labels: {nlbl_all}')
print(f'num of images in all positive image labels: {npos}')
print(f'num of images in all negative image labels: {nneg}')
print(f'num of images in dev set: {ndev}')
print(f'num of images in dep set: {ndep}')
print(f'ratio of ndev/npos = {ndev / npos}')
print(f'number of negative images to the dev set: {nneg_dev}')
print(f'number of negative images to the dep set: {nneg_dep}')

# do random split 
idx = np.arange(nneg)
idx_dev = set(random.sample(range(nneg), nneg_dev))
idx_dep = [x for x in idx if not x in idx_dev]
idx_dev = list(idx_dev)

# append indices of image names to dev and dep sets
# NOTE consider making this append images to image set

# neg image names
dev_img_names = list(lbl_neg_df.iloc[idx_dev, 0])
dep_img_names = list(lbl_neg_df.iloc[idx_dep, 0])
dev_img_class = list(lbl_neg_df.iloc[idx_dev, 1])  # should all just be negative cases, so could just make an array all 8's
dep_img_class = list(lbl_neg_df.iloc[idx_dep, 1])

# combine into single df
dev_neg_df = pd.DataFrame({'Filename': dev_img_names, 'Label': dev_img_class})
dep_neg_df = pd.DataFrame({'Filename': dep_img_names, 'Label': dep_img_class})
dev_df_ex = pd.concat([dev_df, dev_neg_df], ignore_index=True)
dep_df_ex = pd.concat([dep_df, dep_neg_df], ignore_index=True)

# output array for histograms
dev_ex_arr = np.array(dev_df_ex.iloc[:, 1])
dep_ex_arr = np.array(dep_df_ex.iloc[:, 1])

# for each set, show histogram in side-by-side mode?
fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)
binboundary = np.arange(0, len(CLASSES)+1)
N1, bins1, patches1 = ax1.hist(dev_ex_arr, bins=binboundary)
N2, bins2, patches2 = ax2.hist(dep_ex_arr, bins=binboundary)

for i, thispatch in enumerate(patches1):
    thispatch.set_facecolor(CLASS_COLOURS[i])

ax1.set_xlabel('classes')
ax1.set_ylabel('image count')
ax1.set_title('class distribution: development')

for i, thispatch in enumerate(patches2):
    thispatch.set_facecolor(CLASS_COLOURS[i])
ax2.set_xlabel('classes')
ax2.set_ylabel('image count')
ax2.set_title('class distribution: deployment')

save_folder = os.path.join('output')
os.makedirs(save_folder, exist_ok=True)
save_img_name = 'dev_vs_dep_days_combined_class_histogram_before_trim_with_neg.png'
save_img_path = os.path.join(save_folder, save_img_name)
plt.savefig(save_img_path)
# plt.show()
plt.close(fig)

# TODO save dev, dep as new label sets
dev_labels_file = 'development_labels.csv'
dep_labels_file = 'deployment_labels.csv'
dev_df.to_csv(os.path.join(lbl_dir, dev_labels_file), index=False)
dep_df.to_csv(os.path.join(lbl_dir, dep_labels_file), index=False)


import code
code.interact(local=dict(globals(), **locals()))