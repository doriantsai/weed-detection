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
CLASSES = (0, 1, 2, 3, 4, 5, 6, 7)
CLASS_NAMES = ('Chinee apple',
                'Lantana',
                'Parkinsonia',
                'Parthenium',
                'Prickly acacia',
                'Rubber vine',
                'Siam weed',
                'Snake weed')
CLASS_DICT = {i: CLASS_NAMES[i] for i in range(0, len(CLASSES))}

# set colours for histogram based on ones used in paper
    # RGB
pink = np.r_[255, 105,180]/255
blue = np.r_[0, 0, 255]/255
green = np.r_[0, 255, 0]/255
yellow = np.r_[255, 255, 0]/255
cyan = np.r_[0, 255, 255]/255
red = np.r_[255, 0, 0]/255
purple = np.r_[255, 0, 255]/255
orange = np.r_[255, 127, 80]/255
CLASS_COLOURS = [pink,
                blue,
                green,
                yellow,
                cyan,
                red,
                purple,
                orange]


# find label files
lbl_dir = 'labels'
# glob all label files with "nonneg_"
lbl_files = glob.glob('labels/nonneg_*.csv', recursive=False)
lbl_files.sort()

# remove labels/ str from lbl_files:
# this is because we want to use lbl_files to name our output figures
for i, lbl in enumerate(lbl_files):
    lbl_files[i] = lbl[7:]

# specify which lbl files go into which set
# TODO  ensure there is no overlap
dev_days = [2, 3, 5, 6, 7, 9, 10, 12, 13, 15,
            24, 25, 28, 29, 31, 
            32, 33, 34, 35, 37, 38, 41, 42]

dep_days = [1, 8, 14, 16, 17, 18, 20, 22, 23, 26, 30, 36, 39, 40, 44, 45]

# NOTE removed days: 0, 4, 11, 19, 21,  43, 27


# dep_lbl_days = [lbl_files[i] for i in dep_days]

dev_arr, dev_df = process_label_set(lbl_files, dev_days)
dep_arr, dep_df = process_label_set(lbl_files, dep_days)

# for d in dev_lbl_days:
#     lbl_filepath = os.path.join(lbl_dir, d)
#     lbl_dataframe = pd.read_csv(lbl_filepath)
#     dev_dataframe.append(lbl_dataframe)

# dep_dataframe = pd.DataFrame()
# for d in dep_lbl_days:
#     lbl_filepath = os.path.join(lbl_dir, d)
#     lbl_dataframe = pd.read_csv(lbl_filepath)
#     dep_dataframe.append(lbl_dataframe)

# TODO maybe make a function on how we treat each set

# import code
# code.interact(local=dict(globals(), **locals()))



# for each set, show histogram in side-by-side mode?
fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)
N1, bins1, patches1 = ax1.hist(dev_arr, bins=CLASSES)
N2, bins2, patches2 = ax2.hist(dep_arr, bins=CLASSES)

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
save_img_name = 'dev_vs_dep_days_combined_class_histogram_before_trim.png'
save_img_path = os.path.join(save_folder, save_img_name)
plt.savefig(save_img_path)
plt.show()
plt.close(fig)

# TODO save dev, dep as new label sets
dev_labels_file = 'development_labels.csv'
dep_labels_file = 'deployment_labels.csv'
dev_df.to_csv(os.path.join(lbl_dir, dev_labels_file), index=False)
dep_df.to_csv(os.path.join(lbl_dir, dep_labels_file), index=False)


import code
code.interact(local=dict(globals(), **locals()))