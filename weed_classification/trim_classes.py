#! /usr/bin/env python

""" script to remove certain classes from deepweeds dataset """

# folder/file locations
# load csv file
# specify class to trime down by x amount
# randomly remove class
# save csv file

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

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


lbl_dir = 'labels'
lbl_file = 'development_labels.csv'
lbl_path = os.path.join(lbl_dir, lbl_file)

# read csv, generate list of image names, labels
lbls_df = pd.read_csv(lbl_path)
img_names = lbls_df.iloc[:, 0]
lbls = lbls_df.iloc[:, 1]

lbls_arr = np.array(lbls)
# show histogram to check
fig, ax = plt.subplots(1, 1, tight_layout=True)
N, bins, patches = ax.hist(lbls_arr, bins=np.arange(0, len(CLASSES)+1))

for i, thispatch in enumerate(patches):
    thispatch.set_facecolor(CLASS_COLOURS[i])
plt.xlabel('classes')
plt.ylabel('image count')
plt.title('class distribution: ' + lbl_file)
# plt.show()

# find all lbls that are == class_trim
lbls_by_class = []
nlbls_per_class = []
idxs_per_class  = []
for c in CLASSES:
    lbls_class = [lbl for lbl in lbls if lbl == c]

    lbls_by_class.append(lbls_class)
    nlbls_per_class.append(len(lbls_class))
    print(f'len of lbls_class({c}): {len(lbls_class)}')

    # the really useful set is the indices of the class
    idx_class = [i for i in range(len(lbls)) if lbls[i] == c]
    idxs_per_class.append(idx_class)



# specify class to trim/amount to remove
# class_trim = 7
target_class_nimages = min(nlbls_per_class[:-1]) # reduce current image count by this number, excluding the negative cases
print(f'target class nimages = {target_class_nimages}')
# for each class, randomly remove diff in classes
lbls_trim_by_class = []
idxs_trim_by_class = []

for i, c in enumerate(CLASSES): # here, we do not trim the negative class cases
    if i == CLASSES[-1]:
        # if negative class, append all of negative images
        idxs_trim_by_class.append(idxs_per_class[i])
    else:
        nelements_delete = nlbls_per_class[i] - target_class_nimages
        print(f'{i}: nelements delete = {nelements_delete}')
        to_delete = set(random.sample(range(len(lbls_by_class[i])), nelements_delete))
        lbls_trim_by_class = [x for k, x in enumerate(lbls_by_class[i]) if not k in to_delete]
        print(f'trimmed labels len: {len(lbls_trim_by_class)}')

        idx_trim_class = [x for k, x in enumerate(idxs_per_class[i]) if not k in to_delete]
        print(f'trimmed idx len: {len(idx_trim_class)}')
        idxs_trim_by_class.append(idx_trim_class)

    



# now combine all idxs into one long list, get labels and image names together
idxs = []
for i in range(len(CLASSES)):
    idxs.extend(idxs_trim_by_class[i])

# now save lbls_trim_by_class and accompanying image names to csv file
lbls_out = lbls_arr[idxs]
img_names_out = img_names[idxs]
print(len(idxs))
print(len(lbls))
print(len(lbls_out))

labels_file_out = 'development_labels_trim.csv'
data = {'Filename': img_names_out, 'Label': lbls_out}
lbl_df = pd.DataFrame(data)
lbl_df.to_csv(os.path.join(lbl_dir, labels_file_out), index=False)
# dep_df.to_csv(os.path.join(lbl_dir, dep_labels_file), index=False)


import code
code.interact(local=dict(globals(), **locals()))





