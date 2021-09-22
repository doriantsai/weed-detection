#! /usr/bin/env python

"""
show class distribution of given .csv file
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob


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

# set colors for histogram based on ones used in paper
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
CLASS_COLORS = [pink,
                blue,
                green,
                yellow,
                cyan,
                red,
                purple,
                orange,
                grey]

# location of labels:
lbl_dir = 'labels'
# lbl_file = 'deployment_day38_img558.csv'
# lbl_file = 'development_labels.csv'
# lbl_file = 'deployment_labels.csv'
# lbl_file = 'nonnegative_labels.csv'
# lbl_filepath = os.path.join(lbl_dir, lbl_file)
# lbl_file = 'labels.csv' # includes negative images

# glob all label files with "nonneg_"
# lbl_files = glob.glob('labels/nonneg_*.csv', recursive=False)
# lbl_files = glob.glob('labels/nonnegative_labels.csv', recursive=False)
# lbl_files = glob.glob('labels/development_labels_trim.csv', recursive=False)
lbl_files = glob.glob('labels/deployment_labels_trim.csv', recursive=False)
# lbl_files = glob.glob('labels/labels.csv', recursive=False)
# lbl_files = glob.glob('labels/labels_day*.csv', recursive=False)
lbl_files.sort()

# remove labels/ str from lbl_files:
# this is because we want to use lbl_files to name our output figures
for i, lbl in enumerate(lbl_files):
    lbl_files[i] = lbl[7:]

for lbl in lbl_files:
    print(lbl)

for lbl in lbl_files:
    print(f'lbl file: {lbl}')
    lbl_filepath = os.path.join(lbl_dir, lbl)
    data_frame = pd.read_csv(lbl_filepath)

    print(f'number of images in file: {len(data_frame)}')

    img_list = list(data_frame.iloc[:, 0])
    lbl_list = list(data_frame.iloc[:, 1])


    # convert to numpy arrays for histogram?
    lbls = np.array(lbl_list)

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    binboundary = np.arange(0, len(CLASSES)+1)
    N, bins, patches = ax.hist(lbls, bins=binboundary)


    for i, thispatch in enumerate(patches):
        thispatch.set_facecolor(CLASS_COLORS[i])
    plt.xlabel('classes')
    plt.ylabel('image count')
    plt.title('class distribution: ' + lbl)

    # plt.show()

    save_folder = os.path.join('output')
    os.makedirs(save_folder, exist_ok=True)
    save_img_name = lbl[:-4] + '_class_histogram.png'
    save_img_path = os.path.join(save_folder, save_img_name)
    plt.savefig(save_img_path)

    plt.close(fig)

    print(f'done saving figure: {save_img_path}')
    # save image for each given development/deployment file



import code
code.interact(local=dict(globals(), **locals()))