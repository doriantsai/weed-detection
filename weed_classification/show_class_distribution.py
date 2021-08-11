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

# location of labels:
lbl_dir = 'labels'
# lbl_file = 'deployment_day38_img558.csv'
# lbl_file = 'development_labels.csv'
# lbl_file = 'deployment_labels.csv'
# lbl_file = 'nonnegative_labels.csv'
# lbl_filepath = os.path.join(lbl_dir, lbl_file)
# lbl_file = 'labels.csv' # includes negative images

# glob all label files with "nonneg_"
lbl_files = glob.glob('labels/nonneg_*.csv', recursive=False)
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
    # idx = np.arange(len(lbl_list))
    lbls = np.array(lbl_list)
    nclasses = len(CLASSES)

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    N, bins, patches = ax.hist(lbls, bins=CLASSES)

    for i, thispatch in enumerate(patches):
        thispatch.set_facecolor(CLASS_COLOURS[i])
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