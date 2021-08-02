#! /usr/bin/env python

"""
show class distribution of given .csv file
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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


# location of labels:
lbl_dir = 'labels'
# lbl_file = 'deployment_day38_img558.csv'
lbl_file = 'development_labels.csv'
# lbl_file = 'deployment_labels.csv'
# lbl_file = 'nonnegative_labels.csv'
lbl_filepath = os.path.join(lbl_dir, lbl_file)

data_frame = pd.read_csv(lbl_filepath)

print(f'number of images in file: {len(data_frame)}')

img_list = list(data_frame.iloc[:, 0])
lbl_list = list(data_frame.iloc[:, 1])


# convert to numpy arrays for histogram?
idx = np.arange(len(lbl_list))
lbls = np.array(lbl_list)
nclasses = len(CLASSES)

fig, ax = plt.subplots(1, 1, tight_layout=True)
N, bins, patches = ax.hist(lbls, bins=CLASSES)

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

for i, thispatch in enumerate(patches):
    thispatch.set_facecolor(CLASS_COLOURS[i])
plt.xlabel('classes')
plt.ylabel('image count')
plt.title('class distribution: ' + lbl_file)

plt.show()
#   self.weed_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.weed_frame)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         img_name = os.path.join(self.root_dir, self.weed_frame.iloc[idx, 0])
#         image = plt.imread(img_name)
#         label = self.weed_frame.iloc[idx, 1]

import code
code.interact(local=dict(globals(), **locals()))