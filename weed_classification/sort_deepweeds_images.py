#! /usr/bin/env python

"""
solrt deepweeds' images into runs/sequences based on filename time into different days
"""

# deepweed's images have filenames that have capture time encoded
# given directory of images
# read all image filenames
# sort them according to capture time
# plot capture times
# partition them based on

import os
import datetime
from dateutil import parser
import numpy as np
import pandas as pd

# img_dir = 'nonnegative_images'
img_dir = 'images'
img_list = os.listdir(img_dir)
img_list.sort()

img_name = img_list[0]
print(img_name)
time_str = img_name[:-6]
print(time_str)

# format = "%Y%m%d%-%H%M%S"
# dt_object = datetime.datetime.strptime(time_str, format)
dt = parser.parse(time_str)
print(dt)

# iterate over entire img_list
# obtain parsed datetime object for every image filename
dti = []
for img_name in img_list:
    time_str = img_name[:-6]
    dt = parser.parse(time_str)
    dti.append(dt)

# sort the list, which actually should already be sorted due to the filename structure
# a.sorted()
# determine how many runs there were in each year
dates = []
for dt in dti:
    dates.append(str(dt.date()))

print(dates)
dates_set = set(dates)
# since set only stores the same values once, this automatically gives us the unique dates
unique_dates = list(dates_set)
unique_dates.sort()

print('unique dates')
for i, ud in enumerate(unique_dates):
    print(f'{i}: {ud}')


# thus, 45 days of runs

# now we sort these into "runs"
# find all images that belong to a certain date
# choose a date:
print('all same dates')
j = 0
print(unique_dates[j])
# find all indices in dates that == unique_dates[0]
idx_same_date = [i for i in range(len(dates)) if dates[i] == unique_dates[j]]
# check by printing them out, should all be the same
# for i in idx_same_date:
#     print(f'{i}: {dates[i]}')

# for each unique date, find all corresponding indices
idx_per_date = []
len_idx = []
cum_img = 0
for j, u in enumerate(unique_dates):
    idx_same_date = [i for i in range(len(dates)) if dates[i] == u]
    idx_per_date.append(idx_same_date)
    len_idx.append(len(idx_same_date))
    cum_img += len(idx_same_date)
    cum_img_per = cum_img / len(img_list)
    print(f'u_idx: {j}, date: {u}, nimg = {len(idx_same_date)}, cum img = {cum_img}, cum % = {cum_img_per:.2f}')

print(f'number of unique dates = {len(unique_dates)}')
len_idx = np.array(len_idx)
print(f'sum of idx_per_date = {len_idx.sum()}')
print(f'length of list dir = {len(img_list)}')

# good, the lengths match
# next, we want to divide each into training/val/testing sets by some
# nearest percentage?

# choose a split ratio?
# data development vs deployment split 70/30
data_split = 0.7
# TODO automatically find uidx_deply, pt where cum % crosses data_split threshold
uidx_deploy = 38 # manually read from the above table where cum % crosses 70%
# everything before this date = for model development (train/val/test)
# everything after this date = for model deployment (see how bad we do)

# assemble a list of image filenames for development
# assemble remaining image filenames for deployment
dev_idx_list = []
dep_idx_list = []
for i in range(len(unique_dates)):
    if i <= uidx_deploy:
        dev_idx_list.extend(idx_per_date[i])
    else:
        dep_idx_list.extend(idx_per_date[i])


print(f'development: {len(dev_idx_list)}')
print(f'deployment: {len(dep_idx_list)}')
# 6050/2353

# convert idx to filenames via img_list
dev_img_list = [img_list[d] for d in dev_idx_list]
dep_img_list = [img_list[d] for d in dep_idx_list]

# read label from filename:
lbl_dir = 'labels'
labels_file = os.path.join(lbl_dir, 'labels.csv')
# TODO fixme, name[16] is not the label. Label must be found in the labels file
# read in labels file
# find all image names in labels file
data_frame = pd.read_csv(labels_file)
lbl_img_list = list(data_frame.iloc[:, 0])
lbl_lbl_list = list(data_frame.iloc[:, 1])

# find all image names that match the dev_img_list/dep_img_list
# TODO definitely be a more elegant/efficient way of solving this
# but for now, we just iterate
dev_label = []
for dev in dev_img_list:
    # find dev in lbl_img_list, note the index number
    idx = lbl_img_list.index(dev)
    dev_label.append(lbl_lbl_list[idx])

dep_label = []
for dep in dep_img_list:
    idx = lbl_img_list.index(dep)
    dep_label.append(lbl_lbl_list[idx])

# take the corresponding indexes and thus, their labels
# dev_label = [name[16] for name in dev_img_list]
# dep_label = [name[16] for name in dep_img_list]

# create dictionary for csv export
dev_dict = {'Filename': dev_img_list, 'Label': dev_label}
dep_dict = {'Filename': dep_img_list, 'Label': dep_label}

# export or save to .csv file
# read labels


dev_labels_file = 'sort_development_labels.csv'
dep_labels_file = 'sort_deployment_labels.csv'

# labels_file = os.path.join(labels_folder, 'nonnegative_labels.csv')
# labels = pd.read_csv(labels_file)
dv = pd.DataFrame(dev_dict)
dp = pd.DataFrame(dep_dict)
dv.to_csv(os.path.join(lbl_dir, dev_labels_file), index=False)
dp.to_csv(os.path.join(lbl_dir, dep_labels_file), index=False)

# further split deployment into individual days:
# uidx_deploy
for u in range(0, len(unique_dates)):
# for u in range(uidx_deploy, len(unique_dates)):
    # dep_name = 'deployment_day' + str(u) + '_img' + str(len(idx_per_date[u])) + '.csv'
    dep_name = 'labels_day' + '{0:02d}'.format(u) + '_img' + '{0:03d}'.format(len(idx_per_date[u])) + '.csv'
    dep_list = [img_list[d] for d in idx_per_date[u]]
    dep_label = []
    for d in dep_list:
        idx = lbl_img_list.index(d)
        dep_label.append(lbl_lbl_list[idx])
    # dep_label = [name[16] for name in dep_list]
    dep_dict = {'Filename': dep_list, 'Label': dep_label}
    dp = pd.DataFrame(dep_dict)
    dp.to_csv(os.path.join(lbl_dir, dep_name), index=False)

import code
code.interact(local=dict(globals(), **locals()))