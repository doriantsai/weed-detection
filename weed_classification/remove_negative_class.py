#! /usr/bin/env python

import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import shutil

# trim DeepWeeds dataset by removing the negative class




CLASSES = (0, 1, 2, 3, 4, 5, 6, 7, 8)
CLASS_NAMES = ('Chinee apple', 
                'Lantana', 
                'Parkinsonia', 
                'Parthenium',
                'Prickly acacia', 
                'Rubber vine', 
                'Siam weed', 
                'Snake weed', 
                'Negative')
CLASS_DICT = {i: CLASS_NAMES[i] for i in range(0, len(CLASSES))}


class DeepWeedsDataset(Dataset):
    """ Deep weeds dataset """

    def __init__(self, csv_file, root_dir, transform=None):
        # csv_file (string): Path to csv file with labels
        # root_dir (string): Directory with all images
        # transform (callable, opt): Transform to be applied to sample

        self.weed_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.weed_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.weed_frame.iloc[idx, 0])
        image = plt.imread(img_name)
        label = self.weed_frame.iloc[idx, 1]

        # make sample into a dictionary
        sample = {'image': image, 'label': label, 'image_name': self.weed_frame.iloc[idx, 0]}

        # apply transform
        if self.transform:
            sample = self.transform(sample)

        return sample


# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    # edit the dataset - remove the negative class:
    # go through the csv file, make a new csv file with no labels from the negative class
    # save the corresponding images
    # copy over new set of images to a new folder?    
    images_folder = './images'
    labels_folder = './labels'

    # read labels
    labels_file = os.path.join(labels_folder, 'labels.csv')
    images_folder = './images'

    fulldataset = DeepWeedsDataset(csv_file=labels_file,
                                    root_dir=images_folder)

    # iterate through dataset samples to find/remove negative_class
    negative_class = 8
    img_name = []
    label = []
    positive_dataset_folder = './nonnegative_images'
    positive_csv_file = os.path.join(labels_folder, 'nonnegative_labels.csv')

    # if files exist in the folder, remove the folder first
    # then makedir
    if os.path.isdir(positive_dataset_folder):
        print('removing existing {} folder'.format(positive_dataset_folder))
        shutil.rmtree(positive_dataset_folder)

    os.mkdir(positive_dataset_folder)

    neg_label = []
    neg_name = []
    for i in range(len(fulldataset)):
        sample = fulldataset[i]
        img, lbl, name = sample['image'], sample['label'], sample['image_name']

        if i % 1000 == 999:
            print('Through dataset index: {}'.format(i))

        if not lbl == negative_class: # the negative class, which we want to remove
            # save image to folder
            plt.imsave(os.path.join(positive_dataset_folder, name), img)

            # save label to list
            label.append(lbl)
            img_name.append(name)
        else:
            neg_label.append(lbl)
            neg_name.append(name)

    print('finished iterating through dataset')
    print('initial number of images: {}'.format(len(fulldataset)))
    print('final number of nonnegative class images: {}'.format(len(label)))

    # import code
    # code.interact(local=dict(globals(), **locals()))  

    # save lists to csv file
    outfile = pd.DataFrame({'Filename': img_name, 'Label': label})
    outfile.to_csv(positive_csv_file, index=False)

    negative_csv_file = os.path.join(labels_folder, 'negative_labels.csv')
    neg_df = pd.DataFrame({'Filename': neg_name, 'Label': neg_label})
    neg_df.to_csv(negative_csv_file, index=False)

    print('creating nonnegative dataset complete')
    # import code
    # code.interact(local=dict(globals(), **locals())) 
    # 
      