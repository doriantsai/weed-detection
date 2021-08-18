#! /usr/bin/env python

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from skimage import io, transform
import pandas as pd
import torch


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
        sample = {'image': image, 'label': label, 'image_id': idx}

        # apply transform
        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """ Rescale image to given size """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        # open up the dict
        image, label, id = sample['image'], sample['label'], sample['image_id']

        # handle the aspect ratio
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        # do the transform
        img = transform.resize(image, (new_h, new_w))

        # return as a dictionary, as before
        return {'image': img, 'label': label, 'image_id': id}


class RandomCrop(object):
    """ Randomly crop image """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    def __call__(self, sample):
        # unpack the dictionary
        # get h,w
        # new h,w from output size
        # randomly crop from top/left
        # return image in sample
        image, label, id = sample['image'], sample['label'], sample['image_id']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top, left = np.random.randint(0, h - new_h), np.random.randint(0, w - new_w)
        image = image[top:top + new_h, left:left + new_w]

        return {'image': image, 'label': label, 'image_id': id}


class ToTensor(object):
    """ convert ndarray to sample in Tensors """
    def __call__(self, sample):
        image, label, id = sample['image'], sample['label'], sample['image_id']

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'label': label, 'image_id': id}

########### classes #############
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
