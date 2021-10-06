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
import random
from PIL import Image
from torchvision.transforms import functional as tvtransfunc


class DeepWeedsDataset(Dataset):
    """ Deep weeds dataset """

    def __init__(self, csv_file, root_dir, transform=None):
        # csv_file (string): Path to csv file with labels
        # root_dir (string): Directory with all images
        # transform (callable, opt): Transform to be applied to sample
        print(csv_file)
        self.weed_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.weed_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.get_image_name(idx)
        image = Image.open(img_name)
        label = self.weed_frame.iloc[idx, 1]

        # make sample into a dictionary
        sample = {'image': image, 'label': label, 'image_id': idx}

        # apply transform
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image_name(self, idx):
        return os.path.join(self.root_dir, self.weed_frame.iloc[idx, 0])

    # getter/setter for transforms
    def set_transform(self, transform=None):
        self.transform = transform

    def get_transform(self):
        return self.transform


class Rescale(object):
    """ Rescale image to given size """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        # open up the dict
        image, label, id = sample['image'], sample['label'], sample['image_id']

        # handle the aspect ratio
        h, w = image.size[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        # do the transform
        img = transforms.Resize((new_w, new_h))(image)

        # return as a dictionary, as before
        return {'image': img, 'label': label, 'image_id': id}


class RandomRotate(object):
    """ randomly rotate object """


    def __init__(self,  degrees):
        self._degrees = degrees

    def __call__(self, sample):
        image, label, id = sample['image'], sample['label'], sample['image_id']
        Rotation = transforms.RandomRotation(degrees=self._degrees)
        img = Rotation(image)
        return {'image': img, 'label': label, 'image_id': id}


class RandomResizedCrop(object):
    """ Randomly crop image """

    # def __init__(self, output_size):
    #     assert isinstance(output_size, (int, tuple))
    #     if isinstance(output_size, int):
    #         self.output_size = (output_size, output_size)
    #     else:
    #         assert len(output_size) == 2
    #         self.output_size = output_size
    # def __call__(self, sample):
    #     # unpack the dictionary
    #     # get h,w
    #     # new h,w from output size
    #     # randomly crop from top/left
    #     # return image in sample
    #     image, label, id = sample['image'], sample['label'], sample['image_id']

    #     h, w = image.shape[:2]
    #     new_h, new_w = self.output_size

    #     top, left = np.random.randint(0, h - new_h), np.random.randint(0, w - new_w)
    #     image = image[top:top + new_h, left:left + new_w]

    #     return {'image': image, 'label': label, 'image_id': id}

    def __init__(self, size=(224, 224), scale=(0.5, 1.0)):

        self._size = size
        self._scale = scale

    def __call__(self, sample):
        image, label, id = sample['image'], sample['label'], sample['image_id']
        ResizeCrop = transforms.RandomResizedCrop(self._size, self._scale)
        img = ResizeCrop(image)
        return {'image': img, 'label': label, 'image_id': id}


class RandomColorJitter(object):
    """ randomly jitter color """

    def __init__(self, brightness=0, hue=0, contrast=0, saturation=0):
        self._brightness = brightness
        self._hue = hue
        self._contrast = contrast
        self._saturation = saturation

    def __call__(self, sample):
        image, label, id = sample['image'], sample['label'], sample['image_id']
        ColorJitter = transforms.ColorJitter(brightness=self._brightness,
                                             contrast=self._contrast,
                                             saturation=self._saturation,
                                             hue=self._hue)
        img = ColorJitter(image)
        return {'image': img, 'label': label, 'image_id': id}


class RandomAffine(object):
    """ randomly transform with affine transformation """

    def __init__(self, degrees=5, translate=(0.05, 0.05)):
        self._degrees = degrees
        self._translate = translate

    def __call__(self, sample):
        image, label, id = sample['image'], sample['label'], sample['image_id']
        Affine = transforms.RandomAffine(self._degrees, self._translate)
        img = Affine(image)
        return {'image': img, 'label': label, 'image_id': id}



class RandomHorizontalFlip(object):
    """ randomly apply horizontal flip to image """

    def __init__(self, prob=0.5):
        self._prob = prob

    def __call__(self, sample):
        image, label, id = sample['image'], sample['label'], sample['image_id']
        Flip = transforms.RandomHorizontalFlip(self._prob)
        img = Flip(image)
        return {'image': img, 'label': label, 'image_id': id}


class RandomPixelIntensityScaling(object):
    """ randomly scale pixel intensities in an image """

    def __init__(self, scale_min=0.75, scale_max=1.25):
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        id = sample['image_id']

        # TODO we have a PIL image
        # if prob:
        # do transform
        # randomly choose a scale value between scale_min/max
        # multiply entire image intensities by scale
        # save to image sample/return image sample
        # else
        # just return normal
        # if random.random() >= self.prob:


            # random scale factor between the two min/max values
        scale = random.uniform(self.scale_min, self.scale_max)
        # scale image data
        img = tvtransfunc.adjust_brightness(img, scale)
        # img = np.array(img, np.float32) * float(scale)
        # # fix min/max, since
        # img = Image.fromarray(img)
        return {'image': img, 'label': label, 'image_id': id}


# TODO rotation, then  scaled horz/vert, each color channel shifted
# can't we just call torchvision.transforms.RandomX?
# class RandomHorizontalFlip(object):
#     """ random horizontal flip """
#     def __init__(self, prob):
#         """ probability of horizontal image flip """
#         self.prob = prob

# class RandomRotate(object):
#     """ randomly rotate image """
#     def __init__(self, prob):
#         self.prob = prob

# class RandomColourJitter(object):
#     """ randomly shift pixel intensity, color"""
#     # sere torchvision.transforms.ColorJitter(brightness)
#     def __init__(self, prob):
#         self.prob = prob

# class RandomPerspective(object):
#     """ randomly apply affine transformation for perspective shift """
#     # see RandomAffine
#     def ___init__(self, prob):
#         self.prob = prob



class ToTensor(object):
    """ convert ndarray to sample in Tensors """
    def __call__(self, sample):
        image, label, id = sample['image'], sample['label'], sample['image_id']

        # image = image.transpose((2, 0, 1))
        # return {'image': torch.from_numpy(image), 'label': label, 'image_id': id}
        image = tvtransfunc.to_tensor(image)
        image = torch.as_tensor(image, dtype=torch.float32)

        return {'image': image, 'label': label, 'image_id': id}


class Compose(object):
    """ Compose for set of transforms """


    def __init__(self, transforms):
        self.transforms = transforms


    def __call__(self, sample):
        # NOTE this is done because the built-in PyTorch Compose transforms function
        # only accepts a single (image/tensor) input. To operate on both the image
        #  as well as the sample/target, we need a custom Compose transform function
        for t in self.transforms:
            sample= t(sample)
        return sample



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

CLASS_COLORS = ['pink',
            'blue',
            'green',
            'yellow',
            'cyan',
            'red',
            'purple',
            'orange',
            'grey']

pink = np.r_[255, 105, 180]/255
blue = np.r_[0, 0, 255]/255
green = np.r_[0, 255, 0]/255
yellow = np.r_[255, 255, 0]/255
cyan = np.r_[0, 255, 255]/255
red = np.r_[255, 0, 0]/255
purple = np.r_[135, 0, 135]/255
orange = np.r_[255, 127, 80]/255
grey = np.r_[175, 175, 175]/255
CLASS_COLOR_ARRAY = [pink,
                      blue,
                      green,
                      yellow,
                      cyan,
                      red,
                      purple,
                      orange,
                      grey]

