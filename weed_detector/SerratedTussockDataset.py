#! /usr/bin/env python

import os
import numpy as np
import torch
import torch.utils.data
import json
import matplotlib.pyplot as plt
import torchvision.transforms as T
import random

# from skimage import transform as sktrans
from PIL import Image
from torchvision.transforms import functional as tvtransfunc
from torchvision.datasets.video_utils import VideoClips


class SerratedTussockDataset(object):
    """ Serrated tussock dataset """


    def __init__(self, root_dir, json_file, transforms):
        """
        initialise the dataset
        annotations - json file of annotations of a prescribed format
        root_dir - root directory of dataset, under which are Images and Annotations folder
        transforms - list of transforms randomly applied to dataset for training
        """

        annotations = json.load(open(os.path.join(root_dir, json_file)))
        self.annotations = list(annotations.values())
        self.root_dir = root_dir
        self.transforms = transforms


    def __getitem__(self, idx):
        """
        given an index, return the corresponding image and sample from the dataset
        converts images and corresponding sample to tensors
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image
        img_name = os.path.join(self.root_dir, 'Images', self.annotations[idx]['filename'])
        image =  Image.open(img_name).convert("RGB")

        # number of bboxes
        nobj = len(self.annotations[idx]['regions'])

        # get bbox
        # bounding box is read in a xmin, ymin, width and height
        # bounding box is saved as xmin, ymin, xmax, ymax
        boxes = []
        for i in range(nobj):
            xmin = self.annotations[idx]['regions'][str(i)]['shape_attributes']['x']
            ymin = self.annotations[idx]['regions'][str(i)]['shape_attributes']['y']
            width = self.annotations[idx]['regions'][str(i)]['shape_attributes']['width']
            height = self.annotations[idx]['regions'][str(i)]['shape_attributes']['height']
            xmax = xmin + width
            ymax = ymin + height
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float64)

        # compute area
        if len(boxes) == 0:
            # no boxes
            area = 0
        else:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float64)

        # only one class + background:
        labels = torch.ones((nobj,), dtype=torch.int64)

        # TODO iscrowd?
        iscrowd = torch.zeros((nobj,), dtype=torch.int64)

        # image_id is the index of the image in the folder
        # TODO should test this
        image_id = torch.tensor([idx], dtype=torch.int64)

        sample = {}
        sample['boxes'] = boxes
        sample['labels'] = labels
        sample['image_id'] = image_id
        sample['area'] = area
        sample['iscrowd'] = iscrowd

        # apply transforms to image and sample
        if self.transforms:
            image, sample = self.transforms(image, sample)

        return image, sample


    def __len__(self):
        """
        return the number of images in the entire dataset
        """
        return len(self.annotations)


    def set_transform(self, tforms):
        """
        set the transforms
        """
        # TODO assert for valid input
        # tforms must be callable and operate on an image
        self.transforms = tforms


class Compose(object):
    """ Compose for set of transforms """


    def __init__(self, transforms):
        self.transforms = transforms


    def __call__(self, image, target):
        # NOTE this is done because the built-in PyTorch Compose transforms function
        # only accepts a single (image/tensor) input. To operate on both the image
        #  as well as the sample/target, we need a custom Compose transform function
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Rescale(object):
    """ Rescale image to given size """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image, sample=None):

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
        img = T.Resize((new_w, new_h))(image)

        # apply transform to bbox as well
        if sample is not None:
            xChange = float(new_w) / float(w)
            yChange = float(new_h) / float(h)
            bbox = sample["boxes"]  # [xmin ymin xmax ymax]

            if len(bbox) > 0:
                bbox[:, 0] = bbox[:, 0] * yChange
                bbox[:, 1] = bbox[:, 1] * xChange
                bbox[:, 2] = bbox[:, 2] * yChange
                bbox[:, 3] = bbox[:, 3] * xChange
                sample["boxes"] = np.float64(bbox)

            return img, sample
        else:
            return img


class ToTensor(object):
    """ convert ndarray to sample in Tensors """

    def __call__(self, image, sample):
        """ convert image and sample to tensors """

        # convert image
        image = tvtransfunc.to_tensor(image)

        # convert samples
        boxes = sample['boxes']
        if not torch.is_tensor(boxes):
            boxes = torch.from_numpy(boxes)
        sample['boxes'] = boxes

        return image, sample


class RandomHorizontalFlip(object):
    """ Random horozintal flip """

    def __init__(self, prob):
        """ probability of a horizontal image flip """

        self.prob = prob


    def __call__(self, image, sample):
        """ apply horizontal image flip to image and sample """

        if random.random() < self.prob:
            w, h = image.size[:2]
            # flip image
            image = image.transpose(method=Image.FLIP_LEFT_RIGHT)

            # flip bbox
            bbox = sample['boxes']

            # bounding box is saved as xmin, ymin, xmax, ymax
            # only changing xmin and xmax
            if len(bbox) > 0:
                bbox[:, [0, 2]] = w - bbox[:, [2, 0]]  # note the indices switch (must flip the box as well!)
                sample['boxes'] = bbox

        return image, sample


class Blur(object):
    """ Gaussian blur images """

    def __init__(self, kernel_size=3, sigma=(0.1, 2.0)):
        """ kernel size and standard deviation (sigma) of Gaussian blur """
        self.kernel_size = kernel_size
        self.sigma = sigma


    def __call__(self, image, sample):
        """ apply blur to image """
        image = tvtransfunc.gaussian_blur(image, self.kernel_size, self.sigma)

        return image, sample