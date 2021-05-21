#! /usr/bin/env python

"""
weed dataset object and associated transforms as classes
"""

import os
import numpy as np
import torch
import torch.utils.data
import json
import matplotlib.pyplot as plt
from torchvision.models.detection.rpn import RegionProposalNetwork
import torchvision.transforms as T
import random

# from skimage import transform as sktrans
from PIL import Image
from torchvision.transforms import functional as tvtransfunc
from torchvision.datasets.video_utils import VideoClips


class WeedDataset(object):
    """ weed dataset object """


    # TODO maybe should actually hold the datasets/dataloader objects?
    def __init__(self, root_dir, json_file, transforms, img_dir=None):
        """
        initialise the dataset
        annotations - json file of annotations of a prescribed format
        root_dir - root directory of dataset, under which are Images/All/Test/Train folders
        transforms - list of transforms randomly applied to dataset for training
        """

        # TODO annotations shjould have root_dir/Annotations/json_file
        annotations = json.load(open(os.path.join(root_dir, 'Annotations', json_file)))
        self.annotations = list(annotations.values())
        self.root_dir = root_dir
        self.transforms = transforms

        if img_dir is not None:
            self.img_dir = img_dir
        else:
            self.img_dir = root_dir


    def __getitem__(self, idx):
        """
        given an index, return the corresponding image and sample from the dataset
        converts images and corresponding sample to tensors
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image
        img_name = os.path.join(self.img_dir, self.annotations[idx]['filename'])
        image =  Image.open(img_name).convert("RGB")

        # number of bboxes
        nobj = len(self.annotations[idx]['regions'])

        # get bbox
        # bounding box is read in a xmin, ymin, width and height
        # bounding box is saved as xmin, ymin, xmax, ymax
        boxes = []

        # import code
        # code.interact(local=dict(globals(), **locals()))

        if nobj > 0:
            for i in range(nobj):
                if isinstance(self.annotations[idx]['regions'], dict):
                    j = str(i)
                else:  # regions is a list type
                    j = i
                xmin = self.annotations[idx]['regions'][j]['shape_attributes']['x']
                ymin = self.annotations[idx]['regions'][j]['shape_attributes']['y']
                width = self.annotations[idx]['regions'][j]['shape_attributes']['width']
                height = self.annotations[idx]['regions'][j]['shape_attributes']['height']
                xmax = xmin + width
                ymax = ymin + height
                boxes.append([xmin, ymin, xmax, ymax])

        if nobj == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float64)


        # compute area
        if nobj == 0:
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


class RandomBlur(object):
    """ Gaussian blur images """

    def __init__(self, kernel_size=3, sigma=(0.1, 2.0)):
        """ kernel size and standard deviation (sigma) of Gaussian blur """
        # kernel must be an odd number
        self.kernel_size = kernel_size
        self.sigma = sigma


    def __call__(self, image, sample):
        """ apply blur to image """
        image = tvtransfunc.gaussian_blur(image, self.kernel_size, self.sigma)
        return image, sample


class RandomVerticalFlip(object):
    """ Random vertical flip """

    def __init__(self, prob):
        """ probability of a vertical image flip """
        self.prob = prob

    def __call__(self, image, sample):
        """ apply a vertical image flip to image and sample """

        if random.random() < self.prob:
            w, h = image.size[:2]
            # flip image
            image = image.transpose(method=Image.FLIP_TOP_BOTTOM)

            # flip bbox [xmin, ymin, xmax, ymax]
            bbox = sample['boxes']
            if len(bbox) > 0:
                # bbox[:, [0, 2]] = w - bbox[:, [2, 0]]
                bbox[:, [1, 3]] = h - bbox[:, [3, 1]]
                sample['boxes'] = bbox
        return image, sample


class RandomBrightness(object):
    """ Random color jitter transform """

    def __init__(self,
                 prob,
                 brightness=0):
        self.prob = prob
        # check brightness single non-negative 0 gives a black image, 1
        # gives the original image while 2 increases the brightness by a factor
        # of 2
        self.brightness = brightness

    def __call__(self, image, sample):
        """ apply change in brightnes/constrast/saturation/hue """

        if random.random() < self.prob:
            image = tvtransfunc.adjust_brightness(image, self.brightness)
        return image, sample


class RandomContrast(object):
    """ Random constrast jitter transform """

    def __init__(self,
                 prob,
                 contrast=0):
        self.prob = prob
        # Can be any non negative number. 0 gives a solid gray image, 1
        # gives the original image while 2 increases the contrast by a factor of
        # 2.
        self.contrast=contrast

    def __call__(self, image, sample):
        """ apply change in brightnes/constrast/saturation/hue """

        if random.random() < self.prob:
            image = tvtransfunc.adjust_contrast(image, self.contrast)
        return image, sample


class RandomHue(object):
    """ Random hue jitter transform """

    def __init__(self,
                 prob,
                 hue=0):
        self.prob = prob
        # hue is a single number ranging from
        # Should be in [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of
        # hue channel in HSV space in positive and negative direction
        # respectively. 0 means no shift. Therefore, both -0.5 and 0.5 will give
        # an image with complementary colors while 0 gives the original image.
        self.hue = hue

    def __call__(self, image, sample):
        """ apply change in brightnes/constrast/saturation/hue """

        if random.random() < self.prob:
            image = tvtransfunc.adjust_hue(image, self.hue)
        return image, sample


class RandomSaturation(object):
    """ Random saturation """

    def __init__(self, prob, saturation=0):
        self.prob = prob
        # 0 will give a black and white image, 1 will give the original
        # image while 2 will enhance the saturation by a factor of 2.
        self.saturation = saturation

    def __call__(self, image, sample):
        """ apply change in saturation """
        if random.random() < self.prob:
            image = tvtransfunc.adjust_saturation(image, self.saturation)
        return image, sample