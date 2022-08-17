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

# CLASSES 
# CLASSES = (0, 1, 2)
# CLASS_NAMES = ('background', 'Tussock', 'Horehound')
CLASS_DICT = {0: "background", 1:"Tussock", 2: "Horehound"}
CLASS_COLORS = ('black', 'blue', 'orange')
black = np.r_[0, 0, 0]/255
purple = np.r_[135, 0, 135]/255
orange = np.r_[255, 127, 80]/255
CLASS_COLOR_ARRAY = [black, purple, orange]

class WeedDatasetPoly(object):
    """ weed dataset object for polygons """

    # TODO maybe should actually hold the datasets/dataloader objects?
    def __init__(self,
                 root_dir,
                 json_file,
                 transforms,
                 img_dir=None,
                 mask_dir=None):
        """
        initialise the dataset
        annotations - json file of annotations of a prescribed format
        root_dir - root directory of dataset, under which are Images/All/Test/Train folders
        transforms - list of transforms randomly applied to dataset for training
        """

        # TODO annotations should have root_dir/Annotations/json_file
        annotations = json.load(open(os.path.join(root_dir, 'metadata', json_file)))
        self.annotations = list(annotations.values())
        self.root_dir = root_dir
        self.transforms = transforms


        if img_dir is not None:
            self.img_dir = img_dir
        else:
            self.img_dir = os.path.join(self.root_dir, 'images')

        if mask_dir is not None:
            self.mask_dir = mask_dir
        else:
            self.mask_dir = os.path.join(self.root_dir, 'masks')

        # load all image files, sorting them to ensure aligned (dictionaries are unsorted)
        self.imgs = list(sorted(os.listdir(self.img_dir)))
        self.masks = list(sorted(os.listdir(self.mask_dir)))



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

        # get mask
        mask_name = os.path.join(self.mask_dir, self.annotations[idx]['filename'][:-4] + '_mask.png')
        mask = Image.open(mask_name)
        # # convert PIL image to np array
        mask = np.array(mask)
        # print('get item')
        # import code
        # code.interact(local=dict(globals(), **locals()))
        # if we have a normal mask of 0's and 1's
        if mask.max() > 0:
            # instances are encoded as different colors
            obj_ids = np.unique(mask)
            # first id is the background, so remove it
            obj_ids = obj_ids[1:]

            # split the color-encoded mask into a set of binary masks
            # NOTE unsure if this will work as intended
            masks = mask == obj_ids[:, None, None]

            nobj = len(obj_ids)

        else:
            # for a negative image, mask is all zeros, or just empty
            # masks = np.expand_dims(mask == 1, axis=1)

            nobj = 0

        if nobj > 0:
            masks = torch.as_tensor(masks, dtype=torch.uint8)
        else:
            masks = torch.zeros((0, image.size[0], image.size[1]), dtype=torch.uint8)



        # get bounding boxes for each object
        # number of bboxes
        # nobj = len(self.annotations[idx]['regions'])

        # get bbox
        # bounding box is read in a xmin, ymin, width and height
        # bounding box is saved as xmin, ymin, xmax, ymax
        boxes = []

        # import code
        # code.interact(local=dict(globals(), **locals()))

        # bounding boxes code for fasterrcnn
        # if nobj > 0:
        #     for i in range(nobj):
        #         if isinstance(self.annotations[idx]['regions'], dict):
        #             j = str(i)
        #         else:  # regions is a list type
        #             j = i
        #         xmin = self.annotations[idx]['regions'][j]['shape_attributes']['x']
        #         ymin = self.annotations[idx]['regions'][j]['shape_attributes']['y']
        #         width = self.annotations[idx]['regions'][j]['shape_attributes']['width']
        #         height = self.annotations[idx]['regions'][j]['shape_attributes']['height']
        #         xmax = xmin + width
        #         ymax = ymin + height
        #         boxes.append([xmin, ymin, xmax, ymax])
        if nobj > 0:
            for i in range(nobj):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])

        if nobj == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # TODO read in points!
        points = []
        if nobj > 0:

            reg = self.annotations[idx]['regions']
            for i, r in enumerate(reg):
                if isinstance(self.annotations[idx]['regions'], dict):
                    j = str(i)
                else:  # regions is a list type
                    j = i
                name = r['shape_attributes']['name']
                if name == 'point':
                    cx = self.annotations[idx]['regions'][j]['shape_attributes']['cx']
                    cy = self.annotations[idx]['regions'][j]['shape_attributes']['cy']
                    points.append([cx, cy])
            # for i in range(nobj):
            #     # find which regions has shape_attributes name == point

                # points.append()
            # if not len(points) == nobj:
            #     print('WARNING: npoints not == nobj')
            #     print(f'idx = {idx}, img_name = {img_name}')
                # TODO need to do a python script that checks for this ahead of time
                # NOTE: not every polygon (nobj) should contain a spraypoint, because we don't want to spray every 
                # polygon necessarily. 

            points = torch.as_tensor(points, dtype=torch.float32)
        else:
            points = torch.zeros((0, 2), dtype=torch.float32)

        # compute area
        if nobj == 0:
            # no boxes
            area = 0
        else:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # only one class + background:
        # TODO: update for multiclass case
        # labels = torch.ones((nobj,), dtype=torch.int64)
        # read in all region attributes to apply label based on class names:
        # NOTE: metadata/annotation files will often have multiple regions for a single label
        # for now, we focus on the labels associated with polygons 
        # TODO: need labels associated with each type of annotation!
        labels_box = []
        labels_pt = []
        labels_poly = []
        if nobj > 0:
            reg = self.annotations[idx]['regions']
            for i, r in enumerate(reg):
                if isinstance(self.annotations[idx]['regions'], dict):
                    j = str(i)
                else:
                    j = i
                name = r['shape_attributes']['name']
                if name == 'rect':
                    species_name = r['region_attributes']['species']
                    labels_box.append( list(CLASS_DICT.values()).index(species_name) )

                if name == 'point':
                    species_name = r['region_attributes']['species']
                    labels_pt.append( list(CLASS_DICT.values()).index(species_name) )

                if name == 'polygon':
                    species_name = r['region_attributes']['species']
                    # find the value of the dictionary whose key equals weed_name
                    # NOTE: can fail if weed_name does not match any values from CLASS_DICT
                    # should probably run a pre-processing check that all json weed attributes match CLASS_DICT

                    labels_poly.append( list(CLASS_DICT.values()).index(species_name) )

        # for now, we don't care about point or box annotations, poly and box annotations should also be equivalent
        labels = torch.as_tensor(labels_poly, dtype=torch.int64)

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
        sample['masks'] = masks
        sample['points'] = points

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

class ToTensor(object):
    """ convert ndarray to sample in Tensors """

    def __call__(self, image, sample):
        """ convert image and sample to tensors """

        # convert image
        image = tvtransfunc.to_tensor(image)
        image = torch.as_tensor(image, dtype=torch.float32)
        # make sure to convert to float64

        # convert samples
        boxes = sample['boxes']
        if not torch.is_tensor(boxes):
            boxes = torch.from_numpy(boxes)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        sample['boxes'] = boxes

        masks = sample['masks']
        if not torch.is_tensor(masks):
            masks = torch.from_numpy(masks)
            masks = torch.as_tensor(masks, dtype=torch.float32)
        sample['masks'] = masks

        points = sample['points']
        if not torch.is_tensor(points):
            points = torch.from_numpy(points)
            points = torch.as_tensor(points, dtype=torch.float32)
        sample['points'] = points

        return image, sample


class Rescale(object):
    """ Rescale image to given size """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image, sample=None):

        # handle the aspect ratio
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image) # convert to PIL image

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

        img = T.Resize((new_w, new_h))(image) # only works for PIL images


        # apply transform to bbox as well
        if sample is not None:

            # apply resize to mask as well
            mask = sample["masks"]
            # print('we are here at the mask resizing')
            # import code
            # code.interact(local=dict(globals(), **locals()))

            if len(mask) > 0:
                m = T.Resize((new_w, new_h))(mask)  # HACK FIXME
                sample['masks'] = m

            xChange = float(new_w) / float(w)
            yChange = float(new_h) / float(h)
            bbox = sample["boxes"]  # [xmin ymin xmax ymax]

            if len(bbox) > 0:
                bbox[:, 0] = bbox[:, 0] * yChange
                bbox[:, 1] = bbox[:, 1] * xChange
                bbox[:, 2] = bbox[:, 2] * yChange
                bbox[:, 3] = bbox[:, 3] * xChange
                sample["boxes"] = np.float64(bbox)

            points = sample['points']
            if len(points) > 0:
                points[:, 0] = points[:, 0] * yChange
                points[:, 1] = points[:, 1] * xChange
                sample['points'] = np.float64(points)

            return img, sample
        else:
            return img



class RandomHorizontalFlip(object):
    """ Random horozintal flip """

    def __init__(self, prob):
        """ probability of a horizontal image flip """
        self.prob = prob


    def __call__(self, image, sample):
        """ apply horizontal image flip to image and sample """

        # import code
        # code.interact(local=dict(globals(), **locals()))


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

            # flip mask
            mask = sample['masks']

            # import code
            # code.interact(local=dict(globals(), **locals()))

            # mask = mask.transpose(method=Image.FLIP_LEFT_RIGHT)
            if len(mask) > 0:
                mask = torch.flip(mask, [2])
                sample['masks'] = mask

            points = sample['points']
            if len(points) > 0:
                points[:, 0] = w - points[:, 0]
                sample['points'] = points

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

            # flip mask
            mask = sample['masks']
            # mask = mask.transpose(method=Image.FLIP_TOP_BOTTOM)
            if len(mask) > 0:
                mask = torch.flip(mask, [1])
                sample['masks'] = mask

            # flip points
            points = sample['points']
            if len(points) > 0:
                points[:, 1] = h - points[:, 1]
                sample['points'] = points

        return image, sample


class RandomBlur(object):
    """ Gaussian blur images """

    def __init__(self, kernel_size=5, sigma=(0.1, 2.0)):
        """ kernel size and standard deviation (sigma) of Gaussian blur """
        # TODO consider the amount of blur in x- and y-directions
        # might be similar to simulating motion blur due to vehicle motion
        # eg, more y-dir blur = building robustness against faster moving vehicle?
        # kernel must be an odd number
        self.kernel_size = kernel_size
        self.sigma = sigma


    def __call__(self, image, sample):
        """ apply blur to image """
        # image = tvtransfunc.gaussian_blur(image, self.kernel_size) # sigma calculated automatically
        image = tvtransfunc.gaussian_blur(image, self.kernel_size, self.sigma)
        # TODO not sure if I should blur the mask? Mask RCNN accepts only binary mask, or can it be weighted?
        return image, sample



class RandomBrightness(object):
    """ Random color jitter transform """

    def __init__(self,
                 prob,
                 brightness=0,
                 range = [0.25, 1.75],
                 rand=True):
        self.prob = prob
        # check brightness single non-negative 0 gives a black image, 1
        # gives the original image while 2 increases the brightness by a factor
        # of 2
        self.brightness = brightness
        self.rand = rand
        self.range = range

    def __call__(self, image, sample):
        """ apply change in brightnes/constrast/saturation/hue """

        if random.random() < self.prob:
            if self.rand:
                # create random brightness within given range
                # random.random() is between 0, 1 random float
                brightness = random.random()* (self.range[1] - self.range[0]) + self.range[0]
                image = tvtransfunc.adjust_brightness(image, brightness)
            else:
                image = tvtransfunc.adjust_brightness(image, self.brightness)
        return image, sample


class RandomContrast(object):
    """ Random constrast jitter transform """

    def __init__(self,
                 prob,
                 contrast=0,
                 range = [0.5, 1.5],
                 rand=True):
        self.prob = prob
        # Can be any non negative number. 0 gives a solid gray image, 1
        # gives the original image while 2 increases the contrast by a factor of
        # 2.
        self.contrast=contrast
        self.rand = rand
        self.range = range

    def __call__(self, image, sample):
        """ apply change in brightnes/constrast/saturation/hue """

        if random.random() < self.prob:
            if self.rand:
                contrast = random.random()* (self.range[1] - self.range[0]) + self.range[0]
                image = tvtransfunc.adjust_contrast(image, contrast)
            else:
                image = tvtransfunc.adjust_contrast(image, self.contrast)
        return image, sample


# NOTE changing the hue seemed to really screw up the results, or potentially should only make the slightest
class RandomHue(object):
    """ Random hue jitter transform """

    def __init__(self,
                 prob,
                 hue=0,
                 range=[-0.1, 0.1],
                 rand=True):
        self.prob = prob
        # hue is a single number ranging from
        # Should be in [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of
        # hue channel in HSV space in positive and negative direction
        # respectively. 0 means no shift. Therefore, both -0.5 and 0.5 will give
        # an image with complementary colors while 0 gives the original image.
        self.hue = hue
        self.rand = rand
        self.range = range

    def __call__(self, image, sample):
        """ apply change in brightnes/constrast/saturation/hue """

        if random.random() < self.prob:
            if self.rand:
                hue = random.random()* (self.range[1] - self.range[0]) + self.range[0]
                image = tvtransfunc.adjust_hue(image, hue)
            else:
                image = tvtransfunc.adjust_hue(image, self.hue)
        return image, sample


class RandomSaturation(object):
    """ Random saturation """

    def __init__(self,
                 prob,
                 saturation=0,
                 range=[0.5, 1.5],
                 rand=True):
        self.prob = prob
        # 0 will give a black and white image, 1 will give the original
        # image while 2 will enhance the saturation by a factor of 2.
        self.saturation = saturation
        self.rand = rand
        self.range = range


    def __call__(self, image, sample):
        """ apply change in saturation """
        if random.random() < self.prob:
            if self.rand:
                sat = random.random()* (self.range[1] - self.range[0]) + self.range[0]
                image = tvtransfunc.adjust_saturation(image, sat)
            else:
                image = tvtransfunc.adjust_saturation(image, self.saturation)
        return image, sample