#! /usr/bin/env python

import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
# import pandas as pd
import json
import matplotlib.pyplot as plt
# from skimage import transform as sktrans
from torchvision.transforms import functional as tvtransfunc
import torchvision.transforms as T
import random
from torchvision.datasets.video_utils import VideoClips


""" Serrated tussock dataset """
class SerratedTussockDataset(object):

    def __init__(self, root_dir, json_file, transforms):
        # self.weed_frame = pd.read_csv(csv_file)
        annotations = json.load(open(os.path.join(root_dir, json_file)))
        self.annotations = list(annotations.values())
        self.root_dir = root_dir
        self.transforms = transforms

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image
        # img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        img_name = os.path.join(self.root_dir, 'Images', self.annotations[idx]['filename'])
        image =  Image.open(img_name).convert("RGB")
        # image = plt.imread(img_name)

        # number of bboxes
        nobj = len(self.annotations[idx]['regions'])

        # get bbox
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
        if len(boxes) == 0:
            # no boxes
            area = 0
        else: 
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        area = torch.as_tensor(area, dtype=torch.float64)

        # only one class, so:
        labels = torch.ones((nobj,), dtype=torch.int64)

        # TODO iscrowd?
        iscrowd = torch.zeros((nobj,), dtype=torch.int64)

        image_id = torch.tensor([idx], dtype=torch.int64)

        sample = {}
        sample['boxes'] = boxes
        sample['labels'] = labels
        sample['image_id'] = image_id
        sample['area'] = area
        sample['iscrowd'] = iscrowd
        # sample['image_name'] = img_name
        # sample['image'] = image

        if self.transforms:
            # https://discuss.pytorch.org/t/t-compose-typeerror-call-takes-2-positional-arguments-but-3-were-given/62529/2
            image, sample = self.transforms(image, sample)
            # image = sample['image']

        return image, sample


    def __len__(self):
        return len(self.annotations)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Rescale(object):
    """ Rescale image to given size """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image, sample=None):
        # image = sample['image']
        
        # handle the aspect ratio
        h, w = image.size[:2]

        # import code
        # code.interact(local=dict(globals(), **locals()))

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        
        new_h, new_w = int(new_h), int(new_w)

        # do the transform
        # sktrans works on a numpy array
        # img = sktrans.resize(image, (new_h, new_w))
        img = T.Resize((new_w, new_h))(image)

        # import code
        # code.interact(local=dict(globals(), **locals()))

        # TODO apply transform to bbox as well
        if sample is not None:
            bbox = sample["boxes"]  # [xmin ymin xmax ymax]
            # bbox[:, [0, 2]] = width - bbox[:, [2, 0]]

            xChange = float(new_w) / float(w)
            yChange = float(new_h) / float(h)

            if len(bbox) == 0:
                # do nothing, since there are no bboxes
                print('rescale transform: no boxes here')
            else:
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
        # image = sample['image']

        # image = image.transpose((2, 0, 1))
        # image = torch.from_numpy(image)
        image = tvtransfunc.to_tensor(image)
        boxes = sample['boxes']
        if not torch.is_tensor(boxes):
            boxes = torch.from_numpy(boxes)
        sample['boxes'] = boxes

        # sample['image'] = image
        return image, sample


class RandomHorizontalFlip(object):
    """ Random horozintal flip """
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, sample):
        if random.random() < self.prob:
            w, h = image.size[:2]
            # flip image
            image = image.transpose(method=Image.FLIP_LEFT_RIGHT)

            # flip bbox
            bbox = sample['boxes']
            # bbox[:, [0, 2]] = w - bbox[:, [0, 2]]
            # import code
            # code.interact(local=dict(globals(), **locals())) 

            if len(bbox) == 0:
                print('horizontal flip transform: no boxes here')
            else:
                bbox[:, [0, 2]] = w - bbox[:, [2, 0]]  # note the indices switch (must flip the box as well!)
                sample['boxes'] = bbox

        return image, sample


class Blur(object):
    """ Gaussian blur images """

    def __init__(self, kernel_size=3, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, image, sample):

        image = tvtransfunc.gaussian_blur(image, self.kernel_size, self.sigma)

        return image, sample