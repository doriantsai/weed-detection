#! /usr/bin/env python

"""
WeedDataset is a class required by Pytorch for training neural networks that
provides the image and the training "sample" (i.e. annotations) for a given
index

Also after the WeedDataset class, there are associated transforms as classes
used for data augmentation and resizing image tensors
"""

import os
import numpy as np
import torch
import torch.utils.data
import json
import random
import torchvision

# from skimage import transform as sktrans
from PIL import Image as PILImage
from torchvision.transforms import functional as tvtransfunc

from weed_detection.Annotations import Annotations


class WeedDataset(object):
    """ weed dataset object for polygons """


    def __init__(self,
                 annotation_filename,
                 img_dir,
                 transforms=None,
                 mask_dir=None,
                 classes_file=None,
                 imgtxt_file=None):
        """__init__
        initialise the dataset

        Args:
            annotation_filename (_type_): absolute filename to dataset.json
            img_dir (_type_): absolute filepath to image directory
            transforms (_type_, optional): transforms applied to the image automatically. Defaults to None.
            mask_dir (_type_, optional): absolute filepath to mask directory. Defaults to None.
            classes_file (_type_, optional): absolute filepath to class file. Defaults to None.
            imgtxt_file (_type_, optional): absolute filepath to imagelists textfile. Defaults to None.
        """        
        
        self.transforms = transforms
        self.img_dir = img_dir

        if mask_dir is not None:
            self.mask_dir = mask_dir
        else:
            self.mask_dir = os.path.join(os.path.dirname(img_dir), 'masks') # assume parallel to image folder

        if classes_file is not None:
            self.config_file = classes_file
        else:
            self.config_file = os.path.join('config/classes.json')

        # load config_file json:
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        self.classes = config['names']
        self.class_colours = config['colours']
        
        # absolute filepaths, the annotation object
        annotations = Annotations(filename=annotation_filename,
                                    img_dir=img_dir,
                                    mask_dir=mask_dir)
        
        # added to deal with training/testing sets from text files without
        # touching original dataset.json annotation files
        if imgtxt_file is not None:
            self.imgtxt_file = imgtxt_file
            print(f'pruning annotations using {imgtxt_file}')
            annotations.prune_annotations_from_imagelist_txt(imgtxt_file)
                
        self.annotations = annotations.annotations
        self.num_classes = annotations.num_classes
        self.imgs = annotations.imgs
        self.masks = annotations.masks
        self.define_labels()


    def define_labels(self):
        """define_labels
        given the number of classes and names of species in the annotations,
        define labels starting from 1 and incrementing +1 each time a new
        species is found
        """        
        # list of objects is self.annotations[image_index].regions[regions_index] region name is given as self.annotations[image_index].regions[regions_index].class_name
        
        check_label = [] # just for debugging
        unique_species = {} # a dictionary to keep track of encountered species names (strings)
        species_label = 1 # a label counter to keep track of the label increment, +1 each time we encounter a new species
        for img in self.annotations:
            for reg in img.regions:
                species_name = reg.class_name
                if species_name not in unique_species:
                    unique_species[species_name] = species_label
                    species_label += 1
                reg.label = unique_species[species_name]
                check_label.append(unique_species[species_name])
        print('successfully applied labels to each region')


    def __getitem__(self, idx):
        """__getitem__
        given an index, return the corresponding image and sample from the dataset
        converts images and corresponding sample to tensors

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # get image
        img_name = os.path.join(self.img_dir, self.imgs[idx])
        image =  PILImage.open(img_name).convert("RGB")

        # get mask
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        mask =  np.array(PILImage.open(mask_path))
       
        # convert masks with different instances of different colours to pure binary mask
        # if we have a normal mask of 0's and 1's
        if mask.max() > 0:
            obj_ids = np.unique(mask) # instances are encoded as different colors
            obj_ids = obj_ids[1:] # first id is the background, so remove it
            masks = mask == obj_ids[:, None, None] # split the color-encoded mask into a set of binary masks
            nobj = len(obj_ids)
        else:
            # for a negative image, mask is all zeros, or just empty 
            # masks = np.expand_dims(mask == 1, axis=1)
            nobj = 0     

        if nobj > 0:
            masks = torch.as_tensor(masks, dtype=torch.uint8)
        else:
            masks = torch.zeros((0, image.size[0], image.size[1]), dtype=torch.uint8)

        boxes = []
        if nobj > 0:
            for i in range(nobj):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        # get annotations
        labels = []
        area = []
        if nobj > 0:
            for reg in self.annotations[idx].regions:
                # if reg.shape_type == 'point':
                    # labels_pt.append( list(self.classes.values()).index(reg.class_name) )
                #     points.append((reg.shape.x, reg.shape.y))
                # coords_pt.append(reg.shape.bounds) # returns bounding box, for pt, xy1, xy2 are the same
                # TODO can +- the bounds to do centre tussock detection

                if reg.shape_type == 'polygon':
                    labels.append(int(reg.label))
                    area.append(reg.shape.area) # pixels
        else:
            area = torch.zeros(0, dtype=torch.float32)
            labels = []
        
        # convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = torch.as_tensor(area, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx], dtype=torch.int64) # image_id is the index of the image in the folder

        # package into sample
        sample = self.package_sample(masks, labels, image_id, boxes=boxes, area=area)

        # apply transforms to image and sample and mask
        if self.transforms:
            image, sample = self.transforms(image, sample)

        return image, sample


    def get_key(self, my_dict, val):
        """get_key
        helper function to get key of dictionary given its value

        Args:
            my_dict (_type_): _description_
            val (_type_): _description_

        Returns:
            _type_: _description_
        """        
        for key, value in my_dict.items():
            if val == value:
                return key
    

    def package_sample(self, masks, labels, image_id, boxes, area=None):
        """package_sample
        helper function to package inputs into sample dictionary

        Args:
            masks (_type_): _description_
            labels (_type_): _description_
            image_id (_type_): _description_
            boxes (_type_): _description_
            area (_type_, optional): _description_. Defaults to None.
        """        
        sample = {}
        sample['labels'] = labels
        sample['image_id'] = image_id
        sample['masks'] = masks
        # if iscrowd is not None:
        #     sample['iscrowd'] = iscrowd
        if boxes is not None:
            sample['boxes'] = boxes
        if area is not None:
            sample['area'] = area
        # if points is not None:
        #     sample['points'] = points
        return sample
    

    def check_species(self, species):
        """check_species
        check species names, if matches self.classes, then returns true, else
        returns false 
        
        Args:
            species (_type_): _description_

        Returns:
            _type_: _description_
        """       
        if species in self.classes.values():
            return True
        else:
            return False
        

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
            boxes = torch.as_tensor(torch.from_numpy(boxes), dtype=torch.float32)
        sample['boxes'] = boxes

        masks = sample['masks']
        if not torch.is_tensor(masks):
            masks = torch.as_tensor(torch.from_numpy(masks), dtype=torch.float32)
        sample['masks'] = masks

        # points = sample['points']
        # if not torch.is_tensor(points):
        #     points = torch.as_tensor(torch.from_numpy(points), dtype=torch.float32)
        # sample['points'] = points

        return image, sample


class Rescale(object):
    """ Rescale image to given size """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image, sample=None):

        # handle the aspect ratio
        if isinstance(image, np.ndarray):
            image = PILImage.fromarray(image) # convert to PIL image

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
        img = torchvision.transforms.Resize((new_w, new_h))(image) # only works for PIL images

        # apply transform to bbox as well
        if sample is not None:

            # apply resize to mask as well
            mask = sample["masks"]

            if len(mask) > 0:
                m = torchvision.transforms.Resize((new_w, new_h))(mask)
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

            # points = sample['points']
            # if len(points) > 0:
            #     points[:, 0] = points[:, 0] * yChange
            #     points[:, 1] = points[:, 1] * xChange
            #     sample['points'] = np.float64(points)

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

        if random.random() < self.prob:
            w, h = image.size[:2]

            # flip image
            image = image.transpose(method=PILImage.FLIP_LEFT_RIGHT)

            # flip bbox
            bbox = sample['boxes']

            # bounding box is saved as xmin, ymin, xmax, ymax
            # only changing xmin and xmax
            if len(bbox) > 0:
                bbox[:, [0, 2]] = w - bbox[:, [2, 0]]  # note the indices switch (must flip the box as well!)
                sample['boxes'] = bbox

            # flip mask
            mask = sample['masks']

            # mask = mask.transpose(method=PILImage.FLIP_LEFT_RIGHT)
            if len(mask) > 0:
                mask = torch.flip(mask, [2])
                sample['masks'] = mask

            # points = sample['points']
            # if len(points) > 0:
            #     points[:, 0] = w - points[:, 0]
            #     sample['points'] = points

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
            image = image.transpose(method=PILImage.FLIP_TOP_BOTTOM)

            # flip bbox [xmin, ymin, xmax, ymax]
            bbox = sample['boxes']
            if len(bbox) > 0:
                # bbox[:, [0, 2]] = w - bbox[:, [2, 0]]
                bbox[:, [1, 3]] = h - bbox[:, [3, 1]]
                sample['boxes'] = bbox

            # flip mask
            mask = sample['masks']
            # mask = mask.transpose(method=PILImage.FLIP_TOP_BOTTOM)
            if len(mask) > 0:
                mask = torch.flip(mask, [1])
                sample['masks'] = mask

            # flip points
            # points = sample['points']
            # if len(points) > 0:
            #     points[:, 1] = h - points[:, 1]
            #     sample['points'] = points

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


if __name__ == "__main__":
    
    print('WeedDataset.py')
    
    # old via annotation format
    # jsonfile = '/home/dorian/Data/03_Tagged/2021-10-19/Jugiong/Thistle-10/metadata/Jugiong-10-Final.json'
    # img_dir = '/home/dorian/Data/03_Tagged/2021-10-19/Jugiong/Thistle-10/images'
    # tform = Compose([Rescale(1024),
    #                 RandomBlur(5, (0.5, 2.0)),
    #                 RandomHorizontalFlip(0),
    #                 RandomVerticalFlip(0),
    #                 ToTensor()])
    # WD = WeedDataset(annotation_filename=jsonfile, img_dir=img_dir, transforms=tform)
        
    ann_file = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_canberra_20220422_first500/dataset.json'
    img_dir = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_canberra_20220422_first500/annotated_images'
    mask_dir = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_canberra_20220422_first500/masks'
    config_file = '/home/agkelpie/Code/agkelpie_weed_detection/weed-detection/config/classes.json'

    tform = Compose([Rescale(1024),
                    RandomBlur(5, (0.5, 2.0)),
                    RandomHorizontalFlip(0),
                    RandomVerticalFlip(0),
                    ToTensor()])
    WD = WeedDataset(annotation_filename=ann_file,
                     img_dir=img_dir,
                     transforms=tform,
                     mask_dir=mask_dir,
                     classes_file=config_file)

    print(WD[0])
    
    # check if any have no annotaions:
    # for i, ann in enumerate(WD.annotations):
    #     if len(ann.regions) < 1:
    #         print(f'{i}: found negative image')
    #     else:
    #         print(f'{i}: num regions = {len(ann.regions)}')
    
    # check labels:
    # for i, batch in enumerate(WD):
    #     img, target = batch
    #     labels = target['labels']
    #     print(f'{i}: {labels}')

    # check bounding boxes:
    print('iterating through the entire weed dataset')
    for i, batch in enumerate(WD):
        img, target = batch
        # boxes = target['boxes']
        # print(f'{i}: {boxes}')
        mask = target['masks']
        print(f'{i}: mask size = {mask.shape}')
    
        
    # import code
    # code.interact(local=dict(globals(), **locals()))