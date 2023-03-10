#! /usr/bin/env python

"""
weed dataset object and associated transforms as classes
"""

import os
import numpy as np
import torch
import torch.utils.data
import json
import torchvision.transforms as T
import random

# from skimage import transform as sktrans
from PIL import Image
from torchvision.transforms import functional as tvtransfunc
from Annotations import Annotations

class WeedDataset(object):
    """ weed dataset object for polygons """

    # annotation format, either VIA or AGKELPIE
    AGKELPIE_FORMAT = 'AGKELPIE'

    # old properties for VIA formatting
    VIA_FORMAT = 'VIA'
    
    # TODO put defaults, similar to James' code
    def __init__(self,
                 annotation_filename,
                 img_dir,
                 transforms=None,
                 mask_dir=None,
                 config_file=None,
                 format=AGKELPIE_FORMAT):
        """
        initialise the dataset
        annotations - absolute path to json file of annotations of a prescribed format
        img_dir - image directory
        transforms - list of transforms randomly applied to dataset for training
        mask_dir - mask directory (if maskrcnn)
        
        """
        # TODO address if masks folder not available, config classes, auto-create, etc
        # absolute filepath
        self.format = format
        if self.format == self.VIA_FORMAT:
            annotations = json.load(open(os.path.join(annotation_filename)))
            self.annotations = list(annotations.values())
            self.imgs = list(sorted(os.listdir(self.img_dir)))
        else:
            annotations = Annotations(filename=annotation_filename,
                                      img_dir=img_dir,
                                      ann_format=format)
            self.annotations = annotations.annotations
            self.imgs = annotations.imgs
            
        self.transforms = transforms

        self.img_dir = img_dir

        if mask_dir is not None:
            self.mask_dir = mask_dir
        else:
            self.mask_dir = os.path.join(img_dir, '..', 'masks') # assume parallel to image folder

        if config_file is not None:
            self.config_file = config_file
        else:
            self.config_file = os.path.join('config/classes.json')

        # load config_file json:
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        self.classes = config['names']
        self.class_colours = config['colours']
        
        # load all image files, sorting them to ensure aligned (dictionaries are unsorted)
        self.masks = list(sorted(os.listdir(self.mask_dir)))


    def __getitem__(self, idx):
        """
        given an index, return the corresponding image and sample from the dataset
        converts images and corresponding sample to tensors
        """
        if self.format == self.AGKELPIE_FORMAT:
            image, sample = self.getitem_agkelpie(idx)
        elif self.format == self.VIA_FORMAT:
            image, sample = self.getitem_via(idx)
        else:
            ValueError(self.format, 'annotation format unknown')
        return image, sample
        
        
    def getitem_agkelpie(self, idx):
        """
        given an index, return the corresponding image and sample from the dataset
        converts images and corresponding sample to tensors
        for agkelpie dataset format
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # get image
        img_name = os.path.join(self.img_dir, self.imgs[idx])
        image =  Image.open(img_name).convert("RGB")

        # get mask
        mask_name = os.path.join(self.mask_dir, self.masks[idx])
        mask =  np.array(Image.open(mask_name))

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
        # labels_pt = []
        # points = []
        
        area = []
        if nobj > 0:
            for reg in self.annotations[idx].regions:
                # if reg.shape_type == 'point':
                #     labels_pt.append( list(self.classes.values()).index(reg.class_name) )
                #     points.append((reg.shape.x, reg.shape.y))
                    # coords_pt.append(reg.shape.bounds) # returns bounding box, for pt, xy1, xy2 are the same
                    # TODO can +- the bounds to do centre tussock detection
                if reg.shape_type == 'polygon':
                    # labels.append( list(self.classes.values()).index(reg.class_name) )
                    # labels.append(int(self.get_key(self.classes, reg.class_name)))
                    labels.append(1)
                    # boxes.append(reg.shape.bounds)
                    area.append(reg.shape.area) # pixels
        else:
            # points = torch.zeros((0, 2), dtype=torch.float32)
            # boxes = torch.zeros((0, 4), dtype=torch.float32)
            area = torch.zeros(0, dtype=torch.float32)
            # labels = torch.zeros((0,), dtype=torch.int64)
        
        # convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = torch.as_tensor(area, dtype=torch.float32)
        # points = torch.as_tensor(points, dtype=torch.float32)
        # labels = torch.as_tensor(labels, dtype=torch.int64)
        # temporary - single class work-around:
        labels = torch.ones((nobj,), dtype=torch.int64)
        # iscrowd = torch.zeros((nobj,), dtype=torch.int64) # currently unused, but potential for crowded weed images
        image_id = torch.tensor([idx], dtype=torch.int64) # image_id is the index of the image in the folder

        # package into sample
        sample = self.package_sample(masks, labels, image_id, boxes=boxes, area=area)

        # apply transforms to image and sample
        if self.transforms:
            image, sample = self.transforms(image, sample)

        return image, sample


    def get_key(self, my_dict, val):
        """ helper function, get key of dictionary given value """
        for key, value in my_dict.items():
            if val == value:
                return key
    

    def package_sample(self, masks, labels, image_id, boxes, area=None):
    # def package_sample(self, masks, labels, image_id, boxes=None,area=None, iscrowd=None, points=None):
        """ helper function to package inputs into sample dictionary """
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
    
    
    def getitem_via(self, idx):
        """
        given an index, return the corresponding image and sample from the dataset
        converts images and corresponding sample to tensors
        for via dataset format only
        NOTE: not updated
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image
        img_name = os.path.join(self.img_dir, self.annotations[idx]['filename'])
        image =  Image.open(img_name).convert("RGB")

        # get mask
        mask_name = os.path.join(self.mask_dir, self.annotations[idx]['filename'][:-4] + '_mask.png')
        mask =  np.array(Image.open(mask_name))

        # if we have a normal mask of 0's and 1's
        if mask.max() > 0:
            # instances are encoded as different colors
            obj_ids = np.unique(mask)
            # first id is the background, so remove it
            obj_ids = obj_ids[1:]
            # split the color-encoded mask into a set of binary masks
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
        # bounding box is read in a xmin, ymin, width and height
        # bounding box is saved as xmin, ymin, xmax, ymax
        boxes = []
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

        # get annotation spray points
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
                # TODO need to do a python script that checks for this ahead of time
                # NOTE: not every polygon (nobj) should contain a spraypoint, because we don't want to spray every 
                # polygon necessarily. 
                # NOTE: reportedly, the new AgKelpie image database has this functionality
            points = torch.as_tensor(points, dtype=torch.float32)
        else:
            points = torch.zeros((0, 2), dtype=torch.float32)

        # compute area
        if nobj > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = 0
        area = torch.as_tensor(area, dtype=torch.float32)

        # read in all region attributes to apply label based on class names:
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
                    if self.check_species(species_name):
                        labels_box.append( list(self.classes.values()).index(species_name) )
                    else:
                        ValueError(species_name, f'species_name {species_name} not in self.classes.values()')

                if name == 'point':
                    # species_name = r['region_attributes']['species']
                    species_name = list(self.classes.items())[1] # TODO currently, point's don't have species annotation?
                    if self.check_species(species_name):
                        labels_pt.append( list(self.classes.values()).index(species_name) )
                    else:
                        ValueError(species_name, f'species_name {species_name} not in self.classes.values()')

                if name == 'polygon':
                    species_name = r['region_attributes']['species']
                    if self.check_species(species_name):
                        labels_poly.append( list(self.classes.values()).index(species_name) )
                    else:
                        ValueError(species_name, f'species_name {species_name} not in self.classes.values()')

        # for now, we don't care about point or box annotations, poly and box annotations should also be equivalent
        labels = torch.as_tensor(labels_poly, dtype=torch.int64)

        iscrowd = torch.zeros((nobj,), dtype=torch.int64) # currently unused, but potential for crowded weed images

        # image_id is the index of the image in the folder
        image_id = torch.tensor([idx], dtype=torch.int64)

        sample = self.package_sample(boxes, labels, image_id, area, iscrowd, masks, points)
        
        # apply transforms to image and sample
        if self.transforms:
            image, sample = self.transforms(image, sample)

        return image, sample


    def check_species(self, species):
        """ check species names, if matches self.classes, then returns true, else returns false """
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

            # mask = mask.transpose(method=Image.FLIP_LEFT_RIGHT)
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
                     config_file=config_file)

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
    for i, batch in enumerate(WD):
        img, target = batch
        boxes = target['boxes']
        print(f'{i}: {boxes}')
        
    import code
    code.interact(local=dict(globals(), **locals()))