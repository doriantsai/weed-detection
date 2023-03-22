#! /usr/bin/env python3

"""
Annotation - class to represent all the different annotation types
with respect to shape, species, attributes
"""
from __future__ import annotations
import json
import os
from PIL import Image as PILImage
from copy import deepcopy
import numpy as np
import cv2 as cv
from matplotlib.patches import Polygon as MPLPolygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

from Image import Image
from AnnotationRegion import AnnotationRegion



class Annotations:
    # annotation format, either VIA or AGKELPIE
    AGKELPIE_FORMAT = 'AGKELPIE'

    # old properties for VIA formatting
    VIA_FORMAT = 'VIA'
    
    # old properties for VIA formatting
    # annotation metadata key from VIA tool
    VIA_METADATA = '_via_img_metadata'
    
    # old properties for VIA formatting
    # annotation region shapes
    SHAPE_POLY = 'polygon'
    SHAPE_RECT = 'rect'
    SHAPE_POINT = 'point'

    # old properties for VIA formatting
    # annotation species
    ATTR_SPECIES = 'species'
    SPECIES_TUSSOCK = 'tussock'
    SPECIES_THISTLE = 'thistle'
    SPECIES_HOREHOUND = 'horehound'

    # old properties for VIA formatting
    # attributes
    ATTR_OCCLUDED = 'occluded'
    ATTR_ISPROCESSED = 'processed'


    def __init__(self, 
                 filename: str, 
                 img_dir: str, 
                 mask_dir: str = None,
                 ann_format: str = AGKELPIE_FORMAT,
                 dataset_name: str = None,
                 species: str = None):

        self.filename = filename
        self.dataset_name  = dataset_name
        self.species = species

        # image directory
        self.img_dir = img_dir
        self.imgs = list(sorted(os.listdir(self.img_dir)))
        
        # load annotations data
        if ann_format == self.AGKELPIE_FORMAT:
            self.annotations_raw, self.dataset_name, self.species = self.read_agkelpie_annotations_raw()
            self.annotations = self.convert_agkelpie_annotations() # convert to internal format
            
        elif ann_format == self.VIA_FORMAT:
            self.annotations_raw = self.read_via_annotations_raw()
            self.annotations = self.convert_via_annotations() # convert to internal format
        else:
            raise ValueError(ann_format, 'Unknown annotation format')

        self.annotations, self.imgs = self.find_matching_images_annotations()

        # mask directory
        self.mask_dir = mask_dir
        # if directory is empty (no files), then create masks
        if len(os.listdir(self.mask_dir)) == 0:
            self.create_masks_from_polygons()
        self.masks = list(sorted(os.listdir(self.mask_dir)))
        # end init
        

    def read_agkelpie_annotations_raw(self):
        """
        read the raw annotations from the agkelpie weed reference data library format (dataset.json)
        return img metadata as a list for each image
        """
        metadata = json.load(open(self.filename))
        data = metadata['images_tagged']

        species = metadata['species'] # TODO: what do for multiple species?
        location_name = metadata['name']
        folder_name = metadata['folder_name']
        dataset_name = folder_name + '_' + location_name # dataset_name used to make dirs

        return data, dataset_name, species if len(data) > 0 else False


    def read_via_annotations_raw(self):
        """
        read the raw annotations from the via annotation file
        return img metadata as a list for each image
        """
        metadata = json.load(open(self.filename))
        img_data = metadata[self.VIA_METADATA]
        data = list(img_data.values())

        return data if len(data) > 0 else False


    def check_annotation_image_consistency(self):
        """
        check that the number of annotations in file match number of images in img_dir
        """
        
        # for now, just check length/number of annotations, but in future
        # TODO check filenames with annotated filenames
        if len(self.annotations_raw) == len(self.imgs):
            return True
        else:
            print(f'Number of images in raw annotations: {len(self.annotations_raw)}')
            print(f'Number of images in img_dir: {len(self.imgs)}')
            return False


    def convert_agkelpie_annotations(self): 
        """
        convert raw agkelpie annotations into internal annotation format, which is a nested
        class of annotations with relevant properties that will allow easy filtering based on attributes, such as
        weed species, annotation polygon size, etc
        """
        data = []

        img_names = list(self.annotations_raw.keys())

        # get image size
        im = PILImage.open(os.path.join(self.img_dir, img_names[0]))
        width, height = im.size

        # for key, value in dictionary:
        for img_name, img_data in self.annotations_raw.items():
            camera = img_data['camera_model']
            img = Image(filename=img_name,
                        width = width,
                        height = height,
                        camera = camera)
            
            if len(img_data['annotations']) > 0:
                for ann in img_data['annotations']:
                    shape_attr = ann['shape']
                    occluded = ann['occluded']
                    plant_count = ann['plant_count']
                    species = ann['species']
                    coordinates_str = ann['coordinates'] # coordinates stored as strings formatted like python dictionaries
                    if shape_attr == 'point':
                        coordinates_dict = json.loads(coordinates_str) # convert string to pythonic object (dictionary)
                        img.regions.append(AnnotationRegion(class_name = species,
                                                            x = coordinates_dict['cx'], 
                                                            y = coordinates_dict['cy'],
                                                            shape_type = shape_attr,
                                                            occluded = occluded,
                                                            plant_count=plant_count))
                    elif shape_attr == 'polygon':
                        coordinates_list = json.loads(coordinates_str) # convert text string into list (already formatted as a list)
                        all_x = []
                        all_y = []
                        for pts in coordinates_list:
                            all_x.append(pts['x'])
                            all_y.append(pts['y'])
                        img.regions.append(AnnotationRegion(class_name = species, 
                                                            x = all_x,
                                                            y = all_y,
                                                            shape_type = shape_attr,
                                                            occluded = occluded,
                                                            plant_count=plant_count))
                    else:
                        ValueError(shape_attr, 'unknown or unsupported annotation shape type')        

            data.append(img)

        return data if len(data) > 0 else False


    # same as James Bishop for ease of comparisons
    # poly: x = all x points, y = all y points, length = number of points
    # rect: x = top-left & bottom-right x points, y = top-left & bottom-right y points, length = 2
    # point: x = x, y = y, length = 1
    def convert_via_annotations(self):
        """
        convert raw annotations into interal annotation format (nested classes of annotation regions)
        with relevant properties that will allow easy filtering based on attributes, such as weed species, occluded, etc
        """
        data = []

        # get image height/width from an img file in img_dir
        # assume all images are of the same size in the same img_dir
        im = PILImage.open(os.path.join(self.img_dir, self.imgs[0]))
        width, height = im.size


        # extract camera info from Dataset.txt in the metadata folder
        # NOTE I suspect that it is not currently updated, so for now, just use
        camera = 'MakoG507'

        for img_data in self.annotations_raw:
            filename = img_data['filename']
            file_attr = img_data['file_attributes']

            img = Image(filename = filename, 
                        width = width, 
                        height = height, 
                        camera = camera,
                        file_attr = file_attr)

            if len(img_data['regions']) > 0:
                for annotation in img_data['regions']:
                    shape_attr = annotation['shape_attributes']
                    region_attr = annotation['region_attributes']

                    if len(region_attr) > 0:
                        if self.ATTR_SPECIES in region_attr:
                            class_name = region_attr['species']
                        else:
                            class_name = False
                        if self.ATTR_OCCLUDED in region_attr:
                            occluded = region_attr['occluded']
                        else:
                            occluded = False
                    else:
                        class_name = False
                        occluded = False

                    if shape_attr['name'] == self.SHAPE_POLY:
                        img.regions.append(AnnotationRegion(class_name = class_name, 
                                                            x = shape_attr['all_points_x'],
                                                            y = shape_attr['all_points_y'],
                                                            shape_type = self.SHAPE_POLY,
                                                            occluded = occluded))
                    elif shape_attr['name'] == self.SHAPE_POINT:
                        img.regions.append(AnnotationRegion(class_name = class_name,
                                                            x = shape_attr['cx'],
                                                            y = shape_attr['cy'],
                                                            shape_type = self.SHAPE_POINT,
                                                            occluded = occluded))
                    else:
                        print('ERROR: unknown or unsupported annotation shape type')

            data.append(img)

        return data if len(data) > 0 else False

    # TODO def filter_annotation_shape()
    # TODO def filter_annotations_filename()


    def find_matching_images_annotations(self):
        """
        # removed because is_processed flag (from VIA days) was removed/missed in the development of the agkelpie weed detection library
        # this means # imgs in img_dir is not necessarily == raw_annotations
        # if not self.check_annotation_image_consistency():
        #     print('Number of images in img_dir is not equal to number of raw annotations in Annotations.__init__()')
        #     exit(-1)
        # thus, we need to find whichever is lower/less, and then find the corresponding set...
        # find number of imgs in img_dir
        # find the number of annotations
        # whichever is smaller, take that and (hope to) find the corresponding set in the larger group
        """
        # TODO image download stride is a potential issue - need not just to find min, but also matching
        n_img = len(self.imgs)
        n_ann = len(self.annotations)

        if n_img <= n_ann:
            # number of images is limiting factor, find corresponding annotations based on img_list
            matching_img = self.imgs
            matching_ann = []
            for img_name in self.imgs:
                for ann in self.annotations:
                    if ann.filename == img_name:
                        matching_ann.append(ann)
        else:
            # number of annotations is limiting factor, find corresponding images based on self.annotations
            matching_ann = self.annotations
            matching_img = []
            for ann in self.annotations:
                for img_name in self.imgs:
                    if img_name == ann.filename:
                        matching_img.append(img_name)

        print(f'original img_list = {len(self.imgs)}')
        print(f'original ann_list = {len(self.annotations)}')
        print(f'matching img_list = {len(matching_img)}')
        print(f'matching ann_list = {len(matching_ann)}')

        # check using sets:
        ann_list = [ann.filename for ann in matching_ann]
        if set(ann_list) == set(matching_img):
            print('the lists contain the same elements')
        else:
            print('the lists do not contain the same elements')

        # update the annotations and img_list
        return matching_ann, matching_img


    def create_masks_from_polygons(self,
                                   mask_dir=None):
        """create binary masks for img_list and ann_list, save in mask_dir
        NOTE: only valid for agkelpie format, not via format

        Args:
            mask_dir (_type_, optional): _description_. Defaults to None.
        """
        # assume img_list and ann_list are consistent
        assert len(self.annotations) == len(self.imgs), "num annotations should be == num images"

        # create masks folder
        if mask_dir is None:
            mask_dir = self.mask_dir
        os.makedirs(mask_dir, exist_ok=True)

        for ann in self.annotations:

            # create mask image
            mask = np.zeros((ann.width, ann.height), np.int32)

            # iterate over ann.regions
            # find each annotation that is a polygon ann.shape_type
            # then get x, y coordinates
            count_poly = 0
            poly = []
            for reg in ann.regions:
                if reg.shape_type == 'polygon':
                    # reg is already a polygon
                    x, y = reg.shape.exterior.coords.xy
                    xy = np.array([x, y], np.int32).transpose()
                    count_poly += 1
                    cv.fillPoly(mask, [xy], color=(count_poly))
                    poly.append(xy)

            # save mask
            mask_name = ann.filename[:-4] + '_mask.png'
            cv.imwrite(os.path.join(mask_dir, mask_name), mask)

            # for debugging purposes, make visual mask figures:
            SAVE = False
            if SAVE:
                img = PILImage.open(os.path.join(self.img_dir, ann.filename))
                patches = []
                for p in poly:
                    patches.append(MPLPolygon(p, closed=True))

                colors = 100 * np.random.rand(len(patches))
                poly_patches = PatchCollection(patches, alpha=0.4)
                poly_patches.set_array(colors)
                fig, ax = plt.subplots(2,1)
                ax[0].imshow(img)
                ax[0].add_collection(poly_patches)
                ax[1].imshow(mask)
                plt.savefig(os.path.join(mask_dir, mask_name[:-4] + '_debug.png'))
        print('Successfully created masks from polygons')


    def generate_imagelist_txt(self, txtfile, imglist=None):
        """
        output current self.annotations list to txt for further manipulation/esp useful for adjusting training/testing/validation sets
        assume txtfilename is full absolute path
        """
        # default imglist
        if imglist is None:
            imglist = [ann.filename for ann in self.annotations]
            
        # ann.imgs should be a complete list of all the images in annotations
        assert len(imglist) <= len(self.imgs), "number of images in txtfile must be equal to or less than the number of annotations"
        
        # ensure imglist is contained within set of self.imgs
        missing_imgs_from_ann = set(imglist) - set(self.imgs)
        if len(missing_imgs_from_ann) > 0:
            TypeError(imglist, 'imglist has images that are not contained within self.imgs')
            # TODO print those images
        
        # for each filename in imglist, write it to a text file
        with open(txtfile, 'w') as f:
            for img in imglist:
                f.write(f'{img}\n')
        print(f'Successfully wrote image list to txtfile: {txtfile}')


    def prune_annotations_from_imagelist_txt(self, txtfile):
        """
        read in text file that has a list of images with which to use for training/testing/validation
        update the self.annotations list with relelvant images
        """
        # read image names from txtfile and remove trailing /n
        with open(txtfile, 'r') as f:
            imgs_str = f.readlines()
        txt_imgs = [img.strip() for img in imgs_str]
        
        print(f'number of images listed in txtfile: {len(txt_imgs)}')
        print(f'number of images listed in annotations: {len(self.annotations)}')
        
        # asserts
        assert len(txt_imgs) <= len(self.annotations), "number of images in txtfile must be equal to or less than the number of annotations"
        
        # for each img not in self.annotations.filenames, we want to remove them
        ann_imgs = [ann.filename for ann in self.annotations]
        
        # find matching number of images in self.annotations list vs imgs_str
        missing_imgs_from_txt = set(ann_imgs) - set(txt_imgs)
        missing_imgs_from_ann = set(txt_imgs) - set(ann_imgs)
        
        if len(missing_imgs_from_ann) > 0:
            print('Warning: txt_imgs lists images that aren''t contained within ann_imgs')
        if len(missing_imgs_from_txt) > 0:
            print('These are the images we want to remove/prune:')
            imgs_remove = list(missing_imgs_from_txt)
            for i, img in enumerate(imgs_remove):
                print(f'{i}: {img}')

                # find the corresponding img in self.annotations and remove it
                for j, ann in enumerate(self.annotations):
                    if img == ann.filename:
                        del self.annotations[j]
                
                # also remove in self.imgs:
                self.imgs.remove(img)        
                
                # also remove from self.masks:
                mask_name = img[:-4] + '_mask.png'
                self.masks.remove(mask_name)
        print(f'Successfully pruned annotations, image list and mask list wrt txtfile: {txtfile}')
              

if __name__ == "__main__":

    print('Annotations.py')

    """ testing Annotaions.py read_via_annotations_raw """
    filename = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_canberra_20220422_first500/dataset.json'
    img_dir = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_canberra_20220422_first500/annotated_images'
    
    Ann = Annotations(filename=filename, img_dir=img_dir)
    # data = Ann.read_agkelpie_annotations_raw()
    # print(data)
    
    anns, imgs = Ann.find_matching_images_annotations()
    # data = Ann.read_via_annotations_raw()
    # print(data)

    """ testing convert annotations"""
    # tested via initialization
    i = 10
    Ann.annotations[i].print()

    # print('making masks from polygons')
    # mask_dir = os.path.join(img_dir, '..', 'masks')
    # os.makedirs(mask_dir, exist_ok=True)
    # Ann.create_masks_from_polygons(mask_dir=mask_dir)    
    
    # generate image list txt file:
    # print('generating image list txt file')
    txtfile = os.path.join(img_dir, '..', 'annotated_images_from_annotations.txt')
    # Ann.generate_imagelist_txt(txtfile)
    # print(f'txtfile = {txtfile}')

    print('testing out txtfile reading')
    txtfile = os.path.join(img_dir, '..', 'annotated_images_from_annotations.txt')
    Ann.prune_annotations_from_imagelist_txt(txtfile)
    
    import code
    code.interact(local=dict(globals(), **locals()))






