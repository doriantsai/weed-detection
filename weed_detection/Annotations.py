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

from Image import Image
from AnnotationRegion import AnnotationRegion



class Annotations:
    # annotation metadata key from VIA tool
    METADATA = '_via_img_metadata'

    # annotation region shapes
    SHAPE_POLY = 'polygon'
    SHAPE_RECT = 'rect'
    SHAPE_POINT = 'point'

    # annotation species
    ATTR_SPECIES = 'species'
    SPECIES_TUSSOCK = 'tussock'
    SPECIES_THISTLE = 'thistle'
    SPECIES_HOREHOUND = 'horehound'

    # attributes
    ATTR_OCCLUDED = 'occluded'
    ATTR_ISPROCESSED = 'processed'

    def __init__(self, filename: str, img_dir: str):
        self.filename = filename

        # load anontation data
        self.annotations_raw = self.read_via_annotations_raw()

        # image directory
        self.img_dir = img_dir
        self.img_list = list(sorted(os.listdir(self.img_dir)))

        if not self.check_annotation_image_consistency():
            print('Number of images in img_dir is not equal to number of raw annotations in Annotations.__init__()')
            exit(-1)
        
        # convert to internal format
        self.annotations = self.convert_annotations()
        

    def read_agkelpie_image_database_raw(self):
        """
        read the raw annotations from the agkelpie image database
        return img metadata as a list for each image
        """
        # TODO
    
        
    
    def read_via_annotations_raw(self):
        """
        read the raw annotations from the annotation file
        return img metadata as a list for each image
        """
        metadata = json.load(open(self.filename))
        img_data = metadata[self.METADATA]
        data = list(img_data.values())

        return data if len(data) > 0 else False


    def check_annotation_image_consistency(self):
        """
        check that the number of annotations in file match number of images in img_dir
        """
        
        # for now, just check length/number of annotations, but in future
        # TODO check filenames with annotated filenames
        if len(self.annotations_raw) == len(self.img_list):
            return True
        else:
            print(f'Number of images in raw annotations: {len(self.annotations_raw)}')
            print(f'Number of images in img_dir: {len(self.img_list)}')
            return False


    # same as James Bishop for ease of comparisons
    # poly: x = all x points, y = all y points, length = number of points
    # rect: x = top-left & bottom-right x points, y = top-left & bottom-right y points, length = 2
    # point: x = x, y = y, length = 1
    def convert_annotations(self):
        """
        convert raw annotations into interal annotation format (nested classes of annotation regions)
        with relevant properties that will allow easy filtering based on attributes, such as weed species, occluded, etc
        """
        data = []

        # get image height/width from an img file in img_dir
        # assume all images are of the same size in the same img_dir
        im = PILImage.open(os.path.join(self.img_dir, self.img_list[0]))
        width, height = im.size


        # extract camera info from Dataset.txt in the metadata folder
        # NOTE I suspect that it is not currently updated, so for now, just use
        camera = 'MakoG507'

        for img_data in self.annotations_raw:
            filename = img_data['filename']
            filesize = img_data['size']
            file_attr = img_data['file_attributes']

            img = Image(filename = filename, 
                        filesize= filesize,
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


if __name__ == "__main__":

    print('Annotations.py')

    """ testing Annotaions.py read_via_annotations_raw """
    filename = '/home/dorian/Data/03_Tagged/2021-10-19/Jugiong/Thistle-10/metadata/Jugiong-10-Final.json'
    img_dir = '/home/dorian/Data/03_Tagged/2021-10-19/Jugiong/Thistle-10/images'
    Ann = Annotations(filename=filename, img_dir=img_dir)
    data = Ann.read_via_annotations_raw()
    print(data)

    """ testing convert annotations"""
    # tested via initialization
    i = 10
    Ann.annotations[i].print()

    # import code
    # code.interact(local=dict(globals(), **locals()))






