#! /usr/bin/env python3

"""
Annotated Region class, which has annotation and region properties
"""

from weed_detection.Region import Region

class AnnotationRegion(Region):

    # types of annotation regions
    SHAPE_POLY = 'polygon'
    SHAPE_RECT = 'rect'
    SHAPE_POINT = 'point'
    SHAPE_TYPES = [SHAPE_POINT, SHAPE_RECT, SHAPE_POLY]
    
    def __init__(self, class_name, x, y, shape_type, occluded, plant_count=None):
        Region.__init__(self, class_name, x, y)
        self.occluded = occluded
        self.plant_count = plant_count
        self.shape_type = str(shape_type)
        # if shape_type not in [self.SHAPE_TYPES]:
        #     ValueError(shape_type, 'Unknown shape type passed to AnnotationRegion __init__()')
        # else:
        #     self.shape_type = str(shape_type)

        
    def print(self):
        Region.print(self)
        print('Shape type: ' + self.shape_type)
