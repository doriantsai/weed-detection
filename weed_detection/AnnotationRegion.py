#! /usr/bin/env python3

"""
Annotated Region class, which has annotation and region properties
"""

from Region import Region

class AnnotationRegion(Region):

    # types of annotation regions
    SHAPE_POLY = 'polygon'
    SHAPE_RECT = 'rect'
    SHAPE_POINT = 'point'
    
    def __init__(self, class_name, x, y, shape_type, occluded):
        Region.__init__(self, class_name, x, y)
        self.occluded = occluded

        if shape_type not in [self.SHAPE_POLY or self.SHAPE_POINT or self.SHAPE_RECT]:
            self.error('Unknown shape type passed to AnnotationRegion __init__()')
            exit(-1)
        else:
            self.shape_type = str(shape_type)

        
    def print(self):
        Region.print(self)
        print('Shape type: ' + self.shape_type)
