#! /usr/bin/env python3

"""
Annotated Region class, which has annotation and region properties

Based on James Bishop's UNE weed detector, AnnotatedRegion is a Region that has
annotated properties associated with it, which can be either polygon, rectangle
or point. NOTE use of 'rect' has been superseded by 'polygon', and may be
removed from all weed_detection code

- occluded: boolean 1/0 True/False if plant is occluded, according to manual annotation
- plant_count: string, "Single Plant", "Part of Plant", or "Multiple Plants", according to manual annotation
- shape_type: string, "polygon", 'point', or 'rect' depending on annotation shape type
- all the properties inherited from Region():
- shape: (shapely geometry type)
- label: numeric label assigned during WeedDataset creation
- class_name: string of the actual weed species name

Dorian Tsai
March 2023
dy.tsai@qut.edu.au
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

        
    def print(self):
        Region.print(self)
        print('Shape type: ' + self.shape_type)
