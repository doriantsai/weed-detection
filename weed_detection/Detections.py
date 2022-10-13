#! /usr/bin/env python3

from Region import Region

class Detections(Region):

    SHAPE_POLY = 'polygon'
    SHAPE_RECT = 'rect'
    SHAPE_POINT = 'point'
    SHAPE_TYPES = [SHAPE_POINT, SHAPE_RECT, SHAPE_POLY]

    CLASS_NAMES = ['background', 'tussock'] # TODO what's a better way to approach this? Do we even need this level of info at this point?

    def __init__(self, 
                 label: int, 
                 score: float, 
                 x = False, 
                 y = False, 
                 shape_type: str = False, 
                 centroid = False):

        self.label = label
        self.class_name = self.CLASS_NAMES[label]
        self.score = score

        if shape_type not in [self.SHAPE_TYPES]:
            raise TypeError('Unknown shape_type passed to Detections __init__()')
        else:
            self.shape_type = str(shape_type)

        self.shape = self.make_shape(x, y)
        centroid = self.get_centroid()
        self.centroid_x = centroid(0)
        self.centroid_y = centroid(1)
        

    def get_centroid(self):
        centroid = self.shape.centroid
        cx = centroid.coords[0][0]
        cy = centroid.coords[0][1]
        return (cx, cy) # I think this is shapely's thing?


    def print(self):
        print('Detection:')
        print(f'label: {self.label}')
        print(f'score: {self.score}')
        Region.print(self)
        print(f'centroid: ({self.centroid_x}, {self.centroid_y})')
        
 