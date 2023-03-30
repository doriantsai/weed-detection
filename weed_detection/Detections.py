#! /usr/bin/env python3

"""Detections
Detections class is an object that inherits from a region, and adds a predicted
 score to the Region, which already includes attributes, such as shape and label
Also has a list of possible class names 

Raises:
    TypeError: Unknown str of shape_type (not point, polygon or rect)

Returns:
    _type_: _description_
"""
import numpy as np
from weed_detection.Region import Region

class Detections(Region):

    SHAPE_POLY = 'polygon'
    SHAPE_RECT = 'rect'
    SHAPE_POINT = 'point'
    SHAPE_TYPES = [SHAPE_POINT, SHAPE_RECT, SHAPE_POLY]


    def __init__(self, 
                 label: int, 
                 score: float, 
                 x = False, 
                 y = False, 
                 shape_type: str = False, 
                 centroid = False,
                 class_names: list = ['background', 'Tussock']):

        # list of potential classes, ordered such that label corresponds to the
        # correct class name
        self.class_names = class_names 
        Region.__init__(self, class_names[label], x, y, label=label)

        self.score = score # prediction/detection score of confidence

        if shape_type not in self.SHAPE_TYPES:
            raise TypeError('Unknown shape_type passed to Detections __init__()')
        else:
            self.shape_type = str(shape_type)

        # self.shape = self.make_shape(x, y)
        self.box = self.make_box(x, y) # [xmin, ymin, xmax, ymax]
        centroid = self.get_centroid()
        self.centroid_x = centroid[0]
        self.centroid_y = centroid[1]
         

    def get_centroid(self):
        """get_centroid
        computes centroid of the shape

        Returns:
            tuple of floats: centroid of the shape
        """
        if self.shape is False:
            return (False, False)
        else:
            centroid = self.shape.centroid # does this work for a point?
            cx = centroid.coords[0][0]
            cy = centroid.coords[0][1]
            return (cx, cy)


    def make_box(self, x, y):
        """make_box
        return exterior bounding box of x and y arrays
        top-left of image is (0,0) origin

        Args:
            x (array of floats/ints): horizontal image coordinates of shape
            y (array of floats/ints): vertical imager coordinates of shape

        Returns:
            array: 1x4 array of bounding box, defined as [xmin, ymin, xmax, ymax]
        """
        if x is False and y is False:
            return False
        else:
            xmin = min(x)
            ymin = min(y)
            xmax = max(x)
            ymax = max(y)
            bbox = [xmin, ymin, xmax, ymax]
            return bbox


    def print(self):
        """print
        print all properties of the Detection
        """
        print('Detection:')
        print(f'label: {self.label}')
        print(f'score: {self.score}')
        Region.print(self)
        print(f'centroid: ({self.centroid_x}, {self.centroid_y})')
        print(f'Bounding box (xmin, ymin, xmax, ymax): {self.box}')


if __name__ == "__main__":

    print('Detections.py')

    """ testing input """

    # test basic init for bbox
    x = np.array([0, 0, 1, 1])
    y = np.array([0, 1, 1, 0])
    det = Detections(label = 1,
                     score = 0.75,
                     x = x,
                     y = y,
                     shape_type = 'polygon')
    det.print()

    # test polygon (triangle) with bbox:
    x = np.array([0, 3, 6])
    y = np.array([0, 3, 0])
    det_tri = Detections(0, 0.5, x, y, 'polygon')
    det_tri.print()

    # python debug code
    # import code
    # code.interact(local=dict(globals(), **locals()))