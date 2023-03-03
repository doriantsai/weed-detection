#! /usr/bin/env python3

"""
Region class, which corresponds to polygon, bbox or point, also has a class
"""

import numpy as np
from shapely.geometry import Point
from shapely.geometry import Polygon

class Region:

    def __init__(self, class_name: str, x, y):
        self.class_name = class_name

        # x, y can be a single int/pt, or an array of x's and y's
        self.shape = self.make_shape(x, y)


    @staticmethod
    def make_shape(x, y):
        """
        convert x, y values into shapely geometry object (point/polygon)
        NOTE: bbox is a type of polygon, accessed via shape.exterior.coords.xy
        """
        if type(x) is bool and type(y) is bool:
            shape = False
        elif type(x) is int and type(y) is int:
            shape = Point(x, y)
        else:
            # make sure x, y are same size
            x = np.array(x)
            y = np.array(y)
            if not x.shape == y.shape:
                print('Error: x is not same size/shape as y') # TODO should convert to try/except statement
            else:
                # create polygon from x, y values
                points = []
                i = 0
                while i < len(x):
                    # should have no negative x,y image coordinates
                    # TODO consider bounding wrt image dimensions
                    if x[i] < 0:
                        x[i] = 0
                    if y[i] < 0:
                        y[i] = 0
                    points.append(Point(int(x[i]), int(y[i])))
                    i += 1

            shape = Polygon(points)

        return shape
        
    # print Regions    
    def print(self):
        print('Region: ')
        print('Class: ' + str(self.class_name))
        if self.shape:
            print(f'Shape type: {self.shape.type}')
        else:
            print('shape is of unknown type')
        print('Coordinates: ')
        if type(self.shape) is Point:
            print(self.shape.bounds)
        elif type(self.shape) is Polygon:
            print('polygon coordinates:')
            print(self.shape.exterior.coords.xy)
        else:
            print('Unknown region shape')



if __name__ == "__main__":

    print('Region.py')

    """ testing of shape input"""

    # test a point
    pt_region = Region('tussock', 1, 4)
    pt_region.print()

    # test a polygon
    all_x = (1, 2, 4, 5)
    all_y = (3, 5, 6, 7)
    poly_region = Region('horehound', all_x, all_y)
    poly_region.print()

    # test bad input (TODO)