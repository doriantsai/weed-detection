#! /usr/bin/env python3

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from weed_detection.Detections import Detections

"""MaskDetections
MaskDetections class inherits from Detections and adds mask functionality to the
object. In particular it takes in the confidence mask and converts it to a
binary mask, and then converts that to a polygon.

Raises:
    TypeError: _description_ TypeError: _description_

Returns:
    _type_: _description_
"""
class MaskDetections(Detections):

    MASK_THRESHOLD_DEFAULT = 0.5

    def __init__(self, 
                 label: int, 
                 score: float, 
                 mask_confidence = None, 
                 mask_binary = None, 
                 mask_threshold: float = None,
                 class_names: list = ['background', 'Tussock']):

        # sort out possible inputs and defaults
        if mask_threshold is None:
            self.mask_threshold = self.MASK_THRESHOLD_DEFAULT
        else:
            self.mask_threshold = mask_threshold

        if mask_confidence is None and mask_binary is None:
            self.mask_confidence = []
            self.mask_binary = []
            x = False
            y = False
            self.poly = False

        elif mask_confidence is None and isinstance(mask_binary, np.ndarray):
            self.mask_binary = mask_binary
            # x, y = self.get_bounding_polygon(self.mask_binary)
            x, y = False, False
            self.poly = False

        # the most likely situation:
        elif isinstance(mask_confidence, np.ndarray):
            self.mask_confidence = mask_confidence
            self.mask_binary = self.binarize_confidence_mask(self.mask_confidence, self.mask_threshold)
            x, y = self.get_bounding_polygon(self.mask_binary)
            self.poly = np.array((x, y)) # self.poly = [] # where we will save the polygons

        else:
            raise TypeError('Unrecognised types for mask_confidence and/or mask_binary')
        
        Detections.__init__(self,
                            label=label, 
                            score=score, 
                            x=x, 
                            y=y, 
                            shape_type = 'polygon',
                            class_names=class_names)
        # end init
            

    def binarize_confidence_mask(self, 
                                 mask, 
                                 mask_threshold: float, 
                                 ksize=None, 
                                 MAX_KERNEL_SIZE: int = 11, 
                                 MIN_KERNEL_SIZE: int = 3):
        """binarize_confidence_mask
        binarize the confidence mask, which comes in as a float image and leaves
        as a binary image. Specifically, mask should be a 2D float image  (numpy
        array) with pixel values ranging from 0 to 1 returns binary image and
        contour

        Args:
            mask (_type_): _description_
            mask_threshold (float): _description_
            ksize (_type_, optional): _description_. Defaults to None.
            MAX_KERNEL_SIZE (int, optional): _description_. Defaults to 11.
            MIN_KERNEL_SIZE (int, optional): _description_. Defaults to 3.

        Raises:
            TypeError: _description_

        Returns:
            _type_: _description_
        """        
        if not isinstance(mask, np.ndarray):
            raise TypeError('mask is not a numpy array')
        
        # binary conversion
        ret, mask_bin = cv.threshold(mask, mask_threshold, maxval=1.0, type=cv.THRESH_BINARY)

        # do morphological operations on mask to smooth it out
        h, w = mask_bin.shape
        imsize = min(h, w)

        # set kernel size
        if ksize is None:
            # roughly 1% of the image size
            ksize = np.floor(0.01 * imsize)

            if ksize % 2 == 0:
                ksize +=1 # if even, make it odd
            if ksize > MAX_KERNEL_SIZE:
                ksize = MAX_KERNEL_SIZE
            if ksize < MIN_KERNEL_SIZE:
                ksize = MIN_KERNEL_SIZE
        ksize = int(ksize)
        kernel = np.ones((ksize, ksize), np.uint8)
        mask_open = cv.morphologyEx(mask_bin.astype(np.uint8), cv.MORPH_OPEN, kernel)
        mask_close = cv.morphologyEx(mask_open, cv.MORPH_CLOSE, kernel)

        return mask_close


    def get_bounding_polygon(self, mask_binary):
        """get_bounding_polygon
        Finds the contour surrounding the blobs within the binary mask

        Args:
            mask_binary (uint8 2D numpy array): binary mask

        Returns:
            all_x, all_y: all the x, y points of the contours as lists
        """        
        # find bounding polygon of binary image
        contours_in, hierarchy = cv.findContours(mask_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = list(contours_in)
        # we take the largest contour as the most likely polygon from the mask
        contours.sort(key=len, reverse=True)
        largest_contour = contours[0]
        largest_contour_squeeze = np.squeeze(largest_contour)
        all_x, all_y = [], []
        for c in largest_contour_squeeze:
            all_x.append(c[0])
            all_y.append(c[1])

        return all_x, all_y


    def print(self):
        """print
        print all properties of the MaskDetection object
        """        
        print('MaskDetection: ')
        print(f'mask_threshold: {self.mask_threshold}')
        if isinstance(self.mask_binary, np.ndarray):
            print(f'mask_binary image size: {self.mask_binary.shape}')
        else:
            print(f'mask_binary is False')
        if isinstance(self.mask_confidence, np.ndarray):
            print(f'mask_confidence image size: {self.mask_confidence.shape}')
        else:
            print(f'mask_confidence is False')

        Detections.print(self)

        
        
if __name__ == "__main__":

    print('MaskDetections.py')

    """ testing MaskDetections input """
    # mask_file = '2021-10-13-T_13_50_19_300_mask.png'

    # what we're doing here to verify MaskDetections.py
    # create a blank image (numpy array)
    # add a Gaussian to this from 0-1
    # this is the confidence mask

    # create gaussian
    kernel_size = 50 # for the sake of testing, also the image size
    sigma = 1
    muu = 0
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x**2+y**2)
 
    # lower normal part of gaussian
    normal = 1.0 /(2.0 * np.pi * sigma**2)
 
    # Calculating Gaussian filter
    gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) * normal

    # Adjust the Gauss to span from 0-1
    gauss = gauss / gauss.max()

    plt.imshow(gauss)
    # plt.show()

    md = MaskDetections(1, 0.6, mask_confidence=gauss, mask_threshold=0.75)
    md.print()

    plt.imshow(md.mask_binary)
    plt.show()

    import code
    code.interact(local=dict(globals(), **locals()))
