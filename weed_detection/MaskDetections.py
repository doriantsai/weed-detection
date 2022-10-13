#! /usr/bin/env python3

from Detections import Detections
import numpy as np
import cv2 as cv

class MaskDetections(Detections):

    def __init__(self, label, score, mask, mask_threshold):
        self.label = label
        self.score = score
        self.mask = mask
        self.mask_threshold = mask_threshold
        # TODO mask gets converted to polygon
        self.mask_binary, x, y = self.binarize_confidence_mask(self.mask, self.mask_threshold)
        Detections.__init__(label, score, x, y, 'poly')


    def binarize_confidence_mask(self, 
                                 mask, 
                                 mask_threshold: float, 
                                 ksize=None, 
                                 MAX_KERNEL_SIZE: int = 11, 
                                 MIN_KERNEL_SIZE: int = 3):
        """ mask should be a 2D float image  (numpy array) with pixel values ranging from 0 to 1
        returns binary image and contour
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

        # find bounding polygon of binary image
        contours_in, hierarchy = cv.findContours(mask_close, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = list(contours_in)
        # we take the largest contour as the most likely polygon from the mask
        contours.sort(key=len, reverse=True)
        largest_contour = contours[0]
        largest_contour_squeeze = np.squeeze(contours)
        all_x, all_y = [], []
        for c in largest_contour_squeeze:
            all_x.append(c[0])
            all_y.append(c[1])

        return mask_close, all_x, all_y
