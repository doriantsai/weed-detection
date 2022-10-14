#! /usr/bin/env python3

from Detections import Detections
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class MaskDetections(Detections):

    MASK_THRESHOLD_DEFAULT = 0.5

    def __init__(self, label: int, score: float, mask_confidence = None, mask_binary = None, mask_threshold: float = None):

        if mask_threshold is None:
            self.mask_threshold = self.MASK_THRESHOLD_DEFAULT
        else:
            self.mask_threshold = mask_threshold

        if mask_confidence is None and mask_binary is None:
            self.mask_confidence = []
            self.mask_binary = []
            x = False
            y = False

        elif mask_confidence is None and isinstance(mask_binary, np.ndarray):
            self.mask_binary = mask_binary
            # x, y = self.get_bounding_polygon(self.mask_binary)
            x, y = False, False

        elif isinstance(mask_confidence, np.ndarray):
            self.mask_confidence = mask_confidence
            self.mask_binary = self.binarize_confidence_mask(self.mask_confidence, self.mask_threshold)
            x, y = self.get_bounding_polygon(self.mask_binary)
        
        else:
            raise TypeError('Unrecognised types for mask_confidence and/or mask_binary')
        
        
        # import code
        # code.interact(local=dict(globals(), **locals()))

        Detections.__init__(self,
                            label=label, 
                            score=score, 
                            x=x, 
                            y=y, 
                            shape_type = 'polygon')


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

        return mask_close


    def get_bounding_polygon(self, mask_binary):
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

    # TODO
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
    plt.show()


    md = MaskDetections(1, 0.6, mask_confidence=gauss, mask_threshold=0.75)
    md.print()

    plt.imshow(md.mask_binary)
    plt.show()

    # import code
    # code.interact(local=dict(globals(), **locals()))
