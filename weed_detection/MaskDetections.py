#! /usr/bin/env python3

from Detections import Detections

class MaskDetections(Detections):

    def __init__(self, label, score, mask, mask_threshold):
        self.label = label
        self.score = score
        self.mask = mask
        self.mask_threshold = mask_threshold
        # TODO mask gets converted to polygon
        Detections.__init__(label, score, x, y, shape_type)


   @staticmethod
    def binarize_confidence_mask(self, mask, threshold):
        binary_mask = False
        return binary_mask

        binary_mask, contour, hierarchy, contour_squeeze, poly = self.binarize_confidence_mask(mask, threshold = mask_threshold)