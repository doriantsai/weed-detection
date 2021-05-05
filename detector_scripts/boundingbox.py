#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


class boundingbox(object):
    """
    Boundiong box class
    """
    # see Hungarian function
    def __init__(self,
                 p1,
                 p2=None,
                 gt=None,
                 dt=None,
                 iou=None,
                 score=None,
                 image_id=None,
                 labe=None,
                 class_id=None):
        """
        initialise bounding box as two points
        p1, p2 are a 2-element tuple or numpy array
        conventionally, p1 = closer to image origin
        p2 = farther from image origin (0,0), but maybe this doesn't matter
        this is an assignment problem - could call the Hungarian algorithm in the most general case

        TODO check
        """
        # TODO check p1, p2 are valid types
        self.box = None

        if p2 is None and len(p1) == 4:
            self.box = [p1[0], p1[1], p1[2], p1[3]]
        elif len(p1) == 2 and len(p2) == 2:
            self.box = [p1[0], p1[1], p2[0], p2[1]]
        else:
            ValueError(p1, 'Invalid input type for p1, p2')

        if gt is not None:
            self.gt = gt
        else:
            self.gt = None  # groundtruth index number (tied to list of groundtruth bounding boxes)

        if dt is not None:
            self.dt = dt
        else:
            self.dt = None  # detection index number (tied to list of model predictions)

        # TODO maybe have these lists attached to bounding box class?

        if iou is not None:
            self.iou = iou
        else:
            self.iou = None

        if score is not None:
            self.score = score
        else:
            self.score = None

        if image_id is not None:
            self.image_id = image_id
        else:
            self.image_id = None

        if label is not None:
            self.label = label
        else:
            self.label = None

        if class_id is not None:
            self.class_id = class_id
        else:
            self.class_id = None  # TODO better name for TP/FN/FP?

        self.class_list = ['TP', 'FP', 'FN']

    def class_id(self):
        return self.class_list[self.class_id]

    def tp(self):
        if self.class_id == 'TP':
            return True
        else:
            return False

    def fp(self):
        if self.class_id == 'FP':
            return True
        else:
            return False

    def fn(self):
        if self.class_id == 'FN':
            return True
        else:
            return False








