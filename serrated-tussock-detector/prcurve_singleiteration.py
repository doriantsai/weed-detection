#! /usr/bin/env python

"""
get prcurve efficiently by running detections on dataset once at a 0-decision
threshold then iterating over these decisions and incrementally increasing the
decision threshold to get the rest of the prcurve
"""

import torch
import numpy as np
import json
import pickle
import cv2 as cv
import matplotlib.pyplot as plt
import os
import time

from train import build_model
from PIL import Image
from inference import show_groundtruth_and_prediction_bbox, cv_imshow
from get_prediction import get_prediction_image


def compute_iou_bbox(boxA, boxB):
    """
    compute intersection over union for bounding boxes
    box = [xmin, ymin, xmax, ymax], tensors
    """
    # determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute area of boxA and boxB
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute iou
    iou = interArea / (float(boxAArea + boxBArea - interArea))

    return iou


def compute_match_cost(score, iou, weights=None):
    # compute match cost based on score and iou
    # this is the arithmetic mean, consider the geometric mean
    # https://en.wikipedia.org/wiki/Geometric_mean
    # which automatically punishes the 0 IOU case because of the multiplication
    # small numbers penalized harder
    if weights is None:
        weights = np.array([0.5, 0.5])
    # cost = weights[0] * score + weights[1] * iou # classic weighting
    cost = np.sqrt(score * iou)
    return cost


def compute_outcome_image(pred,
                          sample,
                          DECISION_IOU_THRESH,
                          DECISION_CONF_THRESH):

    # output: outcome, tn, fp, fn, tp for the image
    dt_bbox = pred['boxes']
    dt_scores = pred['scores']
    gt_bbox = sample['boxes']

    ndt = len(dt_bbox)
    ngt = len(gt_bbox)

    # compute ious
    # for each dt_bbox, compute iou and record confidence score
    gt_iou_all = np.zeros((ndt, ngt))

    # for each detection bounding box, find the best-matching groundtruth bounding box
    # gt/dt_match is False - not matched, True - matched
    dt_match = np.zeros((ndt,), dtype=bool)
    gt_match = np.zeros((ngt,), dtype=bool)

    # gt/dt_match_id is the index of the match vector
    # eg. gt = [1 0 2 -1] specifies that
    # gt_bbox[0] is matched to dt_bbox[1],
    # gt_bbox[1] is matched to dt_bbox[0],
    # gt_bbox[2] is matched to dt_bbox[2],
    # gt_bbox[3] is not matched to any dt_bbox
    dt_match_id = -np.ones((ndt,), dtype=int)
    gt_match_id = -np.ones((ngt,), dtype=int)

    # dt_assignment_idx = -np.ones((len(dt_bbox),), dtype=int)
    dt_iou = -np.ones((ndt,))

    for i in range(ndt):
        gt_iou = []
        for j in range(ngt):
            iou = compute_iou_bbox(dt_bbox[i], gt_bbox[j])
            gt_iou.append(iou)
        gt_iou_all[i, :] = np.array(gt_iou)

        # NOTE finding the max may be insufficient
        # dt_score also important consideration
        # find max iou from gt_iou list
        idx_max = gt_iou.index(max(gt_iou))
        iou_max = gt_iou[idx_max]
        dt_iou[i] = iou_max

    # calculate cost of each match/assignment
    cost = np.zeros((ndt, ngt))
    for i in range(ndt):
        for j in range(ngt):
            cost[i,j] = compute_match_cost(dt_scores[i], gt_iou_all[i, j])

    # choose assignment based on max of matrix. If equal cost, we choose the first match
    # higher cost is good - more confidence, more overlap

    for j in range(ngt):
        dt_cost = cost[:, j]
        i_dt_cost_srt = np.argsort(dt_cost)[::-1]

        for i in i_dt_cost_srt:
            if (not dt_match[i]) and \
                (dt_scores[i] >= DECISION_CONF_THRESH) and \
                    (gt_iou_all[i,j] >= DECISION_IOU_THRESH):
                # if the detection box in question is not already matched
                dt_match[i] = True
                gt_match[j] = True
                dt_match_id[i] = j
                gt_match_id[j] = i
                dt_iou[i] = gt_iou_all[i, j]
                break
                # stop iterating once the highest-scoring detection satisfies the criteria\


    tp = np.zeros((ndt,), dtype=bool)
    fp = np.zeros((ndt,), dtype=bool)
    # fn = np.zeros((ngt,), dtype=bool)
    # now determine if TP, FP, FN
    # TP: if our dt_bbox is sufficiently within a gt_bbox with a high-enough confidence score
    #   we found the right thing
    # FP: if our dt_bbox is a high confidence score, but is not sufficiently within a gt_bbox (iou is low)
    #   we found the wrong thing
    # FN: if our dd_bbox is within a gt_bbox, but has low confidence score
    #   we didn't find the right thing
    # TN: basically everything else in the scene, not wholely relevant for PR-curves

    # From the slides:
    # TP: we correctly found the weed
    # FP: we think we found the weed, but we did not
    # FN: we missed a weed
    # TN: we correctly ignored everything else we were not interested in (doesn't really show up for binary detectors)
    dt_iou = np.array(dt_iou)
    dt_scores = np.array(dt_scores)

    # TP: if dt_match True, also, scores above certain threshold
    tp = np.logical_and(dt_match, dt_scores >= DECISION_CONF_THRESH)
    fp = np.logical_or(np.logical_not(dt_match) , \
        np.logical_and(dt_match, dt_scores < DECISION_CONF_THRESH))
    fn_gt = np.logical_not(gt_match)
    fn_dt = np.logical_and(dt_match, dt_scores < DECISION_CONF_THRESH)

    # define outcome as an array, each element corresponding to
    # tp/fp/fn for 0/1/2, respectively
    dt_outcome = -np.ones((ndt,), dtype=np.int16)
    for i in range(ndt):
        if tp[i]:
            dt_outcome[i] = 0
        elif fp[i]:
            dt_outcome[i] = 1
        elif fn_dt[i]:
            dt_outcome[i] = 2

    gt_outcome = -np.ones((ngt,), dtype=np.int16)
    for j in range(ngt):
        if fn_gt[j]:
            gt_outcome = 1

    # package outcomes
    outcomes = {'dt_outcome': dt_outcome, # detections, integer index for tp/fp/fn
                'gt_outcome': gt_outcome, # groundtruths, integer index for fn
                'dt_match': dt_match, # detections, boolean matched or not
                'gt_match': gt_match, # gt, boolean matched or not
                'fn_gt': fn_gt, # boolean for gt false negatives
                'fn_dt': fn_dt, # boolean for dt false negatives
                'tp': tp, # true positives for detections
                'fp': fp, # false positives for detections
                'dt_iou': dt_iou, # intesection over union scores for detections
                'dt_scores': dt_scores,
                'gt_iou_all': gt_iou_all,
                'cost': cost}
    return outcomes
