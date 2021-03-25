#! /usr/bin/env python

import torch
import numpy as np
import json
import pickle
import cv2 as cv
import matplotlib.pyplot as plt
import os

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
    # small numbers penalized harder?
    if weights is None:
        weights = np.array([0.5, 0.5])
    # cost = weights[0] * score + weights[1] * iou
    cost = np.sqrt(score * iou)
    return cost


def compute_outcome_image(pred, sample, IOU_THRESH, CONF_THRESH):

    # output: outcome, tn, fp, fn, tp for the image
    dt_bbox = pred['boxes']
    dt_scores = pred['scores']
    gt_bbox = sample['boxes']

    ndt = len(dt_bbox)
    ngt = len(gt_bbox)

    # compute ious

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

    # for each dt_bbox, compute iou and record confidence score
    for i in range(ndt):
        gt_iou = []
        for j in range(ngt):
            iou = compute_iou_bbox(dt_bbox[i], gt_bbox[j])
            gt_iou.append(iou)

        gt_iou_all[i, :] = np.array(gt_iou)

        # NOTE finding the max is insufficient
        # dt_score also important consideration
        # find max iou from gt_iou list
        # idx_max = gt_iou.index(max(gt_iou))
        # iou_max = gt_iou[idx_max]
        # dt_iou.append(iou_max)
        # dt_assignment_idx[i] = idx_max

    # calculate cost of each match/assignment
    cost = np.zeros((ndt, ngt))
    for i in range(ndt):
        for j in range(ngt):
            cost[i,j] = compute_match_cost(dt_scores[i], gt_iou_all[i, j])

    # choose assignment based on max of matrix. If equal cost, we choose the first match
    # higher cost is good - more confidence, more overlap
    # print(cost)

    # for each ndt, pick the max in cost as the match
    # ids_max = cost.argmax(axis=1)  # axis 1 - find max for each dt_bbox
    # TODO check len(ids_max) == ndt
    # TODO sort cost matrix
    # TODO for each iteratively check from the top until a valid assignment is found
    dt_iou = -np.ones((ndt,))
    for i in range(ndt):
        dt_cost = cost[i, :]
        i_cost_srt = np.argsort(dt_cost)[::-1]   # sort costs in descending order
        # choose assignment based on decision thresholds:
        # if iou is 0, it is not an assignment
        # if dt_score and iou are not high enough, it is not an assignment
        # repeat this until end of gt_bbox, we stop after the first
        for j in i_cost_srt:
            if gt_iou_all[i,j] == 0:
                # no overlap
                dt_iou[i] = gt_iou_all[i,j]
                continue
            elif dt_scores[i] < CONF_THRESH or gt_iou_all[i,j] < IOU_THRESH:
                dt_iou[i] = gt_iou_all[i,j]
                # TODO replace this with a single threshold that combines IOU and CONF?
                # confidence or iou overlap too low
                continue
            else:
                # an assignment is made:
                # save dt_match_id, gt_match_id,
                # dt_match, gt_match = True
                dt_match[i] = True
                gt_match[j] = True
                dt_match_id[i] = j
                gt_match_id[j] = i

                # set dt_iou to the correct i,j from gt_iou_all
                dt_iou[i] = gt_iou_all[i,j]

                continue
                # TODO check if this is done right!
                # TODO ensure unique




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
    tp = np.logical_and(dt_match, dt_scores >= CONF_THRESH)
    # tp = np.logical_and(dt_scores >= CONF_THRESH, dt_iou >= IOU_THRESH)
    fp = np.logical_or(np.logical_not(dt_match), np.logical_and(dt_match, dt_scores < CONF_THRESH))
    # fp = np.logical_or(np.logical_and(dt_scores < CONF_THRESH, dt_iou >= IOU_THRESH), dt_iou < IOU_THRESH )
    # dt_assigned = np.logical_or(tp, fp)
    # fn = np.zeros((len(gt_bbox),), dtype=bool)
    fn_gt = np.logical_not(gt_match)
    fn_dt = np.logical_and(dt_match, dt_scores < CONF_THRESH)

    # import code
    # code.interact(local=dict(globals(), **locals()))


    # gt_assigned = np.zeros((len(gt_bbox),), dtype=bool)

    # for j in range(len(gt_bbox)):

    #     if j in dt_assignment_idx:
    #         # find all idx of dt_assignment_idx that j matches to
    #         # j = dt_assignment_idx[?]
    #         dt_idx = []
    #         for i in dt_assignment_idx:
    #             if j == i:
    #                 dt_idx.append(i)

    #         if len(dt_idx) > 0:
    #             for k in dt_idx:
    #                 if tp[k]:
    #                     gt_assigned[j] = True
    # fn = np.logical_not(gt_assigned)


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
                'dt_iou': dt_iou} # intesection over union scores for detections
    return outcomes


def compute_single_pr_over_dataset(model,
                                   dataset,
                                   save_name,
                                   MODEL_IOU_THRESH,
                                   MODEL_CONF_THRESH,
                                   OUTCOME_IOU_THRESH,
                                   OUTCOME_CONF_THRESH):
    model.eval()
    model.to(device)

    tp_sum = 0
    fp_sum = 0
    tn_sum = 0
    fn_sum = 0
    gt_sum = 0
    idx = 0
    for image, sample in dataset:

        image_id = sample['image_id'].item()
        img_name = 'st' + str(image_id).zfill(3)

        # get predictions
        pred, keep = get_prediction_image(model,
                                          image,
                                          MODEL_CONF_THRESH, # set low to allow FNs
                                          MODEL_IOU_THRESH,
                                          device,
                                          class_names)

        outcomes = compute_outcome_image(pred,
                                         sample,
                                         OUTCOME_IOU_THRESH,
                                         OUTCOME_CONF_THRESH)

        # show for given image:
        img_out = show_groundtruth_and_prediction_bbox(image,
                                                       sample,
                                                       pred,
                                                       outcomes=outcomes)

        imgw = cv.cvtColor(img_out, cv.COLOR_RGB2BGR)
        save_img_name = os.path.join('output', save_name, img_name + '_outcome.png')
        cv.imwrite(save_img_name, imgw)
        # winname = 'outcome: ' + img_name
        # cv_imshow(img_out, winname, 1000)

        tp = outcomes['tp']
        fp = outcomes['fp']
        fn_gt = outcomes['fn_gt']
        fn_dt = outcomes['fn_dt']
        fn = np.concatenate((fn_gt, fn_dt), axis=0)
        gt = len(outcomes['gt_match'])

        tp_sum += np.sum(tp)
        fp_sum += np.sum(fp)
        fn_sum += np.sum(fn)
        gt_sum += np.sum(gt)
        # tn_sum += np.sum(tn)
        # end per image outcome calculations

        idx += 1
        if idx > 10:
            break

    print('tp sum: ', tp_sum)
    print('fp sum: ', fp_sum)
    print('fn sum: ', fn_sum)
    print('gt sum: ', gt_sum)
    print('tp + fn = ', tp_sum + fn_sum)
    # print('tn sum: ', tn_sum)

    prec = tp_sum / (tp_sum + fp_sum)
    rec = tp_sum / (tp_sum + fn_sum)

    print('precision = {}'.format(prec))
    print('recall = {}'.format(rec))

    return prec, rec


# ----------------------------------------------------------------------------#

if __name__ == "__main__":

    # TODO
    # import from save_name, a given model/prediction-output file/dataset
    # (need dataset because it has the groundtruth bboxes,
    # and want to show images)
    # for each image in the dataset, read in the predictions (or do inference)
    # call function ^
    # input: image, predictions (bboxes, scores), ground-truth bboxes,
    #        iou threshold, conf threshold
    # output: outcomes, dt_iou, tp, fp, fn, tn

    # TODO this script borrows code from inference.py,
    # should combine/import to reduce redundant code

    # setup device:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load model
    model = build_model(num_classes=2)
    class_names = ['_background_', 'tussock']

    # in output folder, should contain pickle file, model path file, and is where the output goes
    save_name = 'fasterrcnn-serratedtussock-4'
    model_save_path = os.path.join('output', save_name, save_name + '.pth')
    model.load_state_dict(torch.load(model_save_path))

    # setup dataset
    root_dir = os.path.join('SerratedTussockDataset')
    json_file = os.path.join('Annotations', 'via_region_data.json')

    # here, order matters (order in = order out)
    data_save_path = os.path.join('output', save_name, save_name + '.pkl')
    with open(data_save_path, 'rb') as f:
        dataset_tform_test = pickle.load(f)
        dataset_tform_train = pickle.load(f)
        dataset_train = pickle.load(f)
        dataset_val = pickle.load(f)
        dataset_test = pickle.load(f)
        dataloader_test = pickle.load(f)
        dataloader_train = pickle.load(f)
        dataloader_val = pickle.load(f)
        hp = pickle.load(f)
    # with open(save_detector_train_path, 'wb') as f:
    #         pickle.dump(dataset_tform_test, f)
    #         pickle.dump(dataset_tform_train, f)
    #         pickle.dump(dataset_train, f)
    #         pickle.dump(dataset_val, f)
    #         pickle.dump(dataset_test, f)
    #         pickle.dump(dataloader_test, f)
    #         pickle.dump(dataloader_train, f)
    #         pickle.dump(dataloader_val, f)
    #         pickle.dump(hp, f)

    # set thresholds
    # model iou threshold - used for doing non-maxima suppression at the end of model output
    MODEL_IOU_THRESH = 0.5
    # model confidence threshold - used for thresholding model output
    MODEL_CONF_THRESH = 0.1

    # outcome iou threshold - used for determining how much overlap between
    # detection and groundtruth bounding boxes is sufficient to be a TP
    OUTCOME_IOU_THRESH = 0.5
    # outcome confidence threshold - used for determining how much confidence is
    # required to be a TP
    # OUTCOME_CONF_THRESH = np.linspace(start=0.1, stop=0.18, num=2, endpoint=True)
    OUTCOME_CONF_THRESH = np.array([0.4])

    # iterate over confidence threshold
    prec = []
    rec = []
    if (OUTCOME_CONF_THRESH) <= 1:
        p, r = compute_single_pr_over_dataset(model,
                                              dataset_test,
                                              save_name,
                                              MODEL_IOU_THRESH,
                                              MODEL_CONF_THRESH,
                                              OUTCOME_IOU_THRESH,
                                              OUTCOME_CONF_THRESH)

        prec.append(p)
        rec.append(r)
    else:
        for c, conf in enumerate(OUTCOME_CONF_THRESH):
            # get single pr over entire dataset,
            print('outcome confidence threshold: {}'.format(conf))
            p, r = compute_single_pr_over_dataset(model,
                                                dataset_test,
                                                save_name,
                                                MODEL_IOU_THRESH,
                                                MODEL_CONF_THRESH,
                                                OUTCOME_IOU_THRESH,
                                                conf)
            prec.append(p)
            rec.append(r)
            # import code
            # code.interact(local=dict(globals(), **locals()))

    print(prec)
    print(rec)

    rec = np.array(rec)
    prec = np.array(prec)
    fig, ax = plt.subplots()
    ax.plot(rec, prec)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('precision-recall for varying confidence')
    # TODO make dir if does not yet exist
    save_plot_name = os.path.join('output', save_name, save_name + '_pr.png')
    plt.savefig(save_plot_name)

    plt.show()


    import code
    code.interact(local=dict(globals(), **locals()))



