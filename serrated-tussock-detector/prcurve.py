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


def compute_outcome_image(pred, sample, IOU_THRESH, CONF_THRESH):

    # output: outcome, tn, fp, fn, tp for the image
    dt_bbox = pred['boxes']
    dt_scores = pred['scores']
    gt_bbox = sample['boxes']
    # compute ious
    dt_iou = []
    tp = np.zeros((len(dt_bbox),), dtype=bool)
    fp = np.zeros((len(dt_bbox),), dtype=bool)
    fn = np.zeros((len(dt_bbox),), dtype=bool)
    gt_iou_all = np.zeros((len(dt_bbox), len(gt_bbox)))

    dt_assignment = []
    for i in range(len(dt_bbox)):
        gt_iou = []
        for j in range(len(gt_bbox)):
            iou = compute_iou_bbox(dt_bbox[i], gt_bbox[j])
            gt_iou.append(iou)

        gt_iou_all[i, :] = np.array(gt_iou)

        # find max iou from gt_iou list
        idx_max = gt_iou.index(max(gt_iou))
        iou_max = gt_iou[idx_max]
        dt_iou.append(iou_max)
        dt_assignment.append(idx_max)

    # assign dt_bbox to gt_bbox
    # need to find any gt_indices that are not in dt_assignment
    # for each index in gt_bbox, search dt_assignment, if index matches an
    # element in dt_assignment, then ok/set gt_assignment = True (def False)
    # any falses are true negatives
    gt_assigned = np.zeros((len(gt_bbox),), dtype=bool)
    for j in range(len(gt_bbox)):
        if j in dt_assignment:
            gt_assigned[j] = True
    # print('gt_assigned = ', gt_assigned)
    # gt2dt_assignment = np.array(dt_assignment)

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
    tp = np.logical_and(dt_scores >= CONF_THRESH, dt_iou >= IOU_THRESH)
    fp = np.logical_or(np.logical_and(dt_scores < CONF_THRESH, dt_iou >= IOU_THRESH), dt_iou < IOU_THRESH )
    # fn = np.zeros((len(gt_bbox),), dtype=bool)
    fn = np.logical_not(gt_assigned)

    # define outcome as an array, each element corresponding to
    # tp/fp/fn for 0/1/2, respectively
    outcome = -np.ones((len(dt_bbox),), dtype=np.int16)
    for i in range(len(outcome)):
        if tp[i]:
            outcome[i] = 0
        elif fp[i]:
            outcome[i] = 1
        elif fn[i]:
            outcome[i] = 2

    return outcome, fn, tp, fp, dt_iou


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

        outcome, fn, tp, fp , dt_iou = compute_outcome_image(pred,
                                                             sample,
                                                             OUTCOME_IOU_THRESH,
                                                             OUTCOME_CONF_THRESH)

        # show for given image:
        img_out = show_groundtruth_and_prediction_bbox(image,
                                                       sample,
                                                       pred,
                                                       iou=dt_iou,
                                                       outcome=outcome,
                                                       falseneg=fn)

        imgw = cv.cvtColor(img_out, cv.COLOR_RGB2BGR)
        save_img_name = os.path.join('output', save_name, img_name + '_outcome.png')
        cv.imwrite(save_img_name, imgw)
        # winname = 'outcome: ' + img_name
        # cv_imshow(img_out, winname, 1000)

        tp_sum += np.sum(tp)
        fp_sum += np.sum(fp)
        fn_sum += np.sum(fn)
        # tn_sum += np.sum(tn)
        # end per image outcome calculations

    print('tp sum: ', tp_sum)
    print('fp sum: ', fp_sum)
    print('fn sum: ', fn_sum)
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
        dataset = pickle.load(f)
        dataset_train = pickle.load(f)
        dataset_val = pickle.load(f)
        dataset_test = pickle.load(f)
        dataloader_test = pickle.load(f)
        dataloader_train = pickle.load(f)
        dataloader_val = pickle.load(f)
        hp = pickle.load(f)


    # set thresholds
    # model iou threshold - used for doing non-maxima suppression at the end of model output
    MODEL_IOU_THRESH = 0.5
    # model confidence threshold - used for thresholding model output
    MODEL_CONF_THRESH = 0.5
    # outcome iou threshold - used for determining how much overlap between
    # detection and groundtruth bounding boxes is sufficient to be a TP
    OUTCOME_IOU_THRESH = 0.6
    # outcome confidence threshold - used for determining how much confidence is
    # required to be a TP
    OUTCOME_CONF_THRESH = np.linspace(start=0.1, stop=0.9, num=3, endpoint=True)

    # iterate over confidence threshold
    prec = []
    rec = []
    for c, conf in enumerate(OUTCOME_CONF_THRESH):
        # get single pr over entire dataset,
        print('outcome confidence threshold: {}'.format(conf))
        p, r = compute_single_pr_over_dataset(model,
                                              dataset_train,
                                              save_name,
                                              MODEL_IOU_THRESH,
                                              MODEL_CONF_THRESH,
                                              OUTCOME_IOU_THRESH,
                                              conf)
        prec.append(p)
        rec.append(r)

    print(prec)
    print(rec)

    rec = np.array(rec)
    prec = np.array(prec)
    fig, ax = plt.subplots()
    ax.plot(rec, prec)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('precision-recall for varying confidence')
    save_plot_name = os.path.join('output', save_name, save_name + '_pr.png')
    plt.savefig(save_plot_name)

    plt.show()


    import code
    code.interact(local=dict(globals(), **locals()))



