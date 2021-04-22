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

from scipy.interpolate import interp1d
from train import build_model
from PIL import Image
from inference import show_groundtruth_and_prediction_bbox, cv_imshow, infer_dataset
from get_prediction import get_prediction_image, threshold_predictions
from find_file import find_file


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


def compute_single_pr_over_dataset(model,
                                   dataset,
                                   predictions,
                                   save_name,
                                   NMS_IOU_THRESH,
                                   MODEL_REFJECT_CONF_THRESH,
                                   DECISION_IOU_THRESH,
                                   DECISION_CONF_THRESH,
                                   imsave=False,
                                   dataset_annotations=None):
    model.eval()
    model.to(device)

    tp_sum = 0
    fp_sum = 0
    tn_sum = 0
    fn_sum = 0
    gt_sum = 0
    dt_sum = 0
    idx = 0
    dataset_outcomes = []
    for image, sample in dataset:

        image_id = sample['image_id'].item()
        if dataset_annotations is not None:
            img_name = dataset_annotations[image_id]['filename'][:-4]
        else:
            img_name = 'st' + str(image_id).zfill(3)

        # get predictions
        pred = predictions[idx]
        pred = threshold_predictions(pred, MODEL_REFJECT_CONF_THRESH)
        # now, reject all those below MODEL_REJECT_CONF_THRESH

        # pred, keep = get_prediction_image(model,
        #                                   image,
        #                                   MODEL_REFJECT_CONF_THRESH, # set low to allow FNs
        #                                   NMS_IOU_THRESH,
        #                                   device,
        #                                   class_names)

        outcomes = compute_outcome_image(pred,
                                         sample,
                                         DECISION_IOU_THRESH,
                                         DECISION_CONF_THRESH)

        # show for given image:
        if imsave:
            img_out = show_groundtruth_and_prediction_bbox(image,
                                                        sample,
                                                        pred,
                                                        outcomes=outcomes)

            imgw = cv.cvtColor(img_out, cv.COLOR_RGB2BGR)
            save_folder_name = os.path.join('output',
                                            save_name,
                                            'outcomes',
                                            'conf_thresh_' + str(DECISION_CONF_THRESH))
            if not os.path.exists(save_folder_name):
                os.mkdir(save_folder_name)
            save_img_name = os.path.join(save_folder_name, img_name + '_outcome.png')
            cv.imwrite(save_img_name, imgw)
            # winname = 'outcome: ' + img_name
            # cv_imshow(img_out, winname, 1000)

        tp = outcomes['tp']
        fp = outcomes['fp']
        fn_gt = outcomes['fn_gt']
        fn_dt = outcomes['fn_dt']
        fn = np.concatenate((fn_gt, fn_dt), axis=0)
        gt = len(outcomes['gt_match'])
        dt = len(outcomes['dt_match'])

        tp_sum += np.sum(tp)
        fp_sum += np.sum(fp)
        fn_sum += np.sum(fn)
        gt_sum += np.sum(gt)
        dt_sum += np.sum(dt)
        # end per image outcome calculations

        dataset_outcomes.append(outcomes)
        idx += 1

    print('tp sum: ', tp_sum)
    print('fp sum: ', fp_sum)
    print('fn sum: ', fn_sum)
    print('dt sum: ', dt_sum)
    print('tp + fp = ', tp_sum + fp_sum)
    print('gt sum: ', gt_sum)
    print('tp + fn = ', tp_sum + fn_sum)

    prec = tp_sum / (tp_sum + fp_sum)
    rec = tp_sum / (tp_sum + fn_sum)

    print('precision = {}'.format(prec))
    print('recall = {}'.format(rec))

    f1score = compute_f1score(prec, rec)

    print('f1 score = {}'.format(f1score))
    # 1 is good, 0 is bad

    return dataset_outcomes, prec, rec, f1score


def compute_f1score(p, r):
    return 2 * (p * r) / (p + r)


def extend_pr(prec, rec, conf, n_points=101, PLOT=False, save_name='outcomes'):
    """ extend precision and recall vectors from 0-1 """

    # "smooth" out precision-recall curve by taking max of precision points
    # for when r is very small (takes the top of the stairs )
    # take max:
    diff_thresh = 0.001
    dif = np.diff(rec)
    prec_new = [prec[0]]
    for i, d in enumerate(dif):
        if d < diff_thresh:
            prec_new.append(prec_new[-1])
        else:
            prec_new.append(prec[i+1])
    prec_new = np.array(prec_new)
    # print(prec_new)

    if PLOT:
        fig, ax = plt.subplots()
        ax.plot(rec, prec, marker='o', linestyle='dashed', label='original')
        ax.plot(rec, prec_new, marker='x', color='red', linestyle='solid', label='max-binned')
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title('prec-rec, max-binned')
        ax.legend()
        save_plot_name = os.path.join('output', save_name, save_name + '_test_pr_smooth.png')
        plt.savefig(save_plot_name)

    # now, expand prec/rec values to extent of the whole precision/recall space:
    rec_x = np.linspace(0, 1, num=n_points, endpoint=True)

    # create prec_x by concatenating vectors first
    prec_temp = []
    rec_temp = []
    conf_temp = []
    for r in rec_x:
        if r < rec[0]:
            prec_temp.append(prec_new[0])
            rec_temp.append(r)
            conf_temp.append(conf[0])

    for i, r in enumerate(rec):
        prec_temp.append(prec_new[i])
        rec_temp.append(r)
        conf_temp.append(conf[i])

    for r in rec_x:
        if r >= rec[-1]:
            prec_temp.append(0)
            rec_temp.append(r)
            conf_temp.append(conf[-1])

    prec_temp = np.array(prec_temp)
    rec_temp = np.array(rec_temp)
    conf_temp = np.array(conf_temp)

    # now interpolate:
    # prec_x = np.interp(rec_x, rec_temp, prec_temp)
    prec_interpolator = interp1d(rec_temp, prec_temp, kind='linear')
    prec_x = prec_interpolator(rec_x)

    conf_interpolator = interp1d(rec_temp, conf_temp, kind='linear')
    conf_x = conf_interpolator(rec_x)

    if PLOT:
        fig, ax = plt.subplots()
        ax.plot(rec_temp, prec_temp, color='blue', linestyle='dashed', label='combined')
        ax.plot(rec, prec_new, marker='x', color='red', linestyle='solid', label='max-binned')
        ax.plot(rec_x, prec_x, color='green', linestyle='dotted', label='interp')
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title('prec-rec, interpolated')
        ax.legend()
        save_plot_name = os.path.join('output', save_name, save_name + '_test_pr_interp.png')
        plt.savefig(save_plot_name)
        # plt.show()

    # make the "official" plot:
    p_out = prec_x
    r_out = rec_x
    c_out = conf_x
    return p_out, r_out, c_out


def compute_ap(p, r):
    """ compute ap score """
    n = len(r) - 1
    ap = np.sum( (r[1:n] - r[0:n-1]) * p[1:n] )
    return ap


def get_confidence_from_pr(p, r, c, f1, pg=None, rg=None):
    """ interpolate confidence threshold from p, r and goal values. If no goal
    values, then provide the "best" conf, corresponding to the pr-pair closest
    to the ideal (1,1)
    """
    # TODO check p, r are valid
    # if Pg is not none,
    #   check Pg is valid, if not, set it at appropriate value
    # if Rg is not none, then
    #   check if Rg is valid, if not, set appropriate value
    if pg is not None and rg is not None:
        # print('TODO: check if pg, rg are valid')
        # find the nearest pg-pair, find the nearest index to said pair
        prg = np.array([pg, rg])
        pr = np.array([p, r])
        dist = np.sqrt(np.sum((pr - prg)**2, axis=1))
        # take min dist
        idistmin = np.argmin(dist, axis=1)
        pact = p[idistmin]
        ract = r[idistmin]

        # now interpolate for cg
        f = interp1d(r, c)
        cg = f(ract)

    elif pg is None and rg is not None:
        # TODO search for best pg given rg
        # pg = 1
        # we use rg to interpolate for cg:
        f = interp1d(r, c, bounds_error=False, fill_value=(c[0], c[-1]))
        cg = f(rg)

    elif rg is None and pg is not None:
        # we use pg to interpolate for cg
        # print('using p to interp c')
        f = interp1d(p, c, bounds_error=False, fill_value=(c[-1], c[0]))
        cg = f(pg)

    else:
        # find p,r closest to (1,1) being max of f1 score
        if1max = np.argmax(f1)
        rmax = r[if1max]
        f = interp1d(r, c)
        cg = f(rmax)

    # we use r to interpolate with c

    return cg

# ---------------------------------------------------------------------------- #
if __name__ == "__main__":

    # load model
    # load datasets
    # infer on relevant dataset
    # iterate over decision threshold

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = build_model(num_classes=2)
    class_names = ['__background__', 'tussock']

    save_name = 'Tussock_v0_11'
    model_name = 'Tussock_v0_11'
    model_folder = os.path.join('output', model_name)
    saved_model_name = find_file('.pth', model_folder)
    # save_path = os.path.join('output', save_name, save_name + '.pth')
    saved_model_path = os.path.join(model_folder, saved_model_name[0])
    model.load_state_dict(torch.load(saved_model_path))
    print('loading model: {}'.format(saved_model_path))

    # dataset location
    dataset_name = 'Tussock_v0'
    data_save_path = os.path.join('output',
                                'dataset',
                                dataset_name,
                                dataset_name + '.pkl')
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

    # infer on dataset with 0-decision threshold
    out_raw, predictions = infer_dataset(model,
                                        dataset_test,
                                        confidence_threshold=0,
                                        iou_threshold=0.5,
                                        save_folder_name=save_name,
                                        device=device,
                                        output_folder='prcurve_predictions',
                                        imshow=False,
                                        img_name_suffix='_prcurve_0')

    # compute AP
    # choose best F1 score?
    # report 95% recall and 95% precision values

    NMS_IOU_THRESH = 0.5
    DECISION_IOU_THRESH = 0.5
    # DECISION_CONF_THRESH = 0.5
    DECISION_CONF_THRESH = np.linspace(0.95, 0.05, num=11, endpoint=True)
    DECISION_CONF_THRESH = np.array(DECISION_CONF_THRESH, ndmin=1)

    prec = []
    rec = []
    f1score = []
    start_time = time.time()
    for c, conf in enumerate(DECISION_CONF_THRESH):
        print('{}: outcome confidence threshold: {}'.format(c, conf))
        dataset_outcomes, p, r, f1 = compute_single_pr_over_dataset(model,
                                                                    dataset_test,
                                                                    predictions,
                                                                    save_name,
                                                                    NMS_IOU_THRESH,
                                                                    conf,
                                                                    DECISION_IOU_THRESH,
                                                                    conf,
                                                                    imsave=False,
                                                                    dataset_annotations=dataset_test.dataset.annotations)

        # single-line for copy/paste in terminal:
        # d, p, r, f1 = compute_single_pr_over_dataset(model, dataset_test, predictions, NMS_IOU_THRESH, conf, DECISION_IOU_THRESH, conf, imsave=True)
        prec.append(p)
        rec.append(r)
        f1score.append(f1)
        # do we care about dataset_outcomes?

    end_time = time.time()

    sec = end_time - start_time
    print('training time: {} sec'.format(sec))
    print('training time: {} min'.format(sec / 60.0))
    print('training time: {} hrs'.format(sec / 3600.0))

    # plot raw PR curve
    rec = np.array(rec)
    prec = np.array(prec)
    fig, ax = plt.subplots()
    ax.plot(rec, prec, marker='o', linestyle='dashed')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('precision-recall for varying confidence')
    save_plot_name = os.path.join('output', save_name, save_name + '_test_pr.png')
    plt.savefig(save_plot_name)
    # plt.show()

    # plot F1score
    f1score = np.array(f1score)
    fig, ax = plt.subplots()
    ax.plot(rec, f1score, marker='o', linestyle='dashed')
    plt.xlabel('recall')
    plt.ylabel('f1 score')
    plt.title('f1 score vs recall for varying confidence')
    save_plot_name = os.path.join('output', save_name, save_name + '_test_f1r.png')
    plt.savefig(save_plot_name)
    # plt.show()

    # smooth the PR curve: take the max precision values along the recall curve
    # we do this by binning the recall values, and taking the max precision from
    # each bin

    p_final, r_final, c_final, = extend_pr(prec, rec, DECISION_CONF_THRESH)
    ap = compute_ap(p_final, r_final)

    fig, ax = plt.subplots()
    ax.plot(r_final, p_final)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('prec-rec curve, iou={}, ap = {:.2f}'.format(DECISION_IOU_THRESH, ap))
    # ax.legend()
    save_plot_name = os.path.join('output', save_name, save_name + '_test_pr_final.png')
    plt.savefig(save_plot_name)
    plt.show()

    print('ap score: {:.5f}'.format(ap))
    print('max f1 score: {:.5f}'.format(max(f1score)))

    # save ap, f1score, precision, recall, etc
    res = {'precision': p_final,
           'recall': r_final,
           'ap': ap,
           'f1score': f1score,
           'confidence': c_final}
    save_file = os.path.join('output', save_name, save_name + '_prcurve.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(res, f)

    # choose to do/save outcomes from a specific setting:
    # conf = 0.5
    # d, p, r, f1 = compute_single_pr_over_dataset(model,
    #                                             dataset_test,
    #                                             predictions,
    #                                             NMS_IOU_THRESH,
    #                                             conf,
    #                                             DECISION_IOU_THRESH,
    #                                             conf,
    #                                             imsave=True,
    #                                             dataset_annotations=dataset_test.dataset.annotations)

    # TODO
    # 95% precision, what confidence value?
    # 95% recall, what confidence value?
    import code
    code.interact(local=dict(globals(), **locals()))