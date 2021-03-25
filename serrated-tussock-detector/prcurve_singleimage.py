#! /usr/bin/env python

# NOTE This code is superseded by prcurve.py

import torch
import os
import pickle
import json
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from train import build_model

from PIL import Image
from get_prediction import get_prediction_image
from inference import show_groundtruth_and_prediction_bbox, cv_imshow
from SerratedTussockDataset import ToTensor, Rescale
from torchvision.transforms import functional as tvtransfunc
from prcurve import compute_iou_bbox, compute_outcome_image


# ---------------------------------------------------------------------------- #
if __name__ == "__main__":

    # TODO
    # create image/gt bounding boxes and detection bounding boxes,
    # save as a file (json)
    # import said file
    # for each gt_bbox:
    #   for each dt_bbox:
    #       compute iou
    #       check confidence score
    #       decide if FP or TP
    #       decide if FN or FP
    root_dir = os.path.join('/home','dorian','Data','SerratedTussockDataset_v2')

    # setup device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load model
    num_classes = 2
    class_names = ["_background_", "serrated tussock"]
    # # load instance segmentation model pre-trained on coco:
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    # # get number of input features for the classifier
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # # replace the pre-trained head with a new one
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model = build_model(num_classes=2)
    save_name = 'fasterrcnn-serratedtussock-4'
    save_path = os.path.join('output', save_name, save_name + '.pth')
    # model.load_state_dict(torch.load(save_path))
    model.load_state_dict(torch.load(save_path))
    model.to(device)

    # load gt annotations:
    idx = 106 # 10, 100, 90, 70
    json_file = os.path.join('Annotations', 'via_region_data.json')
    annotations = json.load(open(os.path.join(root_dir, json_file)))
    annotations = list(annotations.values())

    # get image
    img_name = annotations[idx]['filename']
    img_path = os.path.join(root_dir, 'Images', img_name)
    print(img_path)
    img = Image.open(img_path).convert("RGB")

    # select image - corresponding with img_name
    bbox = annotations[idx]['regions']
    gt_bbox = []
    for i in range(len(bbox)):
        xmin = bbox[str(i)]['shape_attributes']['x']
        ymin = bbox[str(i)]['shape_attributes']['y']
        width = bbox[str(i)]['shape_attributes']['width']
        height = bbox[str(i)]['shape_attributes']['height']
        xmax = xmin + width
        ymax = ymin + height
        gt_bbox.append([xmin, ymin, xmax, ymax])

    gt_bbox = torch.as_tensor(gt_bbox, dtype=torch.float64)
    smp = {'boxes': gt_bbox}

    # recale and convert to tensors
    tform_rsc = Rescale(800)
    img, smp = tform_rsc(img, smp)
    tform_totensor = ToTensor()
    img, smp = tform_totensor(img, smp)

    print(img)
    print(smp)

    MODEL_CONF_THRESH = 0.1
    MODEL_IOU_THRESH = 0.5
    pred, keep = get_prediction_image(model,
                                        img,
                                        MODEL_CONF_THRESH, # set low to allow FNs
                                        MODEL_IOU_THRESH,
                                        device,
                                        class_names)

    OUTCOME_CONF_THRESH = 0.5
    OUTCOME_IOU_THRESH = 0.5
    outcomes = compute_outcome_image(pred,
                                    smp,
                                    OUTCOME_IOU_THRESH,
                                    OUTCOME_CONF_THRESH)
    #    outcomes = {'dt_outcome': dt_outcome, # detections, integer index for tp/fp/fn
    #             'gt_outcome': gt_outcome, # groundtruths, integer index for fn
    #             'dt_match': dt_match, # detections, boolean matched or not
    #             'gt_match': gt_match, # gt, boolean matched or not
    #             'fn_gt': fn_gt, # boolean for gt false negatives
    #             'fn_dt': fn_dt, # boolean for dt false negatives
    #             'tp': tp, # true positives for detections
    #             'fp': fp, # false positives for detections
    #             'dt_iou': dt_iou} # intesection over union scores for detections

    tp = outcomes['tp']
    fp = outcomes['fp']
    fn_gt = outcomes['fn_gt']
    fn_dt = outcomes['fn_dt']
    fn = np.concatenate((fn_gt, fn_dt), axis=0)
    gt = len(outcomes['gt_match'])

    print('true positives (if true)')
    print(tp)
    print('false positives (if true)')
    print(fp)
    print('false negatives (if true)')
    print(fn)

    tp_sum = np.sum(tp)
    fp_sum = np.sum(fp)
    fn_sum = np.sum(fn)
    gt_sum = np.sum(gt)

    print('tp sum: ', tp_sum)
    print('fp sum: ', fp_sum)
    print('fn sum: ', fn_sum)
    print('gt sum: ', gt_sum)
    print('tp + fn = ', tp_sum + fn_sum)

     # show for given image:
    img_out = show_groundtruth_and_prediction_bbox(img,
                                                    smp,
                                                    pred,
                                                    outcomes=outcomes)

    imgw = cv.cvtColor(img_out, cv.COLOR_RGB2BGR)
    save_img_name = os.path.join('output', save_name, img_name[:-4] + '_outcome.png')
    cv.imwrite(save_img_name, imgw)
    winname = 'pr, single image: ' + img_name
    cv_imshow(imgw, winname, 2000)


    # compute Precision, Recall:
    tps = np.sum(tp)
    fps = np.sum(fp)
    fns = np.sum(fn)

    prec = tps / (tps + fps)
    rec = tps / (tps + fns)

    print('precision = {}'.format(prec))
    print('recall = {}'.format(rec))



    import code
    code.interact(local=dict(globals(), **locals()))