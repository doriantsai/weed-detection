#! /usr/bin/env python

import torch
import os
import pickle
import json
import cv2 as cv
import matplotlib.pyplot as plt

from PIL import Image
from inference import show_groundtruth_and_prediction_bbox, cv_imshow
from SerratedTussockDataset import ToTensor, Rescale
from torchvision.transforms import functional as tvtransfunc


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
    IOU_THRESH = 0.5
    CONF_THRESH = 0.7


    root_dir = os.path.join('/home','dorian','Data','SerratedTussockDataset_v1')

    # load gt annotations:
    idx = 10
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

    # show:
    # img_gt = show_groundtruth_and_prediction_bbox(img, smp)
    # winname = 'groundtruth bboxes: ' + img_name
    # cv_imshow(img_gt, winname)

    # plt.imshow(img_gt)
    # plt.show()
    # now get/show detections:
    # for now, manually define them:
    # [xmin, ymin, xmax, ymax]
    dt_bbox = [[38, 39, 412, 464],
            [240, 280, 851, 705],
            [644, 73, 1023, 329],
            [1010, 490, 1153, 724]]
    dt_bbox = torch.as_tensor(dt_bbox, dtype=torch.float64)

    # in corresponding order, the confidence scores
    dt_scores = [0.6, 0.9, 0.8, 0.4]

    # put into dictionary and then list form
    pred = {
        'boxes': dt_bbox,
        'scores': dt_scores
    }

    # show
    img_gp = show_groundtruth_and_prediction_bbox(img, smp, pred)
    winname = 'gt (blue) and pred (pred): ' + img_name
    cv_imshow(img_gp, winname, 2000)

    # compute IOU for each dt_bbox wrt gt_bbox:
    igt = 0
    idt = 0
    boxA = gt_bbox[igt, :]
    boxB = dt_bbox[idt, :]
    compute_iou_bbox(boxA, boxB)

    # show IOU for each dt_box:
    dt_iou = []
    for i in range(len(dt_bbox[:, 0])):
        # compute iou for nearest/most overlapping gt_bbox
        gt_iou = []
        for j in range(len(gt_bbox[:, 0])):
            iou = compute_iou_bbox(dt_bbox[i, :], gt_bbox[j, :])
            gt_iou.append(iou)

        # find max iou from gt_iou list
        idx_max = gt_iou.index(max(gt_iou))
        iou_max = gt_iou[idx_max]
        dt_iou.append(iou_max)

    print(dt_iou)

    img_iou = show_groundtruth_and_prediction_bbox(img, smp, pred, iou=dt_iou)
    imgw= cv.cvtColor(img_iou, cv.COLOR_RGB2BGR)
    save_img_name = os.path.join('output', img_name[:-4] + '_iou.png')
    cv.imwrite(save_img_name, imgw)
    winname = 'iou: ' + img_name
    cv_imshow(img_iou, winname, 2000)





    import code
    code.interact(local=dict(globals(), **locals()))