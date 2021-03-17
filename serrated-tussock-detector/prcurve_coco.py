#! /usr/bin/env python

import torch
import os
import matplotlib.pyplot as plt
# import torchvision
import pickle

from train import build_model
# from SerratedTussockDataset import SerratedTussockDataset
# from get_prediction import get_prediction_image
from engine_st import evaluate


def plot_prcurve(ax,
                 prec_index,
                 ccres)
    """
    plot individual pr curve on given axes
    """
    precision = ccres['precision']
    # [TxRxKxAxM]
    # T - iou threshold
    # R - recall thresholds
    # K - categories
    # A - area ranges
    # M - max detections
    recall = ccres['recall']
    recall_param = ccres['params'].recThrs
    iou_param = ccres['params'].iouThrs
    area_range = ccres['params'].areaRng
    area_range_lbl = ccres['params'].areaRngLbl # how to set these variables?
    max_det = ccres['params'].maxDets

    i_cat = 0
    i_iou = prec_index[0]
    i_area = prec_index[1]
    i_maxd = prec_index[2]

    lbl_str = 'iou' + str(iou_param[i_iou]) + \
              ', a' + str(area_range_lbl[i_area]) + \
              ', md' + str(max_det[i_maxd])
    ax.plot(recall_param,precision[i_iou, :, i_cat, i_maxd, i_area],
            label=lbl_str)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = build_model(num_classes=2)
    CLASS_NAMES = ["_background_", "serrated tussock"]


    save_name = 'fasterrcnn-serratedtussock-4'
    model_save_path = os.path.join('output', save_name, save_name + '.pth')
    data_save_path = os.path.join('output', save_name, save_name + '.pkl')

    print('Loading model from ' + save_name)
    print('Model: ' + model_save_path)
    print('Data: ' + data_save_path)
    model.load_state_dict(torch.load(model_save_path))

    # load stuff:
    # order matters
    with open(data_save_path, 'rb') as f:
        dataset = pickle.load(f)
        dataset_train = pickle.load(f)
        dataset_val = pickle.load(f)
        dataset_test = pickle.load(f)
        dataloader_test = pickle.load(f)
        dataloader_train = pickle.load(f)
        dataloader_val = pickle.load(f)
        hp = pickle.load(f)

    # finally, evaulate on the whole dataset
    confidence_thresh = 0.8
    iou_thresh = 0.5

    # evaluate model
    print('Evaluating model')
    mt_eval, ccres = evaluate(model,
                              dataloader_train,
                              device=device,
                              conf=confidence_thresh,
                              iou=iou_thresh,
                              class_names=CLASS_NAMES)

    precision = ccres['precision']
    # [TxRxKxAxM]
    # T - iou threshold
    # R - recall thresholds
    # K - categories
    # A - area ranges
    # M - max detections
    recall = ccres['recall']
    recall_param = ccres['params'].recThrs
    iou_param = ccres['params'].iouThrs
    area_range = ccres['params'].areaRng
    area_range_lbl = ccres['params'].areaRngLbl # how to set these variables?
    max_det = ccres['params'].maxDets

    print(precision.shape)
    print('iou param')
    print(iou_param)

    print('area range')
    print(area_range)

    print('area label')
    print(area_range_lbl)

    print('max detections')
    print(max_det)

    fig, ax = plt.subplots()
    # plot all:
    plot_prcurve(ax, (0, 0, 0), ccres)
    i_iou = 0
    i_area = 0
    i_maxd = 1
    lbl_str = 'iou' + str(iou_param[i_iou]) + \
              ', a' + str(area_range_lbl[i_area]) + \
              ', md' + str(max_det[i_maxd])
    ax.plot(recall_param,precision[i_iou, :, i_cat, i_maxd, i_area],
            label=lbl_str)
    legend = ax.legend(loc='upper right')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('PR-curve: train set, iou' + str(iou_param[i_iou]))
    plt.savefig(os.path.join('output', save_name, save_name + '-pr-train.png'))
    plt.show()


    import code
    code.interact(local=dict(globals(), **locals()))