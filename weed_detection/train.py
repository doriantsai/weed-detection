#! /usr/bin/env python

"""
Purpose: Train serrated tussock detector
@author: Dorian Tsai
@email: dorian.tsai@gmail.com
"""

import os
# import numpy as np
import torch
# import torch.utils.data
import torchvision
# import utils
import json
import pickle
import datetime
import time

# import cv2 as cv
# import matplotlib.pyplot as plt
# HACK make WeedDataset module visible to this place
# TODO
# from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from engine_st import train_one_epoch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

torch.manual_seed(42)


# ------------------------------------------------------------------- #
def build_model(num_classes):
    """
    build the fasterrcnn model for set number of classes (num_classes)
    """

    # load instance segmentation model pre-trained on coco:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# -------------------------------------------------------------------- #
if __name__ == "__main__":

    print('Train model')

    # ------------------------------ #
    LOAD_DATASET = True
    # dataset_name = 'Tussock_v0'
    # dataset_name = 'Horehound_v0'
    dataset_name = 'Tussock_v1'

    if LOAD_DATASET:
        # dataset pickle file should be located in:
        # save_dataset_folder = os.path.join('output','dataset', dataset_name)
        # dataset_name + '.pkl'
        save_dataset_path = os.path.join('dataset',
                                         'dataset',
                                         dataset_name,
                                         dataset_name + '.pkl')
        print('loading dataset: {}'.format(save_dataset_path))
        if os.path.isfile(save_dataset_path):
            # load the data
            with open(save_dataset_path, 'rb') as f:
                ds_train = pickle.load(f)
                ds_test = pickle.load(f)
                ds_val = pickle.load(f)
                dl_train = pickle.load(f)
                dl_test = pickle.load(f)
                dl_val = pickle.load(f)
                hp_train = pickle.load(f)
                hp_test = pickle.load(f)

        else:
            print('File does not exist: {}'.format(save_dataset_path))
    else:
        # run split_dataset
        # TODO talk to Gavin about restructuring code
        print('TODO: run create_datasets')

    print('Training model on dataset: {}'.format(dataset_name))

    # ------------------------------ #
    # directories
    # TODO add date/time to filename
    # model_save_name = 'Tussock_v0_15'
    # model_save_name = 'Horehound_v0_1'

    model_save_name = 'Tussock_v1_01'

    # save_name = 'fasterrcnn-serratedtussock-4'
    save_folder = os.path.join('output', model_save_name)
    # if not os.path.isdir(save_folder):
    #     os.mkdir(save_folder)
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, model_save_name + '.pth')
    print('Model save name: {}'.format(model_save_name))

    # setup device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # build model
    # setup number of classes (1 background, 1 class - serrated tussock)
    model = build_model(num_classes=2)
    model.to(device)

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=hp_train['learning_rate'],
                                momentum=hp_train['momentum'],
                                weight_decay=hp_train['weight_decay'])

    # learning rate scheduler decreases the learning rate by gamma every
    # step_size number of epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=hp_train['step_size'],
                                                   gamma=0.1)

    # record loss value in tensorboard
    now = str(datetime.datetime.now())
    expname = now[0:10] + '_' + now[11:13] + '-' + now[14:16] + '-Run'
    writer = SummaryWriter(os.path.join('runs', expname))

    # set validation epoch frequency
    val_epoch = 2

    # set savepoint epoch frequency
    snapshot_epoch = 5

    # ------------------------------ #
    # training
    start_time = time.time()
    print('begin training')
    for epoch in range(hp_train['num_epochs']):
        # TODO use weights and biases to visualise training/results, rather than tensorboard

        # modified from coco_api tools to take in separate training and
        # validation dataloaders, as well as port the images to device
        mt, mv = train_one_epoch(model,
                                 optimizer,
                                 dl_train,
                                 dl_test,
                                 device,
                                 epoch,
                                 val_epoch,
                                 print_freq=10)

        writer.add_scalar('Detector/Training_Loss', mt.loss.median, epoch + 1)
        # other loss types from metric logger
        # mt.loss.value
        # mt.loss_classifier.median
        # mt.loss_classifier.max
        # mt.loss_box_reg.value
        # mt.loss_objectness.value
        # mt.loss_rpn_box_reg.median

        # update the learning rate
        lr_scheduler.step()

        # evaluate on test dataset ever val_epoch epochs
        if (epoch % val_epoch) == (val_epoch - 1):
            writer.add_scalar('Detector/Validation_Loss', mv.loss.median, epoch + 1)

        # save snapshot every snapshot_epochs
        if (epoch % snapshot_epoch) == (snapshot_epoch - 1):
            print('saving snapshot at epoch: {}'.format(epoch))

            # save epoch
            if not os.path.isdir(os.path.join(save_folder, 'snapshots')):
                os.mkdir(os.path.join(save_folder, 'snapshots'))
            snapshot_name = os.path.join(save_folder,
                                         'snapshots',
                                         model_save_name + '_epoch' + str(epoch + 1) + '.pth')
            torch.save(model.state_dict(), snapshot_name)
            print('snapshot name: {}',format(snapshot_name))


    # for non-max-suppression, need:
    # conf = 0.7
    # iou = 0.5
    # # TODO fix evaluate to deal with null case while evaluating model with nms
    # # TODO consider making a function for calling the model? the "forward" pass?
    # mt_eval, ccres = evaluate(model,
    #                           dataloader_val,
    #                           device=device,
    #                           conf=conf,
    #                           iou=iou,
    #                           class_names=class_names)

    print('training done')
    end_time = time.time()
    sec = end_time - start_time
    print('training time: {} sec'.format(sec))
    print('training time: {} min'.format(sec / 60.0))
    print('training time: {} hrs'.format(sec / 3600.0))

    # save trained model for inference
    torch.save(model.state_dict(), save_path)
    print('model saved: {}'.format(save_path))

    # see inference.py for running model on images/datasets
    print('done training: {}'.format(model_save_name))

    import code
    code.interact(local=dict(globals(), **locals()))