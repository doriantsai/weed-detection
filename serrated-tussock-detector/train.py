#! /usr/bin/env python

"""
Purpose: Train serrated tussock detector
@author: Dorian Tsai
@email: dorian.tsai@gmail.com
"""

import os
import numpy as np
import torch
import torch.utils.data
import torchvision
import utils
import json
import pickle
import datetime

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from engine_st import train_one_epoch, evaluate
from SerratedTussockDataset import SerratedTussockDataset, RandomHorizontalFlip, Rescale, ToTensor, Blur, Compose
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

    # ------------------------------ #
    # hyperparameters
    batch_size = 5
    num_workers = 0
    learning_rate = 0.005
    momentum = 0.9
    weight_decay = 0.0001
    num_epochs = 100
    step_size = round(num_epochs/4)

    # make a hyperparameter dictionary
    hp={}
    hp['batch_size'] = batch_size
    hp['num_workers'] = num_workers
    hp['learning_rate'] = learning_rate
    hp['momentum'] = momentum
    hp['weight_decay'] = weight_decay
    hp['num_epochs'] = num_epochs

    # ------------------------------ #
    # directories
    # TODO add date/time to filename
    save_path = os.path.join('output', 'fasterrcnn-serratedtussock-bootstrap-3.pth')

    # setup device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # setup dataset
    root_dir = os.path.join('/home', 'dorian', 'Data', 'SerratedTussockDataset_v1')
    json_file = os.path.join('Annotations', 'via_region_data.json')

    # setup save pickle file (saved datasets/loaders, etc for inference)
    save_detector_train_path = os.path.join('.', 'output', 'st_data_bootstrap-3.pkl')

    # setup transforms to operate on dataset images
    tforms = Compose([Rescale(800),
                      RandomHorizontalFlip(0.5),
                      Blur(5, (0.5, 2.0)),
                      ToTensor()])

    dataset = SerratedTussockDataset(root_dir=root_dir,
                                     json_file=json_file,
                                     transforms=tforms)

    # split into training, validation and testing
    nimg = len(dataset)
    ntrain_val = 90  # select number of images for training dataset
    dataset_train_and_val, dataset_test = torch.utils.data.random_split(dataset,
                                                [ntrain_val, nimg - ntrain_val])

    # further split the training/val dataset
    ntrain = 80
    dataset_train, dataset_val = torch.utils.data.random_split(dataset_train_and_val,
                                                    [ntrain, ntrain_val - ntrain])

    # setup dataloaders for efficient access to datasets
    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=num_workers,
                                                   collate_fn=utils.collate_fn)

    dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers,
                                                  collate_fn=utils.collate_fn)

    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers,
                                                  collate_fn=utils.collate_fn)

    # build model
    # setup number of classes (1 background, 1 class - serrated tussock)
    model = build_model(num_classes=2)
    model.to(device)

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=learning_rate,
                                momentum=momentum,
                                weight_decay=weight_decay)

    # learning rate scheduler decreases the learning rate by gamma every
    # step_size number of epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=step_size,
                                                   gamma=0.1)

    # record loss value in tensorboard
    now = str(datetime.datetime.now())
    expname = now[0:10] + '_' + now[11:13] + '-' + now[14:16] + '-Run'
    writer = SummaryWriter(os.path.join('runs', expname))

    # set validation epoch frequency
    val_epoch = 2


    # ------------------------------ #
    # training
    print('begin training')
    for epoch in range(num_epochs):


        # modified from coco_api tools to take in separate training and
        # validation dataloaders, as well as port the images to device
        mt, mv = train_one_epoch(model,
                                 optimizer,
                                 dataloader_train,
                                 dataloader_val,
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

        # evaluate on test dataset
        print('evaluating on test set')
        mt_eval = evaluate(model, dataloader_test, device=device)

        if (epoch % val_epoch) == (val_epoch - 1):
            writer.add_scalar('Detector/Validation_Loss', mv.loss.median, epoch + 1)

    # save trained model for inference
    torch.save(model.state_dict(), save_path)

    # should also save the training and testing datasets/loaders for easier "test.py" setup
    with open(save_detector_train_path, 'wb') as f:
        pickle.dump(dataset, f)
        pickle.dump(dataset_train, f)
        pickle.dump(dataset_val, f)
        pickle.dump(dataset_test, f)
        pickle.dump(dataloader_test, f)
        pickle.dump(dataloader_train, f)
        pickle.dump(dataloader_val, f)
        pickle.dump(hp, f)

    # see inference.py for running model on images/datasets
    print('done training')

    import code
    code.interact(local=dict(globals(), **locals()))