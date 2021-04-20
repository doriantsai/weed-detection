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
import time

import cv2 as cv
import matplotlib.pyplot as plt

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
    batch_size = 10
    num_workers = 10
    learning_rate = 0.005 # was 0.005 for v9
    momentum = 0.9
    weight_decay = 0.0001
    num_epochs = 100
    step_size = round(num_epochs/2)

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
    save_name = 'Tussock_v0_12'
    # save_name = 'fasterrcnn-serratedtussock-4'
    save_folder = os.path.join('output', save_name)
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    save_path = os.path.join(save_folder, save_name + '.pth')

    # setup device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # setup dataset
    # root_dir = os.path.join('/home','dorian','Data','SerratedTussockDataset_v2')
    # json_file = os.path.join('Annotations','via_region_data.json')
    root_dir = os.path.join('/home', 'dorian', 'Data', 'AOS_TussockDataset', 'Tussock_v0')
    json_file = os.path.join('Annotations', 'annotations_tussock_21032526_G507_combined.json')


    # setup save pickle file (saved datasets/loaders, etc for inference)
    save_detector_train_path = os.path.join('.', 'output', save_name, save_name + '.pkl')

    # setup transforms to operate on dataset images
    rescale_size = 2056
    tforms_train = Compose([Rescale(rescale_size),
                            RandomHorizontalFlip(0.5),
                            Blur(5, (0.5, 2.0)),
                            ToTensor()])
    tforms_test = Compose([Rescale(rescale_size),
                           ToTensor()])

    # TODO apply training transforms and testing transforms separately
    # otherwise testing set changes
    dataset_tform_train = SerratedTussockDataset(root_dir=root_dir,
                                           json_file=json_file,
                                           transforms=tforms_train)
    dataset_tform_test = SerratedTussockDataset(root_dir=root_dir,
                                          json_file=json_file,
                                          transforms=tforms_test)

    # class definitions
    class_names = ["_background_", "serrated tussock"]

    # split into training, validation and testing - note: we're only after the indices here
    nimg = len(dataset_tform_test)
    ntrain_val = 501 # select number of images for training dataset
    # 570 images total, so 50 images for testing
    dataset_train_and_val, dataset_test = torch.utils.data.random_split(dataset_tform_test,
                                                [ntrain_val, nimg - ntrain_val])

    # further split the training/val dataset
    ntrain = 500
    dataset_train, dataset_val = torch.utils.data.random_split(dataset_train_and_val,
                                                    [ntrain, ntrain_val - ntrain])

    # adjust the transforms for the training set
    # dataset_train.dataset.dataset.set_transform(tforms_train)
    dataset_train.dataset.dataset = dataset_tform_train

    # setup dataloaders for efficient access to datasets
    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=num_workers,
                                                   collate_fn=utils.collate_fn)

    dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=num_workers,
                                                  collate_fn=utils.collate_fn)

    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers,
                                                  collate_fn=utils.collate_fn)

    # should also save the training and testing datasets/loaders for easier "test.py" setup
    with open(save_detector_train_path, 'wb') as f:
        pickle.dump(dataset_tform_test, f)
        pickle.dump(dataset_tform_train, f)
        pickle.dump(dataset_train, f)
        pickle.dump(dataset_val, f)
        pickle.dump(dataset_test, f)
        pickle.dump(dataloader_test, f)
        pickle.dump(dataloader_train, f)
        pickle.dump(dataloader_val, f)
        pickle.dump(hp, f)

    # TODO test if dataloader_train has randomly horizontal flipped images
    # imgs_t, smps_t = next(iter(dataloader_train))
    # print('dataloader_train: image should be flipped')
    # print(smps_t[1]['image_id'])
    # imgnp = imgs_t[1].cpu().numpy()
    # imgnp = np.transpose(imgnp, (1, 2, 0))
    # plt.imshow(imgnp)
    # plt.show()

    # imgs, smps = next(iter(dataloader_test))
    # print('dataloader_test: image should NOT be flipped')
    # print(smps[1]['image_id'])
    # imgnp = imgs[1].cpu().numpy()
    # imgnp = np.transpose(imgnp, (1, 2, 0))
    # plt.imshow(imgnp)
    # plt.show()

    # import code
    # code.interact(local=dict(globals(), **locals()))


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

    # set savepoint epoch frequency
    snapshot_epoch = 5

    # ------------------------------ #
    # training
    start_time = time.time()
    print('begin training')
    for epoch in range(num_epochs):


        # modified from coco_api tools to take in separate training and
        # validation dataloaders, as well as port the images to device
        mt, mv = train_one_epoch(model,
                                 optimizer,
                                 dataloader_train,
                                 dataloader_test,
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
        # TODO only evaluate once in a while - every other
        # print('evaluating on test set')


        if (epoch % val_epoch) == (val_epoch - 1):
            writer.add_scalar('Detector/Validation_Loss', mv.loss.median, epoch + 1)

        if (epoch % snapshot_epoch) == (snapshot_epoch - 1):
            print('saving snapshot at epoch: {}'.format(epoch))

            # save epoch
            # save_path = os.path.join(save_folder, save_name + '.pth')
            if not os.path.isdir(os.path.join(save_folder, 'snapshots')):
                os.mkdir(os.path.join(save_folder, 'snapshots'))
            snapshot_name = os.path.join(save_folder,
                                            'snapshots',
                                            save_name + '_epoch' + str(epoch) + '.pth')
            torch.save(model.state_dict(), snapshot_name)
            print('snapshot name: {}',format(snapshot_name))


    # for non-max-suppression, need:
    # conf = 0.7
    # iou = 0.5
    # # TODO fix evaluate to deal with null case while evaluating model with nms
    # # TODO consider making a function for calling the model? the "forward" pass?
    # # TODO discuss this with david
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
    # x = {'state_dict': model.state_dict(),
    # 'hp': hp
    # }
    # torch.save(x,save_path)
    print('model saved: {}'.format(save_path))
    # suggested code from Sam to load the state dictionary
    # model.load_state_dict(file['state_dict'])
    # print(file['hp'])




    # see inference.py for running model on images/datasets
    print('done training: {}'.format(save_name))

    import code
    code.interact(local=dict(globals(), **locals()))