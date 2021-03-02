#! /usr/bin/env python

from SerratedTussockDataset import SerratedTussockDataset, RandomHorizontalFlip, Rescale, ToTensor, Blur, Compose
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import json
import pickle

import datetime
from torch.utils.tensorboard import SummaryWriter

from engine import train_one_epoch, evaluate
import utils

import matplotlib.pyplot as plt
# import matplotlib.patches as ptch
import matplotlib.patches as mpp

torch.manual_seed(42)


# ------------------------------------------------------------------- #

def build_model(num_classes):
    # load instance segmentation model pre-trained on coco:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def show_image_bbox(image, sample, color='blue'):
    # show image and bounding box together
    # matplotlib_imshow(image)
    imgnp = image.cpu().numpy()

    fig, ax = plt.subplots(1)

    ax.imshow(np.transpose(imgnp, (1, 2, 0)))

    boxes = sample['boxes'] 
    nbb, _ = boxes.size()

    print(imgnp.shape)

    for i in range(nbb):
        print('plot box {}'.format(i))
        bb = np.array(boxes[i, :].cpu(), dtype=np.float32)
        print(bb)  # [xmin, ymin, xmax, ymax]
        rect = mpp.Rectangle((bb[0], bb[1]), 
                             bb[2] - bb[0],
                             bb[3] - bb[1], 
                             color=color,
                             fill=False,
                             linewidth=3)
        ax.add_patch(rect)

        # plt.gca().add_patch(show_single_bbox(image, boxes[i, :]))

    return fig, ax

# --------------------------------------------------------------------------- #
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


    # ------------------------------ #
    # directories
    # TODO add date/time to filename
    save_path = os.path.join('output', 'fasterrcnn-serratedtussock-bootstrap-0.pth')


    # setup device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # setup dataset
    root_dir = os.path.join('/home', 'dorian', 'Data', 'SerratedTussockDataset_v1') 
    json_file = os.path.join('Annotations', 'via_region_data.json')
    
    # tforms = Compose([Rescale(800), ToTensor()])

    tforms = Compose([Rescale(800),
                      RandomHorizontalFlip(0.5),
                      Blur(5, (0.5, 2.0)), # quite a lot of blurring, might be expected due to motion?
                      ToTensor()])

    dataset = SerratedTussockDataset(root_dir=root_dir,
                                     json_file=json_file,
                                     transforms=tforms)

    # split into training, validation and testing
    nimg = len(dataset)
    ntrain = 80
    # TODO obviously, we'll need more data/images in future runs, 
    # but we're just gunning for the pipeline at the moment
    dataset_train, dataset_test = torch.utils.data.random_split(dataset, [ntrain, nimg - ntrain])

    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers,
                                                  collate_fn=utils.collate_fn)
                                                  # tutorial has collate_fn?

    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=num_workers,
                                                   collate_fn=utils.collate_fn)

    # not sure if I have a background class at the moment
    nclasses = 2

    model = build_model(nclasses)
    model.to(device)


    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]  # TODO what is this?
    optimizer = torch.optim.SGD(params, 
                                lr=learning_rate,
                                momentum=momentum,
                                weight_decay=weight_decay)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=step_size,
                                                   gamma=0.1)


    # tensorboard writer
    # record loss value in tensorboard
    now = str(datetime.datetime.now())        
    expname = now[0:10] + '_' + now[11:13] + '-' + now[14:16] + '-Run'
    writer = SummaryWriter(os.path.join('runs', expname))

    # training
    val_epoch = 2
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        # train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
        # import code
        # code.interact(local=dict(globals(), **locals()))

        print('training one epoch on training set')
        mt, mv = train_one_epoch(model, optimizer, dataloader_train, dataloader_test, device, epoch, val_epoch, print_freq=10)
        
        writer.add_scalar('Detector/Training_Loss', mt.loss.median, epoch + 1)


        # mt.loss.value
        # mt.loss_classifier.median
        # mt.loss_classifier.max
        # mt.loss_box_reg.value
        # mt.loss_objectness.value
        # mt.loss_rpn_box_reg.median
        
        
        # writer.add_scalar('Epoch/Loss', loss_value, )

        # update the learning rate
        lr_scheduler.step()

        # evaluate on test dataset
        print('evaluating on test set')
        mt_eval = evaluate(model, dataloader_test, device=device)

        if (epoch % val_epoch) == (val_epoch - 1):
            writer.add_scalar('Detector/Validation_Loss', mv.loss.median, epoch + 1)

        
        # import code
        # code.interact(local=dict(globals(), **locals())) 


    # save trained model for inference
    torch.save(model.state_dict(), save_path)

    # should also save the training and testing datasets/loaders for easier "test.py" setup
    save_detector_train_path = os.path.join('.', 'output', 'st_data_bootstrap.pkl')
    with open(save_detector_train_path, 'wb') as f:
        pickle.dump(dataset, f)
        pickle.dump(dataset_train, f)
        pickle.dump(dataset_test, f)
        pickle.dump(dataloader_test, f)
        pickle.dump(dataloader_train, f)
        pickle.dump(hp, f)


    # import code
    # code.interact(local=dict(globals(), **locals())) 

    # dataiter = next(iter(dataloader_test))
    # img, smp = dataiter[0], dataiter[1]
    # matplotlib_imshow(img)

        
    # get datasample from one instance of the dataloader_test

    SHOW_TRAIN = False
    if SHOW_TRAIN:
        datasample = next(iter(dataloader_train))
        img_batch, smp_batch = datasample[0], datasample[1]
        
        bs = len(img_batch)
        print(bs)
        for i in range(bs):
            print(i)
            figi, axi = show_image_bbox(img_batch[i], smp_batch[i])
            
            plt.savefig(os.path.join('output', 'fasterrcnn-serratedtussock-trainbbox-' + str(i) + '.png'))
            plt.show()

    import code
    code.interact(local=dict(globals(), **locals()))