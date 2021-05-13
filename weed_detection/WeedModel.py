#! /usr/bin/env python

"""
weed model class for weed detection
class to package model-related functionality, such as
training, inference, evaluation
"""

import os
import torch
import torchvision
import json
from weed_detection.WeedDataset import WeedDataset
import time
import datetime
import pickle

# TODO replace tensorboard with weightsandbiases
from torch.utils.tensorboard import SummaryWriter
from engine_st import train_one_epoch, evaluate
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from weed_detection.PreProcessingToolbox import PreProcessingToolbox


class WeedModel:
    """ collection of functions for model's weed detection """

    def __init__(self, weed_name='serrated tussock', model=None):

        self.weed_name = weed_name
        # TODO maybe save model type/architecture
        # also, hyper parameters?
        self.model = model

    
    def build_model(self, num_classes):
        """ build fasterrcnn model for set number of classes """

        # load instance of model pre-trained on coco:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace pre-trained head with new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model


    def create_dataset_dataloader(root_dir,
                                json_file,
                                transforms,
                                hp):
        # assume tforms already defined outside of this function
        batch_size = hp['batch_size']
        num_workers = hp['num_workers']
        shuffle= hp['shuffle']

        dataset = WeedDataset.WeedDataset(root_dir, json_file, transforms)
        # setup dataloaders for efficient access to datasets
        dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers,
                                                collate_fn=dataset.collate_fn)
        return dataset, dataloader


    def create_train_test_val_datasets(self, img_folders, ann_files, hp, dataset_name):
        """ creates datasets and dataloader objects from train/test/val files """
        # arguably, should be in WeedDataset class

        # unpack
        train_folder = img_folders[0]
        test_folder = img_folders[1]
        val_folder = img_folders[2]

        # should be full path list to json files (ann_dir + ann_xx_file)
        ann_train = ann_files[0]
        ann_test = ann_files[1]
        ann_val = ann_files[2]

        # TODO: check hp is valid, should instead have transform parameters dict
        rescale_size = hp['rescale_size']

        hp_train = hp
        hp_test = hp
        hp_test['shuffle'] = False

        tform_train = WeedDataset.Compose([WeedDataset.Rescale(rescale_size),
                          WeedDataset.RandomBlur(5, (0.5, 2.0)),
                          WeedDataset.RandomHorizontalFlip(0.5),
                          WeedDataset.RandomVerticalFlip(0.5),
                          WeedDataset.ToTensor()])
        tform_test = WeedDataset.Compose([WeedDataset.Rescale(rescale_size),
                         WeedDataset.ToTensor()])

        # create dataset and dataloader objects for each set of images
        ds_train, dl_train = self.create_dataset_dataloader(train_folder,
                                                            ann_train,
                                                            tform_train,
                                                            hp_train)

        ds_test, dl_test = self.create_dataset_dataloader(test_folder,
                                                          ann_test,
                                                          tform_test,
                                                          hp_test)
        
        ds_val, dl_val = self.create_dataset_dataloader(val_folder,
                                                        ann_val,
                                                        tform_test,
                                                        hp_test)

        # save datasets/dataloaders for later use
        # TODO dataset_name default?
        save_dataset_folder = os.path.join('dataset', dataset_name)
        os.makedirs(save_dataset_folder, exist_ok=True)
        save_dataset_path = os.path.join(save_dataset_folder, dataset_name + '.pkl')
        with open(save_dataset_path, 'wb') as f:
            pickle.dump(ds_train, f)
            pickle.dump(ds_test, f)
            pickle.dump(ds_val, f)
            pickle.dump(dl_train, f)
            pickle.dump(dl_test, f)
            pickle.dump(dl_val, f)
            pickle.dump(hp_train, f)
            pickle.dump(hp_test, f)

        print('dataset_name: {}'.format(dataset_name))
        print('dataset saved as: {}'.format(save_dataset_path))

        return save_dataset_path


    def train(self, dataset_path):

        # TODO if dataset_path is None, call create_train_test_val_datasets
        # for now, we assume this has been done/dataset_path exists and is valid

        # TODO load dataset from path
        # unpack pickle file
        # choose model name (should be input)
        # set device (maybe global?)
        # build model
        # optimizer
        # learning rate scheduler
        # setup time-specific names/folders
        # set parameters
        # train for-loop for epochs
        # save model
    
    # TODO get_prediction
    # TODO inference_single
    # TODO inference_dataset
    # TODO inference_video
    # TODO prcurve
    # TODO model_compare
    #


