#! /usr/bin/env python

"""
Title: WeedModel.py
Purpose: Weed model class for weed detection class to package model-related
functionality, such as training, inference, evaluation
Author: Dorian Tsai
Date: 28 October 2021
Email: dorian.tsai@gmail.com
"""

import os
from typing import Type
import torch
import torchvision
import re
import time
import datetime
import pickle
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# TODO replace tensorboard with weightsandbiases
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as tv_transform
from scipy.interpolate import interp1d
from shapely.geometry import Polygon

# custom imports
from weed_detection.engine_st import train_one_epoch
import weed_detection.WeedDataset as WD
import weed_detection.WeedDatasetPoly as WDP
from weed_detection.PreProcessingToolbox import PreProcessingToolbox as PT
# from webcam import grab_webcam_image


class WeedModel:
    """ a collection of methods for weed detection model training, detection,
    inference and evaluation """
    # TODO consider splitting into model training/inference/evaluation objects

    def __init__(self,
                 weed_name='serrated tussock',
                 model=None,
                 model_name=None,
                 model_folder=None,
                 model_path=None,
                 device=None,
                 hyper_parameters=None,
                 epoch=None,
                 annotation_type='poly',
                 detector_type='maskrcnn'):
        """ initialise single-class detector object """

        # weed_name is a string of the name of the weed that the detector is
        if isinstance(weed_name, str):
            self._weed_name = weed_name
        else:
            raise TypeError(weed_name, "weed_name must be type str")

        # annotation type for bounding boxes or polygons
        if self.check_annotation_type(annotation_type):
            self._annotation_type = annotation_type
        else:
            raise TypeError(annotation_type, "annotation_type must be either 'box' or 'poly'")

        # model itself, the Pytorch model object that is used for training and
        # image inference
        if (isinstance(model, torchvision.models.detection.mask_rcnn.MaskRCNN) or
            isinstance(model, torchvision.models.detection.faster_rcnn.FasterRCNN) or
            model is None):
            self._model = model
        else:
            raise TypeError(model, "model must be of type MaskRCNN or FasterRCNN")

        # model path is the absolute file path to the .pth detector model
        if (isinstance(model_path, str) or (model_path is None)):
            self._model_path = model_path
        else:
            raise TypeError(model_path, "model_path must be of type str or None")
        if model_path is not None:
            self.load_model(model_path, annotation_type=annotation_type)

        # name of the model, an arbitrary string of text
        if (isinstance(model_name, str) or (model_name is None)):
            self._model_name = model_name
        else:
            raise TypeError(model_name, "model_name must be of type str or None")
        if model_name is None and model_path is not None:
            self._model_name = os.path.basename(model_path)

        # model_folder is the folder of the model, which should be
        # taken from os.path.dirname(model_path)
        if (isinstance(model_folder, str) or (model_folder is None)):
            if model_folder is None:
                self._model_folder = model_name
            else:
                self._model_folder = model_folder
        else:
            raise TypeError(model_folder, "model_folder must be of type str or None")

        # device, whether computation happens on cpu or gpu
        if device is None:
            device = torch.device(
                'cuda') if torch.cuda.is_available() else torch.device('cpu')
            # device = torch.device('cpu')
        else:
            # TODO check device type
            print('TODO: check device type')
        self._device = device

        self._hp = hyper_parameters
        # TODO use parse arges library to convey hyper parameters
        # TODO should get image height/width automatically from images
        self._image_height = int(2056 / 2)  # rescale_size
        self._image_width = int(2464 / 2)

        if epoch is None:
            epoch = 0
        if isinstance(epoch, int) or isinstance(epoch, float):
            self._epoch = int(epoch)
        else:
            raise TypeError(epoch, "epoch must be type int or float")


    @property
    def model(self):
        return self._model

    # getters and setters
    def set_model(self, model):
        self._model = model

    def get_model(self):
        return self._model

    def get_image_tensor_size(self):
        return (3, self._image_height, self._image_width)

    def set_model_folder(self, folder):
        self._model_folder = folder

    def get_model_folder(self):
        return self._model_folder

    def set_model_name(self, name):
        self._model_name = name

    def get_model_name(self):
        return self._model_name

    def set_weed_name(self, name):
        self._weed_name = name

    def get_weed_name(self):
        return self._weed_name

    def set_model_path(self, path):
        self._model_path = path

    def get_model_path(self):
        return self._model_path

    def set_model_epoch(self, epoch):
        self._epoch = epoch

    def get_model_epoch(self):
        return self._epoch

    def set_hyper_parameters(self, hp):
        self._hp = hp

    def get_hyper_parameters(self):
        return self._hp


    def check_annotation_type(self, annotation_type):
        """ checks for valid annotation type """
        if not isinstance(annotation_type, str):
            raise TypeError(annotation_type, 'annotation_type must be a str')
        valid_ann_types = ['box', 'poly']
        if not(annotation_type in valid_ann_types):
            raise TypeError(annotation_type, f'annotation_type must be of valid annotation types: {valid_ann_types}')
        return True


    def build_fasterrcnn_model(self, num_classes):
        """ build fasterrcnn model for set number of classes (int), loads pre-trained
        on coco image database, sets annotation_type to 'box' """

        if not isinstance(num_classes, int):
            raise TypeError(num_classes, 'num_classes must be an int')

        # load instance of model pre-trained on coco:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # replace pre-trained head with new one
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes)
        self._annotation_type = 'box'
        return model


    def build_maskrcnn_model(self, num_classes):
        """ build maskrcnn model for set number of classes (int), loads pre-trained
        model on coco image database, sets annotation_type to 'poly' """

        if not isinstance(num_classes, int):
            raise TypeError(num_classes, 'num_classes must be an int')

        # load instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=True)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # replace pretrained head with new one
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes)

        # now get number of input features for mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256  # TODO check what this variable is

        # replace mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes)
        self._annotation_type = 'poly'
        return model


    def load_dataset_objects(self, dataset_path, return_dict=True):
        """ load/unpackage dataset objects given path. dataset_path is a string
        that is an absolute path to the dataset file object. The dataset file
        object is a .pkl file from the self.create_train_test_val_datasets
        method"""

        if not isinstance(dataset_path, str):
            TypeError(dataset_path, 'dataset_path must be a string')
        if not isinstance(return_dict, bool):
            TypeError(return_dict, 'return_dict must be a boolean')

        # order is important
        if os.path.isfile(dataset_path):
            with open(dataset_path, 'rb') as f:
                ds_train = pickle.load(f)
                ds_test = pickle.load(f)
                ds_val = pickle.load(f)
                dl_train = pickle.load(f)
                dl_test = pickle.load(f)
                dl_val = pickle.load(f)
                hp_train = pickle.load(f)
                hp_test = pickle.load(f)
                dataset_name = pickle.load(f)

        # save as a dictionary
        if return_dict:
            dso = {'ds_train': ds_train,
                   'ds_test': ds_test,
                   'ds_val': ds_val,
                   'dl_train': dl_train,
                   'dl_test': dl_test,
                   'dl_val': dl_val,
                   'hp_train': hp_train,
                   'hp_test': hp_test,
                   'dataset_name': dataset_name}
            return dso
        else:
            return ds_train, ds_test, ds_val, dl_train, dl_test, dl_val, \
                hp_train, hp_test, dataset_name


    def create_dataset_dataloader(self,
                                  root_dir,
                                  json_file,
                                  transforms,
                                  hp,
                                  annotation_type='poly',
                                  mask_dir=None,
                                  img_dir=None):
        """ creates a pytorch dataset and dataloader object for given set of
        data (root_dir, json_file, transforms, hyperparameters (hp), mask_dir,
        img_dir). Typically iteratively by self.create_train_test_val_datasets()
        to do the same operation for training/testing/validation data """

        # check input variables
        if not isinstance(root_dir, str):
            raise TypeError(root_dir, 'root_dir must be a str')
        if not isinstance(json_file, str):
            raise TypeError(json_file, 'json_file must be a str')
        # TODO check for transforms, list of transform objects?
        if not isinstance(hp, dict):
            raise TypeError(hp, 'hp must be a dict')
        self.check_annotation_type(annotation_type)

        # assume transforms  already defined outside of this function
        batch_size = hp['batch_size']
        num_workers = hp['num_workers']
        shuffle = hp['shuffle']

        # default image directory specified as root directory, root_dir
        if img_dir is None:
            img_dir = root_dir

        # different dataset objects for polygons vs boxes
        if annotation_type == 'poly':
            dataset = WDP.WeedDatasetPoly(root_dir,
                                          json_file,
                                          transforms,
                                          img_dir=img_dir,
                                          mask_dir=mask_dir)
        else:
            dataset = WD.WeedDataset(root_dir,
                                     json_file,
                                     transforms,
                                     img_dir=img_dir)

        # setup dataloaders for efficient access to datasets
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=num_workers,
                                                 collate_fn=self.collate_fn)
        return dataset, dataloader


    def collate_fn(self, batch):
        """ collates batch of images together into a tuple for pytorch
        dataloaders """
        return tuple(zip(*batch))


    def create_train_test_val_datasets(self,
                                       img_folders,
                                       ann_files,
                                       hp,
                                       dataset_name,
                                       annotation_type='poly',
                                       mask_folders=None):
        """ creates datasets and dataloader objects from train/test/val folders
        and corresponding annotation files, saves these objects in
        'save_dataset_path' (defined below, TODO should be an optional input),
        outputs the string to the save_dataset_path

        img_folders - a list of strings for each train/test/val folder
        ann_files
        - a list of strings for each annotation file, corresponding to
        train/test/val
        hp - hyperparameters dictionary
        dataset_name - string,
        name of the dataset
        annotation_type - string, type of annotation
        (poly/box)
        mask_folders - for poly annotations, a list of strings for
        each train/test/val folder of mask images
        """
        # NOTE consider moving this to WeedDataset class

        # check valid types
        # TODO

        # unpack
        train_folder = img_folders[0]
        test_folder = img_folders[1]
        val_folder = img_folders[2]

        if mask_folders is not None:
            mask_train_folder = mask_folders[0]
            mask_test_folder = mask_folders[1]
            mask_val_folder = mask_folders[2]
        else:
            mask_train_folder = None
            mask_test_folder = None
            mask_val_folder = None

        # should be full path list to json files (ann_dir + ann_xx_file)
        ann_train = ann_files[0]
        ann_test = ann_files[1]
        ann_val = ann_files[2]

        # TODO: check hp is valid, shou-ld instead have transform parameters dict
        rescale_size = hp['rescale_size']

        hp_train = hp
        hp_test = hp
        hp_test['shuffle'] = False

        # TODO add in other data augmentation methods TODO make dictionary for
        # data augmentation parameters with default values TODO but also as
        # inputs
        if annotation_type == 'poly':
            print('creating poly transforms')
            tform_train = WDP.Compose([WDP.Rescale(rescale_size),
                                       WDP.RandomBlur(5, (0.5, 2.0)),
                                       WDP.RandomHorizontalFlip(0),
                                       WDP.RandomVerticalFlip(0),
                                       WDP.ToTensor()])
            tform_test = WDP.Compose([WDP.Rescale(rescale_size),
                                      WDP.ToTensor()])
        else:
            # import code
            # code.interact(local=dict(globals(), **locals()))
            tform_train = WD.Compose([WD.Rescale(rescale_size),
                                      WD.RandomBlur(5, (0.5, 2.0)),
                                      WD.RandomHorizontalFlip(0.5),
                                      WD.RandomVerticalFlip(0.5),
                                      WD.ToTensor()])
            tform_test = WD.Compose([WD.Rescale(rescale_size),
                                     WD.ToTensor()])

        # create dataset and dataloader objects for each set of images
        ds_train, dl_train = self.create_dataset_dataloader(train_folder,
                                                            ann_train,
                                                            tform_train,
                                                            hp_train,
                                                            annotation_type,
                                                            mask_dir=mask_train_folder)

        ds_test, dl_test = self.create_dataset_dataloader(test_folder,
                                                          ann_test,
                                                          tform_test,
                                                          hp_test,
                                                          annotation_type,
                                                          mask_dir=mask_test_folder)

        ds_val, dl_val = self.create_dataset_dataloader(val_folder,
                                                        ann_val,
                                                        tform_test,
                                                        hp_test,
                                                        annotation_type,
                                                        mask_dir=mask_val_folder)

        # save datasets/dataloaders for later use TODO dataset_name default?
        save_dataset_folder = os.path.join('dataset_objects', dataset_name)
        os.makedirs(save_dataset_folder, exist_ok=True)
        save_dataset_path = os.path.join(
            save_dataset_folder, dataset_name + '.pkl')
        with open(save_dataset_path, 'wb') as f:
            pickle.dump(ds_train, f)
            pickle.dump(ds_test, f)
            pickle.dump(ds_val, f)
            pickle.dump(dl_train, f)
            pickle.dump(dl_test, f)
            pickle.dump(dl_val, f)
            pickle.dump(hp_train, f)
            pickle.dump(hp_test, f)
            pickle.dump(dataset_name, f)

        print('dataset_name: {}'.format(dataset_name))
        print('dataset saved as: {}'.format(save_dataset_path))

        return save_dataset_path

    def get_now_str(self):
        """ get a string of yyyymmdd_hh_mm or something similar """
        # useful for creating unique folder/variable names
        now = str(datetime.datetime.now())
        now_str = now[0:10] + '_' + now[11:13] + '_' + now[14:16]
        return now_str

    def train(self,
              model_name,
              dataset_path=None,
              model_name_suffix=True,
              model_folder=None,
              annotation_type='poly'):

        # TODO if dataset_path is None, call create_train_test_val_datasets for
        # now, we assume this has been done/dataset_path exists and is valid
        if dataset_path is None:
            print('TODO: call function to build dataset objects and return them')
        # else:

        # loading dataset, full path
        print('Loading dataset:' + dataset_path)
        if os.path.isfile(dataset_path):
            with open(dataset_path, 'rb') as f:
                ds_train = pickle.load(f)
                ds_test = pickle.load(f)
                ds_val = pickle.load(f)
                dl_train = pickle.load(f)
                dl_test = pickle.load(f)
                dl_val = pickle.load(f)
                hp_train = pickle.load(f)
                hp_test = pickle.load(f)
                dataset_name = pickle.load(f)
        else:
            print('File does not exist: {}'.format(dataset_path))

        print('Loaded dataset name: {}'.format(dataset_name))

        # get time/date and convert to string,
        now_str = self.get_now_str()

        # eg, we append now_str to the end of model_name
        if model_name_suffix:
            model_name = model_name + '_' + now_str

        print('Training model, model name: {}'.format(model_name))

        # create model's save folder
        if model_folder is None:
            save_folder = os.path.join('output', model_name)
        else:
            save_folder = os.path.join('output', model_folder)
        os.makedirs(save_folder, exist_ok=True)
        print('Model saved in folder: {}'.format(save_folder))

        # setup device, send to gpu if possible, otherwise cpu device =
        # torch.device('cuda') if torch.cuda.is_available() else
        # torch.device('cpu') shifted into object properties and init

        # build model setup number of classes (1 background, 1 class - weed
        # species)
        if annotation_type == 'poly':
            print('building maskrcnn model')
            model = self.build_maskrcnn_model(num_classes=2)
        else:
            print('building fasterrcnn model')
            model = self.build_fasterrcnn_model(num_classes=2)
        model.to(self._device)

        # set optimizer
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

        # create tensorboard writer
        exp_name = now_str + '_' + model_name
        writer = SummaryWriter(os.path.join('runs', exp_name))

        # set validation epoch frequency
        val_epoch = 2

        # set savepoint epoch frequency
        snapshot_epoch = 5

        # ---------------------------------------------- #
        # train for-loop for epochs NOTE we do not do explicit early stopping.
        # We run for a set number of epochs and then choose the appropriate
        # "stopping" point later from the snapshots. This is to clearly identify
        # that in fact, we have reached a low-point in the validation loss.
        start_time = time.time()
        print('start training')
        for epoch in range(hp_train['num_epochs']):
            # modified from coco_api tools to take in separate training and
            # validation dataloaders, as well as port the images to device
            mt, mv = train_one_epoch(model,
                                     optimizer,
                                     dl_train,
                                     dl_val,
                                     self._device,
                                     epoch,
                                     val_epoch,
                                     print_freq=10)

            writer.add_scalar('Detector/Training_Loss',
                              mt.loss.median, epoch + 1)
            # other loss types from metric logger mt.loss.value
            # mt.loss_classifier.median mt.loss_classifier.max
            # mt.loss_box_reg.value mt.loss_objectness.value
            # mt.loss_rpn_box_reg.median

            # update the learning rate
            lr_scheduler.step()

            # evaluate on test dataset ever val_epoch epochs
            if (epoch % val_epoch) == (val_epoch - 1):
                writer.add_scalar('Detector/Validation_Loss',
                                  mv.loss.median, epoch + 1)

            # save snapshot every snapshot_epochs
            if (epoch % snapshot_epoch) == (snapshot_epoch - 1):
                print('saving snapshot at epoch: {}'.format(epoch))

                # save epoch
                os.makedirs(os.path.join(
                    save_folder, 'snapshots'), exist_ok=True)
                snapshot_name = os.path.join(save_folder,
                                             'snapshots',
                                             model_name + '_epoch' + str(epoch + 1) + '.pth')
                torch.save(model.state_dict(), snapshot_name)
                # print('snapshot name: {}',format(snapshot_name))

        print('training complete')

        # print times
        end_time = time.time()
        sec = end_time - start_time
        print('training time: {} sec'.format(sec))
        print('training time: {} min'.format(sec / 60.0))
        print('training time: {} hrs'.format(sec / 3600.0))

        # save model
        model_save_path = os.path.join(save_folder, model_name + '.pth')
        torch.save(model.state_dict(), model_save_path)
        print('model saved: {}'.format(model_save_path))

        # set model
        self._model = model
        self._model_name = model_name
        self._model_folder = save_folder
        self.set_model_path(model_save_path)
        self._epoch = epoch

        return model, model_save_path

    def load_model(self,
                   model_path=None,
                   num_classes=2,
                   map_location="cuda:0",
                   annotation_type='poly'):
        """ load model to self based on model_path """

        if model_path is None:
            model_path = self._model_path

        if annotation_type == 'poly':
            print('loading maskrcnn')
            model = self.build_maskrcnn_model(num_classes)
        else:
            print('loading fasterrcnn')
            model = self.build_fasterrcnn_model(num_classes)
        model.load_state_dict(torch.load(
            model_path, map_location=map_location))
        print('loaded model: {}'.format(model_path))
        model.to(self._device)
        self.set_model(model)
        self.set_model_path(model_path)
        self.set_model_folder(os.path.dirname(model_path))

        return model

    def set_snapshot(self,
                     epoch,
                     snapshot_folder=None):
        """ set snapshot for epoch, deals with early stopping """
        # change the model_path and model of self to epoch given a model name
        # (.pth) and an epoch number find the .pth file of the model name find
        # all the snapshots in the snapshots folder from training replace said
        # .pth file with the nearest epoch notw, instead just set model_path and
        # model to retain traceability

        # this function finds closest epoch in snapshots folder and sets
        # model_path, model to relevant .pth file

        print('old model path: {}'.format(self._model_path))

        if snapshot_folder is None:
            snapshot_folder = os.path.join(self._model_folder, 'snapshots')

        # find all filenames in snapshot folder
        snapshot_files = os.listdir(snapshot_folder)
        pattern = 'epoch'
        e = []
        for f in snapshot_files:
            if f.endswith('.pth'):
                # find the string that matches the pattern and split it
                n = re.split(pattern, f, maxsplit=0, flags=0)
                # take the second portion of the string (after epoch) to reclaim
                # the epoch number from the snapshot file name
                e.append(int(n[1][:-4]))
        e = np.array(e)

        # find closest e[i] to epoch
        diff_e = np.sqrt((e - epoch)**2)
        i_emin = np.argmin(diff_e)

        # closest snapshot index is indexed by i_emin
        print('closest snapshot epoch: {}'.format(snapshot_files[i_emin]))
        print('corresponding epoch number: {}'.format(e[i_emin]))

        # set object model and model path and epoch number
        self._model_path = os.path.join(
            snapshot_folder, snapshot_files[i_emin])
        self._epoch = e[i_emin]
        self.load_model(annotation_type=self._annotation_type)

        return True

    def find_file(self, file_pattern, folder):
        """
        find filename given file pattern in a folder
        """
        # TODO check valid inputs

        files = os.listdir(folder)
        file_find = []
        for f in files:
            if f.endswith(file_pattern):
                file_find.append(f)

        if len(file_find) <= 0:
            print('Warning: no files found matching pattern {}'.format(file_pattern))
        elif len(file_find) == 1:
            print('Found file: {}'.format(file_find[0]))
        elif len(file_find) > 1:
            print('Warning: found multiple files matching string pattern')
            for i, ff in enumerate(file_find):
                print('{}: {}'.format(i, ff))

        return file_find

    def get_predictions_image(self,
                              image,
                              conf_thresh,
                              nms_iou_thresh,
                              annotation_type='poly',
                              mask_threshold=0.5):
        """ take in model, single image, thresholds, return bbox predictions for
        scores > threshold """

        # image incoming is a tensor, since it is from a dataloader object
        # TODO could call self.model.eval(), but for now, just want to port the scripts/functions
        self._model.eval()

        if torch.cuda.is_available():
            image = image.to(self._device)
            # added, unsure if this will cause errors
            self._model.to(self._device)

        # do model inference on single image
        # start_time = time.time()
        pred = self._model([image])
        # end_time = time.time()
        # time_infer = end_time - start_time
        # print(f'time infer just model = {time_infer}')

        # apply non-maxima suppression
        # TODO nms based on iou, what about masks?
        keep = torchvision.ops.nms(
            pred[0]['boxes'], pred[0]['scores'], nms_iou_thresh)

        pred_class = [i for i in list(pred[0]['labels'][keep].cpu().numpy())]
        # bbox in form: [xmin, ymin, xmax, ymax]?
        pred_boxes = [[bb[0], bb[1], bb[2], bb[3]]
                      for bb in list(pred[0]['boxes'][keep].detach().cpu().numpy())]
        # TODO get centre of bounding boxes
        pred_box_centroids = []
        if len(pred_boxes) > 0:
            for box in pred_boxes:
                cen = self.box_centroid(box)
                pred_box_centroids.append(cen)

        # scores are ordered from highest to lowest
        pred_score = list(pred[0]['scores'][keep].detach().cpu().numpy())
        if annotation_type == 'poly':
            pred_masks = list(pred[0]['masks'][keep].detach().cpu().numpy())
            # binarize mask, then redo boxes for new masks
            pred_bin_masks = []
            pred_bin_boxes = []
            pred_box_centroids = []
            pred_poly_centroids = []
            pred_poly = []
            if len(pred_masks) > 0:
                for mask in pred_masks:

                    mask = np.transpose(mask, (1, 2, 0))
                    bin_mask, ctr, hier, ctr_sqz, poly = self.binarize_confidence_mask(
                        mask, threshold=mask_threshold)
                    # get bounding box for bin_mask
                    pred_bin_masks.append(bin_mask)
                    xmin = min(ctr_sqz[:, 0])
                    ymin = min(ctr_sqz[:, 1])
                    xmax = max(ctr_sqz[:, 0])
                    ymax = max(ctr_sqz[:, 1])
                    bin_bbox = [xmin, ymin, xmax, ymax]
                    pred_bin_boxes.append(bin_bbox)

                    # import code
                    # code.interact(local=dict(globals(), **locals()))

                    pred_poly.append(ctr_sqz)
                    # should overwrite original box centroids
                    pred_box_centroids.append(self.box_centroid(bin_bbox))
                    cen_poly = self.polygon_centroid(ctr_sqz)
                    pred_poly_centroids.append(cen_poly)

        # package
        pred_final = {}
        pred_final['boxes'] = pred_boxes
        pred_final['box_centroids'] = pred_box_centroids
        pred_final['classes'] = pred_class
        pred_final['scores'] = pred_score

        if annotation_type == 'poly':
            pred_final['masks'] = pred_masks
            pred_final['bin_masks'] = pred_bin_masks
            pred_final['bin_boxes'] = pred_bin_boxes
            pred_final['poly_centroids'] = pred_poly_centroids
            pred_final['polygons'] = pred_poly

        # apply confidence threshold
        pred_final = self.threshold_predictions(
            pred_final, conf_thresh, annotation_type)

        return pred_final

    def threshold_predictions(self, pred, thresh, annotation_type='poly'):
        """ apply confidence threshold to predictions """

        pred_boxes = pred['boxes']
        pred_class = pred['classes']
        pred_score = pred['scores']
        pred_box_centroids = pred['box_centroids']
        if annotation_type == 'poly':
            pred_masks = pred['masks']
            pred_bin_masks = pred['bin_masks']
            pred_bin_boxes = pred['bin_boxes']
            pred_poly_centroids = pred['poly_centroids']
            pred_poly = pred['polygons']

        if len(pred_score) > 0:
            if max(pred_score) < thresh:  # none of pred_score > thresh, then return empty
                pred_thresh = []
                pred_boxes = []
                pred_class = []
                pred_score = []
                pred_box_centroids = []
                if annotation_type == 'poly':
                    pred_masks = []
                    pred_bin_masks = []
                    pred_bin_boxes = []
                    pred_poly_centroids = []
                    pred_poly = []
            else:
                pred_thresh = [pred_score.index(
                    x) for x in pred_score if x > thresh][-1]
                pred_boxes = pred_boxes[:pred_thresh+1]
                pred_class = pred_class[:pred_thresh+1]
                pred_score = pred_score[:pred_thresh+1]
                pred_box_centroids = pred_box_centroids[:pred_thresh+1]
                if annotation_type == 'poly':
                    pred_masks = pred_masks[:pred_thresh+1]
                    pred_bin_boxes = pred_bin_boxes[:pred_thresh+1]
                    pred_bin_masks = pred_bin_masks[:pred_thresh+1]
                    pred_poly_centroids = pred_poly_centroids[:pred_thresh+1]
                    pred_poly = pred_poly[:pred_thresh+1]
        else:
            pred_thresh = []
            pred_boxes = []
            pred_class = []
            pred_score = []
            pred_box_centroids = []
            if annotation_type == 'poly':
                pred_masks = []
                pred_bin_masks = []
                pred_bin_boxes = []
                pred_poly_centroids = []
                pred_poly = []

        predictions = {}
        predictions['boxes'] = pred_boxes
        predictions['classes'] = pred_class
        predictions['scores'] = pred_score
        predictions['box_centroids'] = pred_box_centroids

        if annotation_type == 'poly':
            predictions['masks'] = pred_masks
            predictions['bin_masks'] = pred_bin_masks
            predictions['bin_boxes'] = pred_bin_boxes
            predictions['poly_centroids'] = pred_poly_centroids
            predictions['polygons'] = pred_poly

        return predictions

    def cv_imshow(self, image, win_name, wait_time=2000, close_window=True):
        """ show image with win_name for wait_time """
        img = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.namedWindow(win_name, cv.WINDOW_GUI_NORMAL)
        cv.imshow(win_name, img)
        cv.waitKey(wait_time)
        if close_window:
            cv.destroyWindow(win_name)

    def show(self,
             image,
             sample=None,
             predictions=None,
             outcomes=None,
             sample_color=(0, 0, 255),  # RGB
             predictions_color=(255, 0, 0),
             iou_color=(255, 255, 255),
             transpose_image_channels=True,
             transpose_color_channels=False,
             resize_image=False,
             resize_height=(256)):
        """ show image, sample/groundtruth, model predictions, outcomes
        (TP/FP/etc) """
        # TODO rename "show" to something like "create_plot" or "markup", as we
        # don't actually show the image assume image comes in as a tensor, as in
        # the same format it was input into the model

        # set plotting parameters
        gt_box_thick = 12   # groundtruth bounding box
        dt_box_thick = 6    # detection bounding box
        out_box_thick = 3   # outcome bounding box/overlay
        font_scale = 2  # font scale should be function of image size
        font_thick = 2

        if transpose_color_channels:
            # image tensor comes in as [color channels, length, width] format
            print('swap color channels in tensor format')
            image = image[(2, 0, 1), :, :]

        # move to cpu and convert from tensor to numpy array since opencv
        # requires numpy arrays
        image_out = image.cpu().numpy()

        if transpose_image_channels:
            # if we were working with BGR as opposed to RGB
            image_out = np.transpose(image_out, (1, 2, 0))

        # normalize image from 0,1 to 0,255
        image_out = cv.normalize(
            image_out, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

        # ----------------------------------- #
        # first plot groundtruth boxes
        if sample is not None:
            # NOTE we assume sample is also a tensor
            boxes_gt = sample['boxes']
            if len(boxes_gt) > 0:
                n_gt, _ = boxes_gt.size()
                for i in range(n_gt):
                    # TODO just specify int8 or imt16?
                    bb = np.array(boxes_gt[i, :].cpu(), dtype=np.float32)
                    # overwrite the original image with groundtruth boxes
                    image_out = cv.rectangle(image_out,
                                             (int(bb[0]), int(bb[1])),
                                             (int(bb[2]), int(bb[3])),
                                             color=sample_color,
                                             thickness=gt_box_thick)

        # ----------------------------------- #
        # second, plot predictions
        if predictions is not None:
            boxes_pd = predictions['boxes']
            scores = predictions['scores']

            if len(boxes_pd) > 0:
                for i in range(len(boxes_pd)):
                    # TODO just specify int8 or imt16?
                    bb = np.array(boxes_pd[i], dtype=np.float32)
                    image_out = cv.rectangle(image_out,
                                             (int(bb[0]), int(bb[1])),
                                             (int(bb[2]), int(bb[3])),
                                             color=predictions_color,
                                             thickness=dt_box_thick)

                    # add text to top left corner of bbox
                    # no decimals, just x100 for percent
                    sc = format(scores[i] * 100.0, '.0f')
                    cv.putText(image_out,
                               '{}: {}'.format(i, sc),
                               # buffer numbers should be function of font scale
                               (int(bb[0] + 10), int(bb[1] + 30)),
                               fontFace=cv.FONT_HERSHEY_COMPLEX,
                               fontScale=font_scale,
                               color=predictions_color,
                               thickness=font_thick)

        # ----------------------------------- #
        # third, add iou info (within the predicitons if statement)
            if outcomes is not None:
                iou = outcomes['dt_iou']

                # iou is a list or array with iou values for each boxes_pd
                if len(iou) > 0 and len(boxes_pd) > 0:
                    for i in range(len(iou)):
                        bb = np.array(boxes_pd[i], dtype=np.float32)
                        # print in top/left corner of bbox underneath bbox # and
                        # score
                        iou_str = format(iou[i], '.2f')  # max 2 decimal places
                        cv.putText(image_out,
                                   'iou: {}'.format(iou_str),
                                   (int(bb[0] + 10), int(bb[1] + 60)),
                                   fontFace=cv.FONT_HERSHEY_COMPLEX,
                                   fontScale=font_scale,
                                   color=iou_color,
                                   thickness=font_thick)

        # ----------------------------------- #
        # fourth, add outcomes
            if (outcomes is not None) and (sample is not None):
                # for each prediction, if there is a sample, then there is a
                # known outcome being an array from 1-4:
                outcome_list = ['TP', 'FP', 'FN', 'TN']
                # choose color scheme default: blue is groundtruth default: red
                # is detection -> red is false negative green is true positive
                # yellow is false positive
                outcome_color = [(0, 255, 0),   # TP - green
                                 (255, 255, 0),  # FP - yellow
                                 (255, 0, 0),   # FN - red
                                 (0, 0, 0)]     # TN - black
                # structure of the outcomes dictionary outcomes = {'dt_outcome':
                # dt_outcome, # detections, integer index for tp/fp/fn
                # 'gt_outcome': gt_outcome, # groundtruths, integer index for fn
                # 'dt_match': dt_match, # detections, boolean matched or not
                # 'gt_match': gt_match, # gt, boolean matched or not 'fn_gt':
                # fn_gt, # boolean for gt false negatives 'fn_dt': fn_dt, #
                # boolean for dt false negatives 'tp': tp, # true positives for
                # detections 'fp': fp, # false positives for detections
                # 'dt_iou': dt_iou} # intesection over union scores for
                # detections
                dt_outcome = outcomes['dt_outcome']
                if len(dt_outcome) > 0 and len(boxes_pd) > 0:
                    for i in range(len(boxes_pd)):
                        # replot detection boxes based on outcome
                        bb = np.array(boxes_pd[i], dtype=np.float32)
                        image_out = cv.rectangle(image_out,
                                                 (int(bb[0]), int(bb[1])),
                                                 (int(bb[2]), int(bb[3])),
                                                 color=outcome_color[dt_outcome[i]],
                                                 thickness=out_box_thick)
                        # add text top/left corner including outcome type prints
                        # over existing text, so needs to be the same starting
                        # string
                        # no decimals, just x100 for percent
                        sc = format(scores[i] * 100.0, '.0f')
                        ot = format(outcome_list[dt_outcome[i]])
                        cv.putText(image_out,
                                   '{}: {}/{}'.format(i, sc, ot),
                                   (int(bb[0] + 10), int(bb[1] + 30)),
                                   fontFace=cv.FONT_HERSHEY_COMPLEX,
                                   fontScale=font_scale,
                                   color=outcome_color[dt_outcome[i]],
                                   thickness=font_thick)

                # handle false negative cases (ie, groundtruth bboxes)
                boxes_gt = sample['boxes']
                fn_gt = outcomes['fn_gt']
                if len(fn_gt) > 0 and len(boxes_gt) > 0:
                    for j in range(len(boxes_gt)):
                        # gt boxes already plotted, so only replot them if false
                        # negatives
                        if fn_gt[j]:  # if True
                            bb = np.array(
                                boxes_gt[j, :].cpu(), dtype=np.float32)
                            image_out = cv.rectangle(image_out,
                                                     (int(bb[0]), int(bb[1])),
                                                     (int(bb[2]), int(bb[3])),
                                                     color=outcome_color[2],
                                                     thickness=out_box_thick)
                            cv.putText(image_out,
                                       '{}: {}'.format(j, outcome_list[2]),
                                       (int(bb[0] + 10), int(bb[1]) + 30),
                                       fontFace=cv.FONT_HERSHEY_COMPLEX,
                                       fontScale=font_scale,
                                       color=outcome_color[2],  # index for FN
                                       thickness=font_thick)

        # resize image to save space
        if resize_image:
            aspect_ratio = self._image_width / self._image_height
            resize_width = int(resize_height * aspect_ratio)
            resize_height = int(resize_height)
            image_out = cv.resize(image_out,
                                  (resize_width, resize_height),
                                  interpolation=cv.INTER_CUBIC)
        return image_out

    def binarize_confidence_mask(self,
                                 img_gray,
                                 threshold,
                                 ksize=None,
                                 MAX_KERNEL_SIZE=11,
                                 MIN_KERNEL_SIZE=3):
        """ given confidence mask, apply threshold to turn mask into binary image, return binary image and contour"""
        # input mask ranging from 0 to 1, assume mask is a tensor? operation is trivial if numpy array
        # lets first assume a numpy array
        if type(img_gray) is np.ndarray:

            # do binary conversion
            # binmask = mask > threshold
            # import code
            # code.interact(local=dict(globals(), **locals()))
            ret, mask_bin = cv.threshold(img_gray,
                                         threshold,
                                         maxval=1.0,
                                         type=cv.THRESH_BINARY)

            # do morphological operations on mask to smooth it out?
            # open followed by close
            h, w = mask_bin.shape
            imsize = min(h, w)
            if ksize is None:
                # kernel size, some vague function of minimum image size
                ksize = np.floor(0.01 * imsize)

                if ksize % 2 == 0:
                    ksize += 1  # if even, make it odd
                if ksize > MAX_KERNEL_SIZE:
                    ksize = MAX_KERNEL_SIZE
                if ksize < MIN_KERNEL_SIZE:
                    ksize = MIN_KERNEL_SIZE
            ksize = int(ksize)
            kernel = np.ones((ksize, ksize), np.uint8)
            mask_open = cv.morphologyEx(
                mask_bin.astype(np.uint8), cv.MORPH_OPEN, kernel)
            mask_close = cv.morphologyEx(mask_open, cv.MORPH_CLOSE, kernel)

            # minor erosion then dilation by the same amount

            # find bounding polygon of binary image
            # convert images to cv_8u
            contours, hierarchy = cv.findContours(
                mask_close, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            # import code
            # code.interact(local=dict(globals(), **locals()))
            # TODO might have to take the largest/longest contour or one with the largest area
            # so far seems to be  mostly the first one, so we're ok?
            # TODO this iterates, so need to stack these somehow (probably as list?)
            # not sure about stacking the images

            # make sure we sort the contour list in case there are multiple, and we take the largest/longest contour
            contours.sort(key=len, reverse=True)
            # NOTE should only be one contour from the confidence mask
            ctr = contours[0]
            ctrs_sqz = np.squeeze(ctr)
            all_x, all_y = [], []
            for c in ctrs_sqz:
                all_x.append(c[0])
                all_y.append(c[1])
            polygon = {'name': 'polygon',
                       'all_points_x': all_x, 'all_points_y': all_y}

        else:
            print('mask type not ndarray')
            mask_bin = []
            contours = []
            hierarchy = []
            ctrs_sqz = []
            polygon = []

        return mask_bin, contours, hierarchy, ctrs_sqz, polygon

    def simplify_polygon(self, polygon_in, tolerance=None, preserve_topology=False):
        """ simplify polygon, 2D ndarray in, 2D array out """

        polygon = Polygon(polygon_in)
        if tolerance is None:
            scale_rate = 0.01
            # scale the tolerance wrt number of points in polygon
            tolerance = int(np.floor(scale_rate * len(polygon_in)) + 1)
            if tolerance > 10:
                tolerance = 10  # tolerance bounded to 10
        polygon_out = polygon.simplify(
            tolerance=tolerance, preserve_topology=preserve_topology)
        polygon_out_coords = np.array(polygon_out.exterior.coords)
        return polygon_out_coords

    def polygon_area(self, polygon_in):
        """ compute area of polygon using Shapely"""
        poly = Polygon(polygon_in)
        return poly.area  # units of pixels

    def polygon_centroid(self, polygon_in):
        """ compute centroid of polygon using Shapely polygon """
        # input must be a [Nx2] 2D array
        poly = Polygon(polygon_in)
        cen = poly.centroid
        cx = cen.coords[0][0]
        cy = cen.coords[0][1]
        return (cx, cy)

    def box_centroid(self, box_in):
        """ compute box centroid """
        # box input as x0, y0, x1, y2
        # bbox in form: [xmin, ymin, xmax, ymax]?
        xmin = box_in[0]
        ymin = box_in[1]
        w = box_in[2] - box_in[0]
        h = box_in[3] - box_in[1]

        cx = xmin + w / 2.0
        cy = ymin + h / 2.0
        return ([cx, cy])

    def show_mask(self,
                  image,
                  sample=None,
                  predictions=None,
                  outcomes=None,
                  sample_color=(0, 0, 255),  # RGB
                  predictions_color=(255, 0, 0),
                  iou_color=(255, 255, 255),
                  transpose_image_channels=True,
                  transpose_color_channels=False,
                  resize_image=False,
                  resize_height=(256),
                  mask_threshold=0.5):
        """ show image, sample/groundtruth, model predictions, outcomes
        (TP/FP/etc) """
        # TODO rename "show" to something like "create_plot" or "markup", as we
        # don't actually show the image assume image comes in as a tensor, as in
        # the same format it was input into the model

        # set plotting parameters
        gt_box_thick = 6   # groundtruth bounding box
        dt_box_thick = 3    # detection bounding box
        out_box_thick = 2   # outcome bounding box/overlay
        font_scale = 1  # TODO font scale should be function of image size
        font_thick = 1
        sample_mask_color = [0, 0, 255]  # RGB
        # sample_mask_alpha = 0.25

        pred_mask_color = [255, 0, 0]  # RGB
        # pred_mask_alpha = 0.25

        if transpose_color_channels:
            # image tensor comes in as [color channels, length, width] format
            print('swap color channels in tensor format')
            image = image[(2, 0, 1), :, :]

        # move to cpu and convert from tensor to numpy array since opencv
        # requires numpy arrays
        image_out = image.cpu().numpy()

        if transpose_image_channels:
            # if we were working with BGR as opposed to RGB
            image_out = np.transpose(image_out, (1, 2, 0))

        # normalize image from 0,1 to 0,255
        image_out = cv.normalize(
            image_out, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

        # ----------------------------------- #
        # first plot groundtruth boxes
        if sample is not None:
            # NOTE we assume sample is also a tensor
            boxes_gt = sample['boxes']
            if len(boxes_gt) > 0:
                n_gt, _ = boxes_gt.size()
                for i in range(n_gt):
                    # TODO just specify int8 or imt16?
                    bb = np.array(boxes_gt[i, :].cpu(), dtype=np.float32)
                   # overwrite the original image with groundtruth boxes
                    image_out = cv.rectangle(image_out,
                                             (int(bb[0]), int(bb[1])),
                                             (int(bb[2]), int(bb[3])),
                                             color=sample_color,
                                             thickness=gt_box_thick)
            points_gt = sample['points']
            if len(points_gt) > 0:
                for p in points_gt:
                    # plot polygon centroid for sample
                    # get image size, make circle a function of image size
                    h, w, _ = image_out.shape
                    imsize = min((h, w))
                    ptsize = int(imsize / 75)
                    xc = int(p[0])
                    yc = int(p[1])
                    image_out = cv.circle(image_out,
                                          center=(xc, yc),
                                          radius=ptsize,
                                          color=sample_color,
                                          thickness=dt_box_thick,
                                          lineType=cv.LINE_8)

            masks = sample['masks']
            if len(masks) > 0:  # probably not necessary - "if there is a mask"
                # mask = mask[(2, 0, 1), :, :] # mask is binary
                for i in range(len(masks)):
                    mask = masks[i]
                    mask = mask.cpu().numpy()
                    # mask = np.transpose(mask, (1,2,0))
                    # NOTE binarize confidence mask is meant to take in a nonbinary image and make it binary
                    # in this case... we probably don't need this full functionality, also opening/closing
                    # shouldn't, but might affect the original gt mask?
                    mask_bin, ctr, hier, ctr_sqz, poly = self.binarize_confidence_mask(
                        mask, mask_threshold)
                    # note: poly here is just for dictionary output, ctr is the 2D numpy array we want!

                    # CONTOUR CODE - CURRENT
                    # polycoord = self.simplify_polygon(ctr)

                    image_out = cv.drawContours(image_out,
                                                ctr,
                                                0,  # show the biggest contour
                                                color=sample_color,
                                                thickness=gt_box_thick,
                                                lineType=cv.LINE_8,
                                                hierarchy=hier,
                                                maxLevel=0)

                    # ADDWEIGHTED CODE - DEPRACATED
                    # import code
                    # code.interact(local=dict(globals(), **locals()))
                    # mask = np.transpose(mask, (1, 2, 0))
                    # image_overlay = image_out.copy()
                    # make mask a colored image, as opposed to a binary thing
                    # mask2 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

                    # mask2[:,:,0] = mask2[:,:,0] * sample_mask_color[0] # BGR
                    # mask2[:,:,1] = mask2[:,:,1] * sample_mask_color[1] # BGR
                    # mask2[:,:,2] = mask2[:,:,2] * sample_mask_color[2] # BGR
                    # import code
                    # code.interact(local=dict(globals(), **locals()))
                    # image_out = cv.addWeighted(src1=mask2,
                    #                         alpha=sample_mask_alpha,
                    #                         src2=image_out,
                    #                         beta=1-sample_mask_alpha,
                    #                         gamma=0)

                # plt.imshow(image_out2)
                # plt.show()

        # ----------------------------------- #
        # second, plot predictions
        if predictions is not None:
            # boxes_pd = predictions['boxes']
            boxes_pd = predictions['bin_boxes']
            scores = predictions['scores']
            masks = predictions['bin_masks']
            boxes_cen = predictions['box_centroids']
            poly_cen = predictions['poly_centroids']

            if len(boxes_pd) > 0:
                for i in range(len(boxes_pd)):
                    # TODO just specify int8 or imt16?
                    bb = np.array(boxes_pd[i], dtype=np.float32)
                    image_out = cv.rectangle(image_out,
                                             (int(bb[0]), int(bb[1])),
                                             (int(bb[2]), int(bb[3])),
                                             color=predictions_color,
                                             thickness=dt_box_thick)

                    # add text to top left corner of bbox
                    # no decimals, just x100 for percent
                    sc = format(scores[i] * 100.0, '.0f')
                    cv.putText(image_out,
                               '{}: {}'.format(i, sc),
                               # buffer numbers should be function of font scale
                               (int(bb[0] + 10), int(bb[1] + 30)),
                               fontFace=cv.FONT_HERSHEY_COMPLEX,
                               fontScale=font_scale,
                               color=predictions_color,
                               thickness=font_thick)

            if len(masks) > 0:
                for i in range(len(masks)):
                    mask = masks[i]
                    # mask = np.transpose(mask, (1, 2, 0))

                    mask_bin, ctr, hier, ctr_qz, poly = self.binarize_confidence_mask(
                        mask, mask_threshold)
                    # note: poly here is just for dictionary output, ctr is the 2D numpy array we want!

                    # CONTOUR CODE - CURRENT
                    # polycoord = self.simplify_polygon(ctr)

                    image_out = cv.drawContours(image_out,
                                                ctr,
                                                0,
                                                color=pred_mask_color,
                                                thickness=dt_box_thick,
                                                lineType=cv.LINE_8,
                                                hierarchy=hier,
                                                maxLevel=0)

                    # mask2 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
                    # mask2[:,:,0] = mask2[:,:,0] * pred_mask_color[0] # BGR
                    # mask2[:,:,1] = mask2[:,:,1] * pred_mask_color[1] # BGR
                    # mask2[:,:,2] = mask2[:,:,2] * pred_mask_color[2] # BGR
                    # mask2 = mask2.astype(np.uint8)

                    # # import code
                    # # code.interact(local=dict(globals(), **locals()))
                    # image_out = cv.addWeighted(src1=mask2,
                    #                             alpha=pred_mask_alpha,
                    #                             src2=image_out,
                    #                             beta=1-pred_mask_alpha,
                    #                             gamma=0)
                    # import code
                    # code.interact(local=dict(globals(), **locals()))

            if len(poly_cen) > 0:
                for pc in poly_cen:
                    # plot polygon centroid
                    # get image size, make circle a function of image size
                    h, w, _ = image_out.shape
                    imsize = min((h, w))
                    ptsize = int(imsize / 75)
                    xc = int(pc[0])
                    yc = int(pc[1])
                    image_out = cv.circle(image_out,
                                          center=(xc, yc),
                                          radius=ptsize,
                                          color=pred_mask_color,
                                          thickness=dt_box_thick,
                                          lineType=cv.LINE_8)

        # ----------------------------------- #
        # third, add iou info (within the predicitons if statement)
            if outcomes is not None:
                iou = outcomes['dt_iou']

                # iou is a list or array with iou values for each boxes_pd
                if len(iou) > 0 and len(boxes_pd) > 0:
                    for i in range(len(iou)):
                        bb = np.array(boxes_pd[i], dtype=np.float32)
                        # print in top/left corner of bbox underneath bbox # and
                        # score
                        iou_str = format(iou[i], '.2f')  # max 2 decimal places
                        cv.putText(image_out,
                                   'iou: {}'.format(iou_str),
                                   # TODO should place text on mask contour
                                   (int(bb[0] + 10), int(bb[1] + 60)),
                                   fontFace=cv.FONT_HERSHEY_COMPLEX,
                                   fontScale=font_scale,
                                   color=iou_color,
                                   thickness=font_thick)

        # ----------------------------------- #
        # fourth, add outcomes
            if (outcomes is not None) and (sample is not None):
                # for each prediction, if there is a sample, then there is a
                # known outcome being an array from 1-4:
                outcome_list = ['TP', 'FP', 'FN', 'TN']
                # choose color scheme default: blue is groundtruth default: red
                # is detection -> red is false negative green is true positive
                # yellow is false positive
                outcome_color = [(0, 255, 0),   # TP - green
                                 (255, 255, 0),  # FP - yellow
                                 (255, 0, 0),   # FN - red
                                 (0, 0, 0)]     # TN - black
                # structure of the outcomes dictionary outcomes = {'dt_outcome':
                # dt_outcome, # detections, integer index for tp/fp/fn
                # 'gt_outcome': gt_outcome, # groundtruths, integer index for fn
                # 'dt_match': dt_match, # detections, boolean matched or not
                # 'gt_match': gt_match, # gt, boolean matched or not 'fn_gt':
                # fn_gt, # boolean for gt false negatives 'fn_dt': fn_dt, #
                # boolean for dt false negatives 'tp': tp, # true positives for
                # detections 'fp': fp, # false positives for detections
                # 'dt_iou': dt_iou} # intesection over union scores for
                # detections
                dt_outcome = outcomes['dt_outcome']
                if len(dt_outcome) > 0 and len(boxes_pd) > 0:
                    for i in range(len(boxes_pd)):
                        # replot detection boxes based on outcome
                        bb = np.array(boxes_pd[i], dtype=np.float32)
                        image_out = cv.rectangle(image_out,
                                                 (int(bb[0]), int(bb[1])),
                                                 (int(bb[2]), int(bb[3])),
                                                 color=outcome_color[dt_outcome[i]],
                                                 thickness=out_box_thick)
                        # add text top/left corner including outcome type prints
                        # over existing text, so needs to be the same starting
                        # string
                        # no decimals, just x100 for percent
                        sc = format(scores[i] * 100.0, '.0f')
                        ot = format(outcome_list[dt_outcome[i]])
                        cv.putText(image_out,
                                   '{}: {}/{}'.format(i, sc, ot),
                                   (int(bb[0] + 10), int(bb[1] + 30)),
                                   fontFace=cv.FONT_HERSHEY_COMPLEX,
                                   fontScale=font_scale,
                                   color=outcome_color[dt_outcome[i]],
                                   thickness=font_thick)

                # handle false negative cases (ie, groundtruth bboxes)
                boxes_gt = sample['boxes']
                fn_gt = outcomes['fn_gt']
                if len(fn_gt) > 0 and len(boxes_gt) > 0:
                    for j in range(len(boxes_gt)):
                        # gt boxes already plotted, so only replot them if false
                        # negatives
                        if fn_gt[j]:  # if True
                            bb = np.array(
                                boxes_gt[j, :].cpu(), dtype=np.float32)
                            image_out = cv.rectangle(image_out,
                                                     (int(bb[0]), int(bb[1])),
                                                     (int(bb[2]), int(bb[3])),
                                                     color=outcome_color[2],
                                                     thickness=out_box_thick)
                            cv.putText(image_out,
                                       '{}: {}'.format(j, outcome_list[2]),
                                       (int(bb[0] + 10), int(bb[1]) + 30),
                                       fontFace=cv.FONT_HERSHEY_COMPLEX,
                                       fontScale=font_scale,
                                       color=outcome_color[2],  # index for FN
                                       thickness=font_thick)

        # resize image to save space
        if resize_image:
            aspect_ratio = self._image_width / self._image_height
            resize_width = int(resize_height * aspect_ratio)
            resize_height = int(resize_height)
            image_out = cv.resize(image_out,
                                  (resize_width, resize_height),
                                  interpolation=cv.INTER_CUBIC)
        return image_out

    def is_valid_image(self, image):
        """ check if image is valid type """
        # is image a... tensor/numpy array/PIL image?
        check = False
        check_type = False
        check_size = False

        if isinstance(image, torch.Tensor):
            check_type = True
        # is the image range correct? (0-1 automatically for Tensors)
        # is image a width x height x 3 matrix?
        # is the image the right width/height?

        if image.size() == self.get_image_tensor_size():
            check_size = True

        if check_type and check_size:
            check = True

        return check


    def check_image(self, image):
        """ check valid image input (numpy array or tensor) """
        # TODO: check for valid array size/shape
        if isinstance(image, np.ndarray) or torch.is_tensor(image):
            return True
        else:
            return False


    def infer_image(self,
                    image,
                    sample=None,
                    imshow=False,
                    imsave=False,
                    save_dir=None,
                    image_name=None,
                    conf_thresh=0.5,
                    iou_thresh=0.5,
                    annotation_type='poly'):
        """ do inference on a single image """
        # assume image comes in as a tensor for now (eg, from image, sample in
        # dataset)

        if not self.check_image(image):
            print(f'image type: {type(image)}')
            raise TypeError(image, 'image must be numpy array or pytorch tensor')

        if isinstance(image, np.ndarray):
            c, h, w = self.get_image_tensor_size()
            [hi, wi, ci] = image.shape

            # if height or width don't match
            if (h != hi) or (w != wi):
                # print('rescaling image to expected image size')
                tform_rsc = WDP.Rescale(h)
                image = tform_rsc(image)

            # check valid image size ()
            # convert np array to image tensor
            # print('image is numpy array, converting to tensor')
            image = tv_transform.to_tensor(image)

        # if isinstance(image, PIL.Image.Image):
        #     print('image is PIL image, convert to tensor')
        #     image = torch.tensor(image)

        with torch.no_grad():
            self._model.to(self._device)
            image = image.to(self._device)

            self._model.eval()

            # TODO accept different types of image input (tensor, numpy array,
            # PIL, filename?)

            if image_name is None:
                image_name = self._model_name + '_image'
            # start_time = time.time()
            pred = self.get_predictions_image(
                image, conf_thresh, iou_thresh, annotation_type)
            # end_time = time.time()
            # time_infer = end_time - start_time
            # print(f'time to infer: {time_infer}')

            if imsave or imshow:
                if annotation_type == 'poly':
                    image = self.show_mask(image,
                                           sample=sample,
                                           predictions=pred)
                else:
                    image = self.show(image,
                                      sample=sample,
                                      predictions=pred)
            if imsave:
                if save_dir is None:
                    save_dir = os.path.join('output', self._model_folder)
                # print('save dir: ', save_folder)
                os.makedirs(save_dir, exist_ok=True)
                save_image_name = os.path.join(save_dir, image_name + '.png')
                # print('save img: ', save_image_name)
                image_out_bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)
                cv.imwrite(save_image_name, image_out_bgr)

            if imshow:
                self.cv_imshow(image, win_name=str(image_name))

        return image, pred

    def infer_dataset(self,
                      dataset,
                      conf_thresh=0.5,
                      iou_thresh=0.5,
                      save_folder=None,
                      save_subfolder='infer_dataset',
                      imshow=False,
                      imsave=False,
                      wait_time=1000,
                      image_name_suffix=None,
                      annotation_type='poly'):
        """ do inference on entire dataset """
        with torch.no_grad():
            self._model.to(self._device)
            self._model.eval()

            # out = []
            predictions = []

            if save_folder is None:
                save_folder = os.path.join(
                    'output', self._model_folder, save_subfolder)

            if imsave:
                os.makedirs(save_folder, exist_ok=True)

            print('number of images to infer: {}'.format(len(dataset)))

            for image, sample in dataset:
                image_id = sample['image_id'].item()
                image_name = dataset.annotations[image_id]['filename'][:-4]

                pred = self.get_predictions_image(image,
                                                  conf_thresh,
                                                  iou_thresh,
                                                  annotation_type)

                if annotation_type == 'poly':
                    image_out = self.show_mask(image,
                                               sample=sample,
                                               predictions=pred)
                else:
                    image_out = self.show(
                        image, sample=sample, predictions=pred)

                if imsave:
                    if image_name_suffix is None:
                        save_image_name = os.path.join(save_folder,
                                                       image_name + '.png')
                    else:
                        save_image_name = os.path.join(save_folder,
                                                       image_name + image_name_suffix + '.png')

                    image_out_bgr = cv.cvtColor(image_out, cv.COLOR_RGB2BGR)
                    cv.imwrite(save_image_name, image_out_bgr)

                if imshow:
                    self.cv_imshow(image_out, image_name, wait_time=wait_time)

                # saving output out_tensor = model()
                predictions.append(pred)

        return predictions

    def infer_video(self,
                    capture=None,
                    fps=10,
                    video_out_name=None,
                    save_folder=None,
                    MAX_FRAMES=1000,
                    vidshow=True,
                    conf_thresh=0.5,
                    iou_thresh=0.5,
                    annotation_type='poly'):
        """ video inference from a webcam defined by capture (see opencv video
        capture object) """

        if capture is None:
            capture = cv.VideoCapture(0)

        # get width/height of orininal image
        w = capture.get(cv.CAP_PROP_FRAME_WIDTH)
        h = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
        print('original video capture resolution: width={}, height={}'.format(w, h))
        # images will get resized to what the model was trained for, so get the
        # output video size
        print('resized video resolution: width={}, height={}'.format(
            self._image_width, self._image_height))

        # TODO set webcam exposure settings

        # save video settings/location
        if save_folder is None:
            save_folder = os.path.join('output', self._model_folder, 'video')
            os.makedirs(save_folder, exist_ok=True)

        if video_out_name is None:
            now_str = self.get_now_str()
            video_out_name = self._model_name + now_str + '_video.avi'

        video_out_path = os.path.join(save_folder, video_out_name)

        # set video writer and encoder
        video_write = cv.VideoWriter_fourcc(*'XVID')
        video_out = cv.VideoWriter(video_out_path,
                                   fourcc=video_write,
                                   fps=fps,
                                   frameSize=(int(self._image_width), int(self._image_height)))

        tform_rsc = WDP.Rescale(self._hp['rescale_size'])

        # read in video
        i = 0
        self._model.to(self._device)
        self._model.eval()

        while (capture.isOpened() and i < MAX_FRAMES):

            # calculate how long it takes to compute with each frame
            start_time = time.time()

            # capture frame (image) from video device
            ret, frame = capture.read()

            if ret:

                # resize to size expected by model
                frame = tform_rsc(frame)

                # convert frame to tensor and send to device
                frame = tv_transform.To_Tensor(frame)
                frame.to(self._device)

                # model inference
                pred = self.get_predictions_image(
                    frame, conf_thresh, iou_thresh, annotation_type)
                frame_out = self.show(frame, predictions=pred)

                # write image to video
                frame_out = cv.cvtColor(frame_out, cv.COLOR_RGB2BGR)
                video_out.write(frame_out)

                if vidshow:
                    self.cv_imshow(frame_out, 'video',
                                   wait_key=0, close_window=False)
                    # NOTE not sure what wait time (ms) should be for video TODO
                    # check default, 0?

                # compute cycle time
                end_time = time.time()
                sec = end_time - start_time
                print('cycle time: {} sec'.format(sec))

                # wait for 1 ms, or if q is pressed, stop video capture cycle
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

                # increment video/frame counter
                i += 1

            else:
                print('Error: ret is not True')
                # TODO should be Raise statement
                break

        # close video object
        capture.release()
        video_out.release()
        cv.destroyAllWindows()

        return video_out_path

    def compute_iou_bbox(self, boxA, boxB):
        """
        compute intersection over union for bounding boxes box = [xmin, ymin,
        xmax, ymax], tensors
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

    def compute_match_cost(self, score, iou, weights=None):
        """ compute match cost metric """
        # compute match cost based on score and iou this is the arithmetic mean,
        # consider the geometric mean
        # https://en.wikipedia.org/wiki/Geometric_mean which automatically
        # punishes the 0 IOU case because of the multiplication small numbers
        # penalized harder
        if weights is None:
            weights = np.array([0.5, 0.5])
        # cost = weights[0] * score + weights[1] * iou # classic weighting
        cost = np.sqrt(score * iou)
        return cost

    def compute_outcome_image(self,
                              pred,
                              sample,
                              DECISION_IOU_THRESH,
                              DECISION_CONF_THRESH):
        """ compute outcome (tn/fp/fn/tp) for single image """
        # output: outcome, tn, fp, fn, tp for the image
        dt_bbox = pred['boxes']
        dt_scores = pred['scores']
        gt_bbox = sample['boxes']

        ndt = len(dt_bbox)
        ngt = len(gt_bbox)

        # compute ious for each dt_bbox, compute iou and record confidence score
        gt_iou_all = np.zeros((ndt, ngt))

        # for each detection bounding box, find the best-matching groundtruth
        # bounding box gt/dt_match is False - not matched, True - matched
        dt_match = np.zeros((ndt,), dtype=bool)
        gt_match = np.zeros((ngt,), dtype=bool)

        # gt/dt_match_id is the index of the match vector eg. gt = [1 0 2 -1]
        # specifies that gt_bbox[0] is matched to dt_bbox[1], gt_bbox[1] is
        # matched to dt_bbox[0], gt_bbox[2] is matched to dt_bbox[2], gt_bbox[3]
        # is not matched to any dt_bbox
        dt_match_id = -np.ones((ndt,), dtype=int)
        gt_match_id = -np.ones((ngt,), dtype=int)

        # dt_assignment_idx = -np.ones((len(dt_bbox),), dtype=int)
        dt_iou = -np.ones((ndt,))

        # find the max overlapping detection box with the gt box
        for i in range(ndt):
            gt_iou = []
            if ngt > 0:
                for j in range(ngt):
                    iou = self.compute_iou_bbox(dt_bbox[i], gt_bbox[j])
                    gt_iou.append(iou)
                gt_iou_all[i, :] = np.array(gt_iou)

                # NOTE finding the max may be insufficient dt_score also
                # important consideration find max iou from gt_iou list
                idx_max = gt_iou.index(max(gt_iou))
                iou_max = gt_iou[idx_max]
                dt_iou[i] = iou_max
            else:
                dt_iou[i] = 0

        # calculate cost of each match/assignment
        cost = np.zeros((ndt, ngt))
        for i in range(ndt):
            if ngt > 0:
                for j in range(ngt):
                    cost[i, j] = self.compute_match_cost(
                        dt_scores[i], gt_iou_all[i, j])
            # else:
                # import code code.interact(local=dict(globals(), **locals()))
                # cost[i,j] = 0

        # choose assignment based on max of matrix. If equal cost, we choose the
        # first match higher cost is good - more confidence, more overlap

        # if there are any gt boxes, then we need to choose an assignment. if no
        # gt, then... dt_iou is all zero, as there is nothing to iou with!
        if ngt > 0:
            for j in range(ngt):
                dt_cost = cost[:, j]
                i_dt_cost_srt = np.argsort(dt_cost)[::-1]

                for i in i_dt_cost_srt:
                    if (not dt_match[i]) and \
                        (dt_scores[i] >= DECISION_CONF_THRESH) and \
                            (gt_iou_all[i, j] >= DECISION_IOU_THRESH):
                        # if the detection box in question is not already
                        # matched
                        dt_match[i] = True
                        gt_match[j] = True
                        dt_match_id[i] = j
                        gt_match_id[j] = i
                        dt_iou[i] = gt_iou_all[i, j]
                        break
                        # stop iterating once the highest-scoring detection
                        # satisfies the criteria\

        tp = np.zeros((ndt,), dtype=bool)
        fp = np.zeros((ndt,), dtype=bool)
        # fn = np.zeros((ngt,), dtype=bool) now determine if TP, FP, FN TP: if
        # our dt_bbox is sufficiently within a gt_bbox with a high-enough
        # confidence score we found the right thing FP: if our dt_bbox is a high
        # confidence score, but is not sufficiently within a gt_bbox (iou is
        # low) we found the wrong thing FN: if our dd_bbox is within a gt_bbox,
        # but has low confidence score we didn't find the right thing TN:
        # basically everything else in the scene, not wholely relevant for
        # PR-curves

        # From the slides: TP: we correctly found the weed FP: we think we found
        # the weed, but we did not FN: we missed a weed TN: we correctly ignored
        # everything else we were not interested in (doesn't really show up for
        # binary detectors)
        dt_iou = np.array(dt_iou)
        dt_scores = np.array(dt_scores)

        # TP: if dt_match True, also, scores above certain threshold
        tp = np.logical_and(dt_match, dt_scores >= DECISION_CONF_THRESH)
        fp = np.logical_or(np.logical_not(dt_match),
                           np.logical_and(dt_match, dt_scores < DECISION_CONF_THRESH))
        fn_gt = np.logical_not(gt_match)
        fn_dt = np.logical_and(dt_match, dt_scores < DECISION_CONF_THRESH)

        # define outcome as an array, each element corresponding to tp/fp/fn for
        # 0/1/2, respectively
        dt_outcome = -np.ones((ndt,), dtype=np.int16)
        for i in range(ndt):
            if tp[i]:
                dt_outcome[i] = 0
            elif fp[i]:
                dt_outcome[i] = 1
            elif fn_dt[i]:
                dt_outcome[i] = 2

        gt_outcome = -np.ones((ngt,), dtype=np.int16)
        if ngt > 0:
            for j in range(ngt):
                if fn_gt[j]:
                    gt_outcome = 1

        # package outcomes
        outcomes = {'dt_outcome': dt_outcome,  # detections, integer index for tp/fp/fn
                    'gt_outcome': gt_outcome,  # groundtruths, integer index for fn
                    'dt_match': dt_match,  # detections, boolean matched or not
                    'gt_match': gt_match,  # gt, boolean matched or not
                    'fn_gt': fn_gt,  # boolean for gt false negatives
                    'fn_dt': fn_dt,  # boolean for dt false negatives
                    'tp': tp,  # true positives for detections
                    'fp': fp,  # false positives for detections
                    'dt_iou': dt_iou,  # intesection over union scores for detections
                    'dt_scores': dt_scores,
                    'gt_iou_all': gt_iou_all,
                    'cost': cost}
        return outcomes

    def compute_pr_dataset(self,
                           dataset,
                           predictions,
                           DECISION_CONF_THRESH,
                           DECISION_IOU_THRESH,
                           imsave=False,
                           save_folder=None):
        """ compute single pr pair for entire dataset, given the predictions """
        # note for pr_curve, see get_prcurve

        self._model.eval()
        self._model.to(self._device)

        # annotations from the original dataset object
        dataset_annotations = dataset.annotations

        tp_sum = 0
        fp_sum = 0
        # tn_sum = 0
        fn_sum = 0
        gt_sum = 0
        dt_sum = 0
        idx = 0
        dataset_outcomes = []
        for image, sample in dataset:

            image_id = sample['image_id'].item()

            img_name = dataset_annotations[image_id]['filename'][:-4]
            # else: img_name = 'st' + str(image_id).zfill(3)

            # get predictions
            pred = predictions[idx]
            pred = self.threshold_predictions(
                pred, DECISION_CONF_THRESH, annotation_type='box')

            outcomes = self.compute_outcome_image(pred,
                                                  sample,
                                                  DECISION_IOU_THRESH,
                                                  DECISION_CONF_THRESH)

            # save for given image:
            if imsave:
                # resize image for space-savings!
                image_out = self.show(image,
                                      sample=sample,
                                      predictions=pred,
                                      outcomes=outcomes,
                                      resize_image=True,
                                      resize_height=720)
                imgw = cv.cvtColor(image_out, cv.COLOR_RGB2BGR)

                if save_folder is None:
                    save_folder = os.path.join('output',
                                               self._model_folder,
                                               'outcomes')

                conf_str = format(DECISION_CONF_THRESH, '.2f')
                save_subfolder = 'conf_thresh_' + conf_str
                save_path = os.path.join(save_folder, save_subfolder)
                os.makedirs(save_path, exist_ok=True)

                save_img_name = os.path.join(
                    save_path, img_name + '_outcome.png')
                cv.imwrite(save_img_name, imgw)

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

        f1score = self.compute_f1score(prec, rec)

        print('f1 score = {}'.format(f1score))
        # 1 is good, 0 is bad

        return dataset_outcomes, prec, rec, f1score

    def compute_f1score(self, p, r):
        """ compute f1 score """
        return 2 * (p * r) / (p + r)

    def extend_pr(self,
                  prec,
                  rec,
                  conf,
                  n_points=101,
                  PLOT=False,
                  save_name='outcomes'):
        """ extend precision and recall vectors from 0-1 """
        # typically, pr curve will only span a relatively small range, eg from
        # 0.5 to 0.9 for most pr curves, we want to span the range of recall
        # from 0 to 1 therefore, we "smooth" out precision-recall curve by
        # taking max of precision points for when r is very small (takes the top
        # of the stairs ) take max:
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
            ax.plot(rec, prec, marker='o',
                    linestyle='dashed', label='original')
            ax.plot(rec, prec_new, marker='x', color='red',
                    linestyle='solid', label='max-binned')
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.title('prec-rec, max-binned')
            ax.legend()

            os.makedirs(os.path.join(
                'output', self._model_folder), exist_ok=True)
            save_plot_name = os.path.join(
                'output', self._model_folder, save_name + '_test_pr_smooth.png')
            plt.savefig(save_plot_name)

        # now, expand prec/rec values to extent of the whole precision/recall
        # space:
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

        # now interpolate: prec_x = np.interp(rec_x, rec_temp, prec_temp)
        prec_interpolator = interp1d(rec_temp, prec_temp, kind='linear')
        prec_x = prec_interpolator(rec_x)

        conf_interpolator = interp1d(rec_temp, conf_temp, kind='linear')
        conf_x = conf_interpolator(rec_x)

        if PLOT:
            fig, ax = plt.subplots()
            ax.plot(rec_temp, prec_temp, color='blue',
                    linestyle='dashed', label='combined')
            ax.plot(rec, prec_new, marker='x', color='red',
                    linestyle='solid', label='max-binned')
            ax.plot(rec_x, prec_x, color='green',
                    linestyle='dotted', label='interp')
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.title('prec-rec, interpolated')
            ax.legend()

            os.makedirs(os.path.join(
                'output', self._model_folder), exist_ok=True)
            save_plot_name = os.path.join(
                'output', self._model_folder, save_name + '_test_pr_interp.png')
            plt.savefig(save_plot_name)
            # plt.show()

        # make the "official" plot:
        p_out = prec_x
        r_out = rec_x
        c_out = conf_x
        return p_out, r_out, c_out

    def compute_ap(self, p, r):
        """ compute ap score """
        n = len(r) - 1
        ap = np.sum((r[1:n] - r[0:n-1]) * p[1:n])
        return ap

    def get_confidence_from_pr(self, p, r, c, f1, pg=None, rg=None):
        """ interpolate confidence threshold from p, r and goal values. If no
        goal values, then provide the "best" conf, corresponding to the pr-pair
        closest to the ideal (1,1)
        """
        # TODO check p, r are valid if Pg is not none, check Pg is valid, if
        # not, set it at appropriate value if Rg is not none, then check if Rg
        # is valid, if not, set appropriate value
        if pg is not None and rg is not None:
            # print('TODO: check if pg, rg are valid') find the nearest pg-pair,
            # find the nearest index to said pair
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
            # TODO search for best pg given rg pg = 1 we use rg to interpolate
            # for cg:
            f = interp1d(r, c, bounds_error=False, fill_value=(c[0], c[-1]))
            cg = f(rg)

        elif rg is None and pg is not None:
            # we use pg to interpolate for cg print('using p to interp c')
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

    def get_prcurve(self,
                    dataset,
                    confidence_thresh,
                    nms_iou_thresh,
                    decision_iou_thresh,
                    save_folder,
                    imshow=False,
                    imsave=False,
                    PLOT=False,
                    annotation_type='poly'):
        """ get complete/smoothed pr curve for entire dataset """
        # infer on 0-decision threshold iterate over diff thresholds extend_pr
        # compute_ap save output

        # infer on dataset with 0-decision threshold
        predictions = self.infer_dataset(dataset,
                                         conf_thresh=0,
                                         iou_thresh=nms_iou_thresh,
                                         save_folder=save_folder,
                                         save_subfolder=os.path.join(
                                             'prcurve', 'detections'),
                                         imshow=imshow,
                                         imsave=imsave,
                                         image_name_suffix='_prcurve_0',
                                         annotation_type=annotation_type)

        # iterate over different decision thresholds
        prec = []
        rec = []
        f1score = []
        start_time = time.time()
        for c, conf in enumerate(confidence_thresh):
            print('{}: outcome confidence threshold: {}'.format(c, conf))

            _, p, r, f1 = self.compute_pr_dataset(dataset,
                                                  predictions,
                                                  conf,
                                                  nms_iou_thresh,
                                                  imsave=imsave,
                                                  save_folder=save_folder)

            prec.append(p)
            rec.append(r)
            f1score.append(f1)

        end_time = time.time()

        sec = end_time - start_time
        print('prcurve time: {} sec'.format(sec))
        print('prcurve time: {} min'.format(sec / 60.0))
        print('prcurve time: {} hrs'.format(sec / 3600.0))

        rec = np.array(rec)
        prec = np.array(prec)

        # plot raw PR curve
        fig, ax = plt.subplots()
        ax.plot(rec, prec, marker='o', linestyle='dashed')
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title('precision-recall for varying confidence')
        os.makedirs(save_folder, exist_ok=True)
        save_plot_name = os.path.join(
            save_folder, self._model_name + '_pr_raw.png')
        plt.savefig(save_plot_name)
        if PLOT:
            plt.show()

        # plot F1score
        f1score = np.array(f1score)
        # fig, ax = plt.subplots() ax.plot(rec, f1score, marker='o',
        # linestyle='dashed') plt.xlabel('recall') plt.ylabel('f1 score')
        # plt.title('f1 score vs recall for varying confidence') save_plot_name
        # = os.path.join('output', save_name, save_name + '_test_f1r.png')
        # plt.savefig(save_plot_name) plt.show()

        # smooth the PR curve: take the max precision values along the recall
        # curve we do this by binning the recall values, and taking the max
        # precision from each bin

        p_final, r_final, c_final, = self.extend_pr(
            prec, rec, confidence_thresh)
        ap = self.compute_ap(p_final, r_final)

        # plot final pr curve
        fig, ax = plt.subplots()
        ax.plot(r_final, p_final)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title(
            'prec-rec curve, iou={}, ap = {:.2f}'.format(decision_iou_thresh, ap))
        # ax.legend()
        save_plot_name = os.path.join(
            save_folder, self._model_name + '_pr.png')
        plt.savefig(save_plot_name)
        if PLOT:
            plt.show()

        print('ap score: {:.5f}'.format(ap))
        print('max f1 score: {:.5f}'.format(max(f1score)))

        # save ap, f1score, precision, recall, etc
        res = {'precision': p_final,
               'recall': r_final,
               'ap': ap,
               'f1score': f1score,
               'confidence': c_final}
        save_file = os.path.join(
            save_folder, self._model_name + '_prcurve.pkl')
        with open(save_file, 'wb') as f:
            pickle.dump(res, f)

        return res

# =========================================================================== #


if __name__ == "__main__":

    # two models
    print('WeedModel.py')
