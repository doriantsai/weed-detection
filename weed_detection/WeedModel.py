#! /usr/bin/env python

"""
weed model class for weed detection class to package model-related
functionality, such as training, inference, evaluation
"""

import os
# from weed_detection.WeedDatasetPoly import WeedDatasetPoly

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

from weed_detection.engine_st import train_one_epoch
# from weed_detection.WeedDataset import *
import weed_detection.WeedDataset as WD
import weed_detection.WeedDatasetPoly as WDP
from weed_detection.PreProcessingToolbox import PreProcessingToolbox as PT

# from webcam import grab_webcam_image


class WeedModel:
    """ collection of functions for model's weed detection """

    def __init__(self,
                 weed_name='serrated tussock',
                 model=None,
                 model_name=None,
                 model_folder=None,
                 model_path=None,
                 device=None,
                 hyper_parameters=None,
                 epoch=None,
                 note=None):

        self._weed_name = weed_name
        # TODO maybe save model type/architecture also, hyper parameters?
        self._model = model
        self._model_name = model_name
        if model_folder is None:
            self._model_folder = model_name
        else:
            self._model_folder = model_folder
        self._model_path = model_path

        # TODO if model_path is not None load model, model name, weed_name, etc
        # everything possible

        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            # device = torch.device('cpu')
        self._device = device

        self._hp = hyper_parameters
        # TODO consider expanding hp from dictionary into actual
        # properties/attributes for more readability
        self._image_height = int(2056/2) # rescale_size
        self._image_width = int(2464 /2)  # should be computed based on aspect ratio


        self._note = note # just a capture-all string TEMP
        self._epoch = epoch


    @property
    def model(self):
        return self._model

    # getters and setters
    def set_model(self, model):
        self._model = model


    def get_model(self):
         return self._model


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


    def build_fasterrcnn_model(self, num_classes):
        """ build fasterrcnn model for set number of classes """

        # load instance of model pre-trained on coco:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace pre-trained head with new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model


    def build_maskrcnn_model(self, num_classes):
        """ build maskrcnn model for set number of classes """

        # load instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # replace pretrained head with new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get number of input features for mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256  # TODO check what this variable is

        # replace mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes)

        return model


    def load_dataset_objects(self, dataset_path, return_dict=True):
        """ load dataset objects given path """
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
                                mask_dir=None):
        # assume tforms already defined outside of this function
        batch_size = hp['batch_size']
        num_workers = hp['num_workers']
        shuffle= hp['shuffle']

        if annotation_type == 'poly':
            dataset = WDP.WeedDatasetPoly(root_dir,
                                      json_file,
                                      transforms,
                                      img_dir=root_dir,
                                      mask_dir=mask_dir)
        else:
            dataset = WD.WeedDataset(root_dir, json_file, transforms)

        # setup dataloaders for efficient access to datasets
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=num_workers,
                                                 collate_fn=self.collate_fn)
        return dataset, dataloader


    def collate_fn(self, batch):
        return tuple(zip(*batch))



    def create_train_test_val_datasets(self,
                                       img_folders,
                                       ann_files,
                                       hp,
                                       dataset_name,
                                       annotation_type='poly',
                                       mask_folders=None):
        """ creates datasets and dataloader objects from train/test/val files
        """
        # arguably, should be in WeedDataset class

        # unpack
        train_folder = img_folders[0]
        test_folder = img_folders[1]
        val_folder = img_folders[2]

        if mask_folders is not None:
            mask_train_folder = mask_folders[0]
            mask_test_folder = mask_folders[1]
            mask_val_folder = mask_folders[2]

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

            writer.add_scalar('Detector/Training_Loss', mt.loss.median, epoch + 1)
            # other loss types from metric logger mt.loss.value
            # mt.loss_classifier.median mt.loss_classifier.max
            # mt.loss_box_reg.value mt.loss_objectness.value
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
                os.makedirs(os.path.join(save_folder, 'snapshots'), exist_ok=True)
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
        self._model_path = model_save_path
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
        model.load_state_dict(torch.load(model_path, map_location=map_location))
        print('loaded model: {}'.format(model_path))
        model.to(self._device)
        self._model = model

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
            snapshot_folder = os.path.join('output', self._model_folder, 'snapshots')

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
        self._model_path = os.path.join(snapshot_folder, snapshot_files[i_emin])
        self._epoch = e[i_emin]
        self.load_model()

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
                              nms_iou_thresh):
        """ take in model, single image, thresholds, return bbox predictions for
        scores > threshold """

        # image incoming is a tensor, since it is from a dataloader object
        self._model.eval()  # TODO could call self.model.eval(), but for now, just want to port the scripts/functions

        if torch.cuda.is_available():
            image = image.to(self._device)
            self._model.to(self._device) # added, unsure if this will cause errors

        # do model inference on single image
        # start_time = time.time()
        pred = self._model([image])
        # end_time = time.time()
        # time_infer = end_time - start_time
        # print(f'time infer just model = {time_infer}')

        # apply non-maxima suppression
        # TODO nms based on iou, what about masks?
        keep = torchvision.ops.nms(pred[0]['boxes'], pred[0]['scores'], nms_iou_thresh)

        pred_class = [i for i in list(pred[0]['labels'][keep].cpu().numpy())]
        pred_boxes = [[bb[0], bb[1], bb[2], bb[3]] for bb in list(pred[0]['boxes'][keep].detach().cpu().numpy())]
        # scores are ordered from highest to lowest
        pred_score = list(pred[0]['scores'][keep].detach().cpu().numpy())
        pred_masks = list(pred[0]['masks'][keep].detach().cpu().numpy())

        # package
        pred_final = {}
        pred_final['boxes'] = pred_boxes
        pred_final['classes'] = pred_class
        pred_final['scores'] = pred_score
        pred_final['masks'] = pred_masks

        # apply confidence threshold
        pred_final = self.threshold_predictions(pred_final, conf_thresh)

        return pred_final


    def threshold_predictions(self, pred, thresh):
        """ apply confidence threshold to predictions """

        pred_boxes = pred['boxes']
        pred_class = pred['classes']
        pred_score = pred['scores']
        pred_masks = pred['masks']

        if len(pred_score) > 0:
            if max(pred_score) < thresh: # none of pred_score > thresh, then return empty
                pred_thresh = []
                pred_boxes = []
                pred_class = []
                pred_score = []
                pred_masks = []
            else:
                pred_thresh = [pred_score.index(x) for x in pred_score if x > thresh][-1]
                pred_boxes = pred_boxes[:pred_thresh+1]
                pred_class = pred_class[:pred_thresh+1]
                pred_score = pred_score[:pred_thresh+1]
                pred_masks = pred_masks[:pred_thresh+1]
        else:
            pred_thresh = []
            pred_boxes = []
            pred_class = []
            pred_score = []
            pred_masks = []

        predictions = {}
        predictions['boxes'] = pred_boxes
        predictions['classes'] = pred_class
        predictions['scores'] = pred_score
        predictions['masks'] = pred_masks

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
             sample_color=(0, 0, 255), # RGB
             predictions_color=(255, 0, 0),
             iou_color=(255, 255, 255),
             transpose_image_channels=True,
             transpose_color_channels=False,
             resize_image=False,
             resize_height=(1080)):
        """ show image, sample/groundtruth, model predictions, outcomes
        (TP/FP/etc) """
        # TODO rename "show" to something like "create_plot" or "markup", as we
        # don't actually show the image assume image comes in as a tensor, as in
        # the same format it was input into the model

        # set plotting parameters
        gt_box_thick = 12   # groundtruth bounding box
        dt_box_thick = 6    # detection bounding box
        out_box_thick = 3   # outcome bounding box/overlay
        font_scale = 2 # font scale should be function of image size
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
        image_out = cv.normalize(image_out, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

        # ----------------------------------- #
        # first plot groundtruth boxes
        if sample is not None:
            # NOTE we assume sample is also a tensor
            boxes_gt = sample['boxes']
            if len(boxes_gt) > 0:
                n_gt, _ = boxes_gt.size()
                for i in range(n_gt):
                    bb = np.array(boxes_gt[i, :].cpu(), dtype=np.float32) # TODO just specify int8 or imt16?
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
                    bb = np.array(boxes_pd[i], dtype=np.float32) # TODO just specify int8 or imt16?
                    image_out = cv.rectangle(image_out,
                                            (int(bb[0]), int(bb[1])),
                                            (int(bb[2]), int(bb[3])),
                                            color=predictions_color,
                                            thickness=dt_box_thick)

                    # add text to top left corner of bbox
                    sc = format(scores[i] * 100.0, '.0f') # no decimals, just x100 for percent
                    cv.putText(image_out,
                               '{}: {}'.format(i, sc),
                               (int(bb[0] + 10), int(bb[1] + 30)), # buffer numbers should be function of font scale
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
                        iou_str = format(iou[i], '.2f') # max 2 decimal places
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
                # choose colour scheme default: blue is groundtruth default: red
                # is detection -> red is false negative green is true positive
                # yellow is false positive
                outcome_color = [(0, 255, 0),   # TP - green
                                (255, 255, 0), # FP - yellow
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
                        sc = format(scores[i] * 100.0, '.0f') # no decimals, just x100 for percent
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
                        if fn_gt[j]: # if True
                            bb = np.array(boxes_gt[j,:].cpu(), dtype=np.float32)
                            image_out = cv.rectangle(image_out,
                                                (int(bb[0]), int(bb[1])),
                                                (int(bb[2]), int(bb[3])),
                                                color=outcome_color[2],
                                                thickness=out_box_thick)
                            cv.putText(image_out,
                                '{}: {}'.format(j, outcome_list[2]),
                                (int(bb[0]+ 10), int(bb[1]) + 30),
                                fontFace=cv.FONT_HERSHEY_COMPLEX,
                                fontScale=font_scale,
                                color=outcome_color[2], # index for FN
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

    def show_mask(self,
                    image,
                    sample=None,
                    predictions=None,
                    outcomes=None,
                    sample_color=(0, 0, 255), # RGB
                    predictions_color=(255, 0, 0),
                    iou_color=(255, 255, 255),
                    transpose_image_channels=True,
                    transpose_color_channels=False,
                    resize_image=False,
                    resize_height=(256),
                    mask_alpha=0.5):
        """ show image, sample/groundtruth, model predictions, outcomes
        (TP/FP/etc) """
        # TODO rename "show" to something like "create_plot" or "markup", as we
        # don't actually show the image assume image comes in as a tensor, as in
        # the same format it was input into the model

        # set plotting parameters
        gt_box_thick = 12   # groundtruth bounding box
        dt_box_thick = 6    # detection bounding box
        out_box_thick = 3   # outcome bounding box/overlay
        font_scale = 2 # font scale should be function of image size
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
        image_out = cv.normalize(image_out, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

        # ----------------------------------- #
        # first plot groundtruth boxes
        if sample is not None:
            # NOTE we assume sample is also a tensor
            boxes_gt = sample['boxes']
            if len(boxes_gt) > 0:
                n_gt, _ = boxes_gt.size()
                for i in range(n_gt):
                    bb = np.array(boxes_gt[i, :].cpu(), dtype=np.float32) # TODO just specify int8 or imt16?
                    # overwrite the original image with groundtruth boxes
                    image_out = cv.rectangle(image_out,
                                            (int(bb[0]), int(bb[1])),
                                            (int(bb[2]), int(bb[3])),
                                            color=sample_color,
                                            thickness=gt_box_thick)
            mask = sample['masks']
            if len(mask) > 0:  # probably not necessary - "if there is a mask"
                # mask = mask[(2, 0, 1), :, :] # mask is binary
                mask = mask.cpu().numpy()
                mask = np.transpose(mask, )
                mask = np.transpose(mask, (1, 2, 0))
                # image_overlay = image_out.copy()
                # make mask a coloured image, as opposed to a binary thing
                mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
                mask2 = mask
                mask_color = [255, 0, 0]
                mask2[:,:,0] = mask[:,:,0] * mask_color[0] # BGR
                mask2[:,:,1] = mask[:,:,1] * mask_color[1] # BGR
                mask2[:,:,2] = mask[:,:,2] * mask_color[2] # BGR
                mask_alpha = 0.5
                import code
                code.interact(local=dict(globals(), **locals()))
                image_out2 = cv.addWeighted(mask2, mask_alpha, image_out, 1-mask_alpha, 0)
                # cv.addWeighted(mask, mask_alpha, )

                plt.imshow(image_out2)
                plt.show()
                import code
                code.interact(local=dict(globals(), **locals()))


        # ----------------------------------- #
        # second, plot predictions
        if predictions is not None:
            boxes_pd = predictions['boxes']
            scores = predictions['scores']

            if len(boxes_pd) > 0:
                for i in range(len(boxes_pd)):
                    bb = np.array(boxes_pd[i], dtype=np.float32) # TODO just specify int8 or imt16?
                    image_out = cv.rectangle(image_out,
                                            (int(bb[0]), int(bb[1])),
                                            (int(bb[2]), int(bb[3])),
                                            color=predictions_color,
                                            thickness=dt_box_thick)

                    # add text to top left corner of bbox
                    sc = format(scores[i] * 100.0, '.0f') # no decimals, just x100 for percent
                    cv.putText(image_out,
                               '{}: {}'.format(i, sc),
                               (int(bb[0] + 10), int(bb[1] + 30)), # buffer numbers should be function of font scale
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
                        iou_str = format(iou[i], '.2f') # max 2 decimal places
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
                # choose colour scheme default: blue is groundtruth default: red
                # is detection -> red is false negative green is true positive
                # yellow is false positive
                outcome_color = [(0, 255, 0),   # TP - green
                                (255, 255, 0), # FP - yellow
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
                        sc = format(scores[i] * 100.0, '.0f') # no decimals, just x100 for percent
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
                        if fn_gt[j]: # if True
                            bb = np.array(boxes_gt[j,:].cpu(), dtype=np.float32)
                            image_out = cv.rectangle(image_out,
                                                (int(bb[0]), int(bb[1])),
                                                (int(bb[2]), int(bb[3])),
                                                color=outcome_color[2],
                                                thickness=out_box_thick)
                            cv.putText(image_out,
                                '{}: {}'.format(j, outcome_list[2]),
                                (int(bb[0]+ 10), int(bb[1]) + 30),
                                fontFace=cv.FONT_HERSHEY_COMPLEX,
                                fontScale=font_scale,
                                color=outcome_color[2], # index for FN
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

    def infer_image(self,
                    image,
                    sample=None,
                    imshow=True,
                    imsave=False,
                    image_name=None,
                    conf_thresh=0.5,
                    iou_thresh=0.5):
        """ do inference on a single image """
        # assume image comes in as a tensor for now (eg, from image, sample in
        # dataset)

        with torch.no_grad():
            self._model.to(self._device)
            image = image.to(self._device)

            self._model.eval()

            # TODO accept different types of image input (tensor, numpy array,
            # PIL, filename?)

            if image_name is None:
                image_name = self._model_name + '_image'
            # start_time = time.time()
            pred = self.get_predictions_image(image, conf_thresh, iou_thresh)
            # end_time = time.time()
            # time_infer = end_time - start_time
            # print(f'time to infer: {time_infer}')


            if imsave or imshow:
                image_out = self.show_mask(image,
                                    sample=sample,
                                    predictions=pred)
            if imsave:
                save_folder = os.path.join('output', self._model_folder)
                os.makedirs(save_folder, exist_ok=True)
                save_image_name = os.path.join(save_folder, image_name + '.png')
                image_out_bgr = cv.cvtColor(image_out, cv.COLOR_RGB2BGR)
                cv.imwrite(save_image_name, image_out_bgr)

            if imshow:
                self.cv_imshow(image_out, win_name=str(image_name))

        return image_out, pred


    def infer_dataset(self,
                      dataset,
                      conf_thresh=0.5,
                      iou_thresh=0.5,
                      save_folder=None,
                      save_subfolder='infer_dataset',
                      imshow=False,
                      imsave=False,
                      wait_time=1000,
                      image_name_suffix=None):
        """ do inference on entire dataset """
        with torch.no_grad():
            self._model.to(self._device)
            self._model.eval()

            # out = []
            predictions = []

            if save_folder is None:
                save_folder = os.path.join('output', self._model_folder, save_subfolder)

            if imsave:
                os.makedirs(save_folder, exist_ok=True)

            print('number of images to infer: {}'.format(len(dataset)))

            for image, sample in dataset:
                image_id = sample['image_id'].item()
                image_name = dataset.annotations[image_id]['filename'][:-4]

                pred = self.get_predictions_image(image,
                                                conf_thresh,
                                                iou_thresh)
                image_out = self.show(image, sample=sample, predictions=pred)

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
                    self.cv_imshow(image_out,image_name, wait_time=wait_time)

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
                    iou_thresh=0.5):
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
        print('resized video resolution: width={}, height={}'.format(self._image_width, self._image_height))

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

        tform_rsc = Rescale(self._hp['rescale_size'])

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
                pred = self.get_predictions_image(frame, conf_thresh, iou_thresh)
                frame_out = self.show(frame, predictions=pred)

                # write image to video
                frame_out = cv.cvtColor(frame_out, cv.COLOR_RGB2BGR)
                video_out.write(frame_out)

                if vidshow:
                    self.cv_imshow(frame_out, 'video', wait_key=0, close_window=False)
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
                    cost[i,j] = self.compute_match_cost(dt_scores[i], gt_iou_all[i, j])
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
                            (gt_iou_all[i,j] >= DECISION_IOU_THRESH):
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
        fp = np.logical_or(np.logical_not(dt_match) , \
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
            pred = self.threshold_predictions(pred, DECISION_CONF_THRESH)

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

                save_img_name = os.path.join(save_path, img_name + '_outcome.png')
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
            ax.plot(rec, prec, marker='o', linestyle='dashed', label='original')
            ax.plot(rec, prec_new, marker='x', color='red', linestyle='solid', label='max-binned')
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.title('prec-rec, max-binned')
            ax.legend()

            os.makedirs(os.path.join('output', self._model_folder), exist_ok=True)
            save_plot_name = os.path.join('output', self._model_folder, save_name + '_test_pr_smooth.png')
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
            ax.plot(rec_temp, prec_temp, color='blue', linestyle='dashed', label='combined')
            ax.plot(rec, prec_new, marker='x', color='red', linestyle='solid', label='max-binned')
            ax.plot(rec_x, prec_x, color='green', linestyle='dotted', label='interp')
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.title('prec-rec, interpolated')
            ax.legend()

            os.makedirs(os.path.join('output', self._model_folder), exist_ok=True)
            save_plot_name = os.path.join('output', self._model_folder, save_name + '_test_pr_interp.png')
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
        ap = np.sum( (r[1:n] - r[0:n-1]) * p[1:n] )
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
                    PLOT=False):
        """ get complete/smoothed pr curve for entire dataset """
        # infer on 0-decision threshold iterate over diff thresholds extend_pr
        # compute_ap save output

        # infer on dataset with 0-decision threshold
        predictions = self.infer_dataset(dataset,
                                        conf_thresh=0,
                                        iou_thresh=nms_iou_thresh,
                                        save_folder=save_folder,
                                        save_subfolder=os.path.join('prcurve', 'detections'),
                                        imshow=imshow,
                                        imsave=imsave,
                                        image_name_suffix='_prcurve_0')



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
        save_plot_name = os.path.join(save_folder, self._model_name + '_pr_raw.png')
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

        p_final, r_final, c_final, = self.extend_pr(prec, rec, confidence_thresh)
        ap = self.compute_ap(p_final, r_final)

        # plot final pr curve
        fig, ax = plt.subplots()
        ax.plot(r_final, p_final)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title('prec-rec curve, iou={}, ap = {:.2f}'.format(decision_iou_thresh, ap))
        # ax.legend()
        save_plot_name = os.path.join(save_folder, self._model_name + '_pr.png')
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
        save_file = os.path.join(save_folder, self._model_name + '_prcurve.pkl')
        with open(save_file, 'wb') as f:
            pickle.dump(res, f)

        return res

# =========================================================================== #

if __name__ == "__main__":

      # two models
    print('WeedModel.py')