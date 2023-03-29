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
from sklearn.metrics import f1_score
import torch
import torchvision
import re
import time
import datetime
import pickle
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob


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
                 weed_name='serrated tussock', # TODO update for multiclass
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


    def get_now_str(self):
        """ return a string of yyyymmdd_hh_mm to provide unique string in
        filenames or model names"""
        now = str(datetime.datetime.now())
        now_str = now[0:10] + '_' + now[11:13] + '_' + now[14:16]
        return now_str


    def load_model(self,
                   model_path=None,
                   num_classes=2,
                   map_location="cuda:0",
                   annotation_type='poly'):
        """ load model to self based on model_path

        model_path - absolute string path to .pth file of trained model
        num_classes - number of classes + background class (eg, for a single-class detector, num_classes = 2)
        map_location - where the model should be loaded (default directly onto gpu)
        annotation_type - string, type of annotation (poly/box)
        """

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
        """ set snapshot for epoch, deals with early stopping

        As an alternative to early stopping, where we may not be 100% sure that
        early stopping requirements have reached, we instead train to X number
        of epochs, and at a specified interval, save a snapshot of the model. We
        observe the performance of each model (via training and validation
        loss), and can choose where the We can select the snapshot and replace the "final
        model" at a later time via this function, set_snapshot()

        input:
        epoch - the desired epoch number
        snapshot_folder - where snapshots are located
        output:
        True if found relevant snapshot and loaded model; otherwise, False

        """
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

        try:
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
        except:
            print('Failed to set snapshot and load model')
            return False


    def find_file(self, file_pattern, folder):
        """
        helper function to find filename given file pattern in a folder
        """

        if not isinstance(file_pattern, str):
            raise TypeError(file_pattern, 'file_pattern must be of a str')
        if not isinstance(folder, str):
            raise TypeError(folder, 'folder must be a str')

        # TODO check if folder is valid directory

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
        scores > threshold

        image - image tensor
        conf_thresh - confidence threshold scalar (0-1)
        nms_iou_thresh - non-maxima suppression interval-over-union threshold (0-1)
        annotation_type - polygon or box (box/poly string)
        mask_threshold - threshold to binarize mask output (which comes as 0-1 mapping)
        """

        # TODO check inputs
        # TODO update for multiple classes

        # image incoming is a tensor, since it is from a dataloader object
        self._model.eval()

        if torch.cuda.is_available():
            image = image.to(self._device)
            # added, unsure if this will cause errors
            self._model.to(self._device)

        # do model inference on single image
        # TODO self._model(image) if image is
        # TODO a list of tensors, should handle in batch, could be much faster for
        # TODO large quantity of images start_time = time.time()
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
        """ apply confidence threshold to predictions, returns predictions in
        same dictionary """

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

            # Use original if open closing operation destroys mask (very small masks)
            if np.amax(mask_close) == 0:
                mask_close = mask_bin

            # minor erosion then dilation by the same amount

            # find bounding polygon of binary image
            # convert images to cv_8u
            contours_in, hierarchy = cv.findContours(
                mask_close, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            contours = list(contours_in)

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
        # TODO redundant with show() (which is limited to bounding boxes at the moment), should refactor to combine the two

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
        # first plot groundtruth
        if sample is not None:
            # NOTE we assume sample is also a tensor
            # plot groundtruth bounding boxes & label
            boxes_gt = sample['boxes']
            class_gt = sample['labels']
            # assume that if there is a box, there is a corresponding label
            # assert len(boxes_gt) == len(class_gt), "num groundtruth boxes != num labels"
            if len(boxes_gt) != len(class_gt):
                print("num groundtruth boxes != num labels")
                import code
                code.interact(local=dict(globals(), **locals()))
                
            if len(boxes_gt) > 0:
                n_gt, _ = boxes_gt.size()
                for i in range(n_gt):
                    lbl_gt = format(class_gt[i], '.0f')
                    # TODO just specify int8 or imt16?
                    bb = np.array(boxes_gt[i, :].cpu(), dtype=np.float32)
                   # overwrite the original image with groundtruth boxes
                    image_out = cv.rectangle(image_out,
                                             (int(bb[0]), int(bb[1])),
                                             (int(bb[2]), int(bb[3])),
                                             color=sample_color,
                                             thickness=gt_box_thick)
                    # plot gt label
                    cv.putText(image_out,
                               '{}'.format(lbl_gt),
                               (int(bb[0] + 10), int(bb[1] + 30)),
                               fontFace=cv.FONT_HERSHEY_COMPLEX,
                               fontScale=font_scale,
                               color=sample_color,
                               thickness=font_thick)                       

            # plot groundtruth spraypoint
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

            # plot groundtruth bounding contour
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
                    annotation_type='poly',
                    image_color_format='RGB'):
        """ do inference on a single image """
        # assume image comes in as a tensor for now (eg, from image, sample in
        # dataset)

        if not self.check_image(image):
            print(f'image type: {type(image)}')
            raise TypeError(image, 'image must be numpy array or pytorch tensor')

        if isinstance(image, np.ndarray):
            # adjust BGR to RGB
            if image_color_format == 'BGR':
                image = image[:, :, [2, 1, 0]] # given BGR(012), want RGB(210)

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

    


# =========================================================================== #


if __name__ == "__main__":

    # two models
    print('WeedModel.py')
