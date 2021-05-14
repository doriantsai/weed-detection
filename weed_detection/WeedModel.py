#! /usr/bin/env python

"""
weed model class for weed detection
class to package model-related functionality, such as
training, inference, evaluation
"""

import os
import torch
import torchvision

import time
import datetime
import pickle
import numpy as np
import cv2 as cv

# TODO replace tensorboard with weightsandbiases
from torch.utils.tensorboard import SummaryWriter
from engine_st import train_one_epoch, evaluate
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from weed_detection.WeedDataset import WeedDataset
from weed_detection.PreProcessingToolbox import PreProcessingToolbox
from webcam import grab_webcam_image


class WeedModel:
    """ collection of functions for model's weed detection """

    def __init__(self, 
                 weed_name='serrated tussock', 
                 model=None, 
                 model_name=None,
                 model_path=None,
                 device=None,
                 hyper_parameters=None):

        self.weed_name = weed_name
        # TODO maybe save model type/architecture
        # also, hyper parameters?
        self.model = model
        self.model_name = model_name
        self.model_path = model_path

        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device

        self.hp = hyper_parameters
        # TODO consider expanding hp from dictionary into actual properties/attributes
        # for more readability
        self.image_width = 2464
        self.image_height = 2056



    # getters and setters
    def set_model(self, model):
        self.model = model


    def get_model(self):
         return self.model


    def set_model_name(self, name):
        self.model_name = name

    
    def get_model_name(self):
        return self.model_name

    
    def set_weed_name(self, name):
        self.weed_name = name

    
    def get_weed_name(self):
        return self.weed_name


    def set_model_path(self, path):
        self.model_path = path


    def get_model_path(self):
        return self.model_path

    
    def set_hyper_parameters(self, hp):
        self.hp = hp
    

    def get_hyper_parameters(self):
        return self.hp


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
              model_name_suffix=True):

        # TODO if dataset_path is None, call create_train_test_val_datasets
        # for now, we assume this has been done/dataset_path exists and is valid
        if dataset_path is None:
            print('TODO: call function to build dataset objects and return them')
        # else:
        
        # loading dataset, full path
        print('loading dataset:' + dataset_path)
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
        save_folder = os.path.join('output', model_name)
        os.makedirs(save_folder, exist_ok=True)
        print('Model saved in folder: {}'.format(save_folder))

        # setup device, send to gpu if possible, otherwise cpu
        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # shifted into object properties and init

        # build model
        # setup number of classes (1 background, 1 class - weed species)
        model = self.build_model(num_classes=2)
        model.to(self.device)

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
        # train for-loop for epochs
        # NOTE we do not do explicit early stopping. We run for a set number of
        # epochs and then choose the appropriate "stopping" point later from 
        # the snapshots. This is to clearly identify that in fact, we have 
        # reached a low-point in the validation loss.
        start_time = time.time()
        print('start training')
        for epoch in range(hp_train['num_epochs']):
            # modified from coco_api tools to take in separate training and
            # validation dataloaders, as well as port the images to device
            mt, mv = train_one_epoch(model,
                                     optimizer,
                                     dl_train,
                                     dl_val,
                                     self.device,
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
        self.model = model
        self.model_name = model_name
        self.model_path = model_save_path

        return model, model_save_path


    def get_predictions_image(self, 
                              model, 
                              image, 
                              conf_thresh, 
                              nms_iou_thresh):
        """ take in model, single image, thresholds, return bbox predictions for scores > threshold """

        # image incoming is a tensor, since it is from a dataloader object
        model.eval()  # TODO could call self.model.eval(), but for now, just want to port the scripts/functions
        if torch.cuda.is_available():
            image.to(self.device)
            model.to(self.device) # added, unsure if this will cause errors
        
        # do model inference on single image
        pred = model([image])

        # apply non-maxima suppression
        keep = torchvision.ops.nms(pred[0]['boxes'], pred[0]['scores'], nms_iou_thresh)
        
        pred_class = [i for i in list(pred[0]['labels'][keep].cpu().numpy())]
        pred_boxes = [[bb[0], bb[1], bb[2], bb[3]] for bb in list(pred[0]['boxes'][keep].detach().cpu().numpy())]
        # scores are ordered from highest to lowest
        pred_score = list(pred[0]['scores'][keep].detach().cpu().numpy())

        # package
        pred_final = {}
        pred_final['boxes'] = pred_boxes
        pred_final['classes'] = pred_class
        pred_final['scores'] = pred_score

        # apply confidence threshold
        pred_final = self.threshold_predictions(pred_final, conf_thresh)

        return pred_final

    
    def threshold_predictions(self, pred, thresh):
        """ apply confidence threshold to predictions """

        pred_boxes = pred['boxes']
        pred_class = pred['classes']
        pred_score = pred['scores']

        if len(pred_score) > 0:
            if max(pred_score) < thresh: # none of pred_score > thresh, then return empty
                pred_thresh = []
                pred_boxes = []
                pred_class = []
                pred_score = []
            else:
                pred_thresh = [pred_score.index(x) for x in pred_score if x > thresh][-1]
                pred_boxes = pred_boxes[:pred_thresh+1]
                pred_class = pred_class[:pred_thresh+1]
                pred_score = pred_score[:pred_thresh+1]
        else:
            pred_thresh = []
            pred_boxes = []
            pred_class = []
            pred_score = []

        predictions = {}
        predictions['boxes'] = pred_boxes
        predictions['classes'] = pred_class
        predictions['scores'] = pred_score

        return predictions


    def cv_imshow(self, image, win_name, wait_time=2000):
        """ show image with win_name for wait_time """
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.namedWindow(win_name, cv.WINDOW_GUI_NORMAL)
        cv.imshow(win_name, img)
        cv.waitKey(wait_time)
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
             transpose_color_channels=False):
        """ show image, sample/groundtruth, model predictions, outcomes (TP/FP/etc) """
        # assume image comes in as a tensor, as in the same format it was input 
        # into the model

        # set plotting parameters
        gt_box_thick = 12   # groundtruth bounding box
        dt_box_thick = 6    # detection bounding box
        out_box_thick = 3   # outcome bounding box/overlay
        font_scale = 1
        font_thick = 2

        if transpose_color_channels:
            # image tensor comes in as [color channels, length, width] format
            print('swap color channels in tensor format')
            image = image[(2, 0, 1), :, :]

        # move to cpu and convert from tensor to numpy array
        # since opencv requires numpy arrays
        image_np = image.cpu().numpy()

        if transpose_image_channels:
            # if we were working with BGR as opposed to RGB
            image_np = np.transpose(image_np, (1, 2, 0))
        
        # normalize image from 0,1 to 0,255
        image_np = cv.normalize(image_np, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

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
                    image_np = cv.rectangle(image_np,
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
                    image_np = cv.rectangle(image_np,
                                            (int(bb[0]), int(bb[1])),
                                            (int(bb[2]), int(bb[3])),
                                            color=predictions_color,
                                            thickness=dt_box_thick)
                    
                    # add text to top left corner of bbox
                    sc = format(scores[i] * 100.0, '.0f') # no decimals, just x100 for percent
                    cv.putText(image_np,
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
                        # print in top/left corner of bbox underneath bbox # and score
                        iou_str = format(iou[i], '.2f') # max 2 decimal places
                        cv.putText(image_np,
                                   'iou: {}'.format(iou_str),
                                   (int(bb[0] + 10), int(bb[1] + 60)),
                                   fontFace=cv.FONT_HERSHEY_COMPLEX,
                                   fontScale=font_scale,
                                   color=iou_color,
                                   thickness=font_thick)

        # ----------------------------------- #
        # fourth, add outcomes
            if (outcomes is not None) and (sample is not None):
                # for each prediction, if there is a sample, then there is a known outcome
                # being an array from 1-4:
                outcome_list = ['TP', 'FP', 'FN', 'TN']
                # choose colour scheme
                # default: blue is groundtruth
                # default: red is detection ->
                #          red is false negative
                #          green is true positive
                #          yellow is false positive
                outcome_color = [(0, 255, 0),   # TP - green
                                (255, 255, 0), # FP - yellow
                                (255, 0, 0),   # FN - red
                                (0, 0, 0)]     # TN - black
                # structure of the outcomes dictionary
                # outcomes = {'dt_outcome': dt_outcome, # detections, integer index for tp/fp/fn
                # 'gt_outcome': gt_outcome, # groundtruths, integer index for fn
                # 'dt_match': dt_match, # detections, boolean matched or not
                # 'gt_match': gt_match, # gt, boolean matched or not
                # 'fn_gt': fn_gt, # boolean for gt false negatives
                # 'fn_dt': fn_dt, # boolean for dt false negatives
                # 'tp': tp, # true positives for detections
                # 'fp': fp, # false positives for detections
                # 'dt_iou': dt_iou} # intesection over union scores for detections
                dt_outcome = outcomes['dt_outcome']
                if len(dt_outcome) > 0 and len(boxes_pd) > 0:
                    for i in range(len(boxes_pd)):
                        # replot detection boxes based on outcome
                        bb = np.array(boxes_pd[i], dtype=np.float32)
                        image_np = cv.rectangle(image_np,
                                                (int(bb[0]), int(bb[1])),
                                                (int(bb[2]), int(bb[3])),
                                                color=outcome_color[dt_outcome[i]],
                                                thickness=out_box_thick)
                        # add text top/left corner including outcome type
                        # prints over existing text, so needs to be the same starting string
                        sc = format(scores[i] * 100.0) # no decimals, just x100 for percent
                        cv.putText(image_np,
                                   '{}: {}/{}'.format(i, sc, outcome_list[dt_outcome[i]]),
                                   fontFace=cv.FONT_HERSHEY_COMPLEX,
                                   fontScale=font_scale,
                                   color=outcome_color[dt_outcome[i]],
                                   thickness=font_thick)
                
                # handle false negative cases (ie, groundtruth bboxes)
                boxes_gt = sample['boxes']
                fn_gt = outcomes['fn_gt']
                if len(fn_gt) > 0 and len(boxes_gt) > 0:
                    for j in range(len(boxes_gt)):
                        # gt boxes already plotted, so only replot them if false negatives
                        if fn_gt[j]: # if True
                            bb = np.array(boxes_gt[j,:].cpu(), dtype=np.float32)
                            imgnp = cv.rectangle(imgnp,
                                                (int(bb[0]), int(bb[1])),
                                                (int(bb[2]), int(bb[3])),
                                                color=outcome_color[2],
                                                thickness=out_box_thick)
                            cv.putText(imgnp,
                                '{}: {}'.format(j, outcome_list[2]),
                                (int(bb[0]+ 10), int(bb[1]) + 30),
                                fontFace=cv.FONT_HERSHEY_COMPLEX,
                                fontScale=font_scale,
                                color=outcome_color[2], # index for FN
                                thickness=font_thick)

        return image_np


    def infer_image(self, 
                    model, 
                    image, 
                    sample=None, 
                    imshow=True, 
                    imsave=False, 
                    image_name=None,
                    conf_thresh=0.5,
                    iou_thresh=0.5):
        """ do inference on a single image """
        # assume image comes in as a tensor for now (eg, from image, sample in dataset)
        model.to(self.device)
        image.to(self.device)

        model.eval()
        
        # TODO accept different types of image input (tensor, numpy array, PIL, filename?)

        if image_name is None:
            image_name = self.model_name + '_image'
        pred = self.get_predictions_image(model, image, conf_thresh, iou_thresh)
        
        if imsave or imshow:
            image_out = self.show(image,
                                sample=sample,
                                predictions=pred)
        if imsave:
            save_folder = os.path.join('output', self.model_name)
            os.makedirs(save_folder, exist_ok=True)
            save_image_name = os.path.join(save_folder, image_name + '.png')
            image_out_bgr = cv.cvtColor(image_out, cv.COLOR_RGB2BGR)
            cv.imwrite(save_image_name, image_out_bgr)

        if imshow:
            self.cv_imshow(image_out, win_name=image_name)
        
        return image_out, pred

    
    def infer_dataset(self,
                      model,
                      dataset,
                      conf_thresh=0.5,
                      iou_thresh=0.5,
                      save_folder=None,
                      imshow=False,
                      imsave=False,
                      wait_time=1000):
        """ do inference on entire dataset """

        model.to(self.device)
        model.eval()

        # out = []
        predictions = []

        if save_folder is None:
            save_folder = os.path.join('output', self.model_name, 'infer_dataset')

        if imsave:
            os.makedirs(save_folder, exist_ok=True)

        print('number of images to infer: {}'.format(len(dataset)))

        for image, sample in dataset:
            image_id = sample['image_id'].item()
            image_name = dataset.dataset.annotations[image_id]['filename'][:-4]

            pred = self.get_predictions_image(model,
                                              image,
                                              conf_thresh,
                                              iou_thresh)
            image_out = self.show(image, sample=sample, predictions=pred)

            if imsave:
                save_image_name = os.path.join(save_folder, image_name + '.png')
                image_out_bgr = cv.cvtColor(image_out, cv.COLOR_RGB2BGR)
                cv.imwrite(save_image_name, image_out_bgr)

            if imshow:
                self.cv_imshow(image_out,image_name, wait_time=wait_time)

            # saving output
            # out_tensor = model()
            predictions.append(pred)

        return predictions

    
    def infer_video(self,
                    model,
                    capture=None,
                    fps=10,
                    video_out_name=None,
                    save_folder=None,
                    max_frames=1000,
                    vidshow=True):
        """ video inference from a webcam defined by capture (see opencv video capture object) """

        if capture is None:
            capture = cv.VideoCapture(0)
        
        # get width/height of orininal image
        w = capture.get(cv.CAP_PROP_FRAME_WIDTH)
        h = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
        print('original video capture resolution: width={}, height={}'.format(w, h))

        # images will get resized to what the model was trained for, so get the output video size
        hp = self.hp
        
        # TODO set webcam exposure settings
        if save_folder is None:
            save_folder = os.path.join('output', self.model_name, 'video')
            os.makedirs(save_folder, exist_ok=True)

        if video_out_name is None:
            now_str = self.get_now_str()
            video_out_name = self.model_name + now_str + '_video.avi'
        
        video_out_path = os.path.join(save_folder, video_out_name)

        # set video writer and encoder
        video_write = cv.VideoWriter_fourcc(*'XVID')
        video_out = cv.VideoWriter(video_out_path,
                                   fourcc=video_write,
                                   fps=fps,
                                   frameSize=(int(self.image_width), int(self.image_height)))

        # TODO continue infer_video - rescale  + while loop

    # TODO inference_video
    # TODO prcurve
    # TODO model_compare
    