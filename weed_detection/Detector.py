#! /usr/bin/env/python3

"""
- load model
- export model (both raw and filtered predictions)
- threshold predictions
- plot image detections
- simplify polygon, confidence mask binarize
- centroids
- do inference on image/dataset/video
- 

- be imported as a detector class (see James Bishop's API)
"""

import os
import json
from shutil import ReadError

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image as PIL_Image
import cv2 as cv
from typing import Tuple
from shapely.geometry import Polygon
from scipy.interpolate import interp1d

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# from Detections import Detections
from MaskDetections import MaskDetections


class Detector:

    # parameters
    IMAGE_INPUT_SIZE_DEFAULT = (1028, 1232) # TODO should make a config file
    SPECIES_FILE_DEFAULT = os.path.join(os.getcwd(), 'model/maskrcnn_species_names.txt')
    MODEL_FILE_DEFAULT = os.path.join('model/XX.pth')
    CONFIDENCE_THRESHOLD_DEFAULT = 0.5
    NMS_THRESHOLD_DEFAULT = 0.5
    MASK_THRESHOLD_DEFAULT = 0.5

    

    def __init__(self, 
                 model_file: str = MODEL_FILE_DEFAULT,
                 names_file: str = SPECIES_FILE_DEFAULT,
                 image_input_size: Tuple[int, int] = IMAGE_INPUT_SIZE_DEFAULT,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD_DEFAULT,
                 nms_threshold: float = NMS_THRESHOLD_DEFAULT,
                 mask_threshold: float = MASK_THRESHOLD_DEFAULT):

        self.model = self.load_model(model_file)
        self.species = self.load_class_names(class_names_file=names_file)
        self.image_width = image_input_size[0]
        self.image_height = image_input_size[1]
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.mask_threshold = mask_threshold

        # to store detections in
        self.detections = []

    def load_class_names(self, class_names_file: str):
        class_names = []

        with open(class_names_file) as file:
            for class_name in file:
                class_names.append(class_name.strip())

        if len(class_names) > 0:
            return class_names
        else:
            raise ReadError(f'No class names read from class_names_file: {class_names_file}')
            return False


    def build_model(self, 
                    num_classes: int = 2,
                    pre_trained: bool = True):
        """ build maskrcnn model for set number of classes (num_classes)
        loads pre-trained model on coco image database
        """
        # load instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pre_trained)

        # get number of input features from the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # replace pretrained head with new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # get number of input features for mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256 # TODO check what this variable is

        # replace mask predictor with new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        return model


    def load_model(self, 
                   model_file: str,
                   num_classes: int = 2,
                   map_location: str = 'cuda:0',
                   pre_trained: bool = True):
        """ load model based on model file (absolute file path) """
        model = self.build_model(num_classes, pre_trained)
        model.load_state_dict(torch.load(os.path.abspath(model_file), map_location=map_location))
        print(f'loaded model: {model_file}')
        model.to(self.device)
        return model

    def convert_input_image(self, image, image_color_format = 'RGB'):
        """ convert image input type numpy array/PIL Image/tensor to model required format
        which is pytorch image tensor, RGB
        """
        color_format = ['RGB', 'BGR']
        # TODO also, rescale image to correct input image size?

        if isinstance(image, np.array):
            # check color format/ordering of image, convert to RGB
            if image_color_format == color_format[1]:
                image = image[:, :, [2, 1, 0]]

            # check size of image, rescale to appropriate size
            hi, wi, ci = image.shape
            if (hi != self.image_height) or (wi != self.image_width):
                image = cv.resize(image, 
                                  (self.image_width, self.image_height),
                                  interpolation=cv.INTER_NEAREST)
            # transform to tensor
            image = torch.from_numpy(image)
        elif isinstance(image, PIL_Image.Image):
            # resize image to fit
            image = PIL_Image.resize((self.image_width, self.image_height))
            pil_transform = transforms.PILToTensor()
            image = pil_transform(image)
        else:
            transform = transforms.ToTensor()
            image = transform(image)
            resize = transforms.Resize()
            image = resize(image, (self.image_width, self.image_height))
        return image


    def detect(self, 
               image, 
               confidence_threshold = False, 
               nms_threshold = False,
               mask_threshold = False):
        """ infer model predictions from single image (object detection) """

        if not confidence_threshold:
            confidence_threshold = self.confidence_threshold
        if not nms_threshold:
            nms_threshold = self.nms_threshold
        if not mask_threshold:
            mask_threshold = self.mask_threshold

        # convert image to valid image tensor
        image = self.convert_input_image(image)

        with torch.no_grad():
            self.model.to(self.device)
            image.to(self.device)

            self.model.eval()

            detections_raw = self.model([image])

        # apply non-maxima suppresion to raw detections
        keep = torchvision.ops.nms(detections_raw[0]['boxes'],
                                    detections_raw[0]['scores'],
                                    nms_threshold)
        
        # NOTE potential optimisation to leave on GPU            
        detections_class = [i for i in list(detections_raw[0]['labels'][keep].cpu().numpy())]
        # bboxes in the form [xmin, ymin, xmax, ymax]
        detections_boxes = [[bb[0], bb[1], bb[2], bb[3]]
                            for bb in list(detections_raw[0]['boxes'][keep].detach().cpu().numpy())]
        detections_masks = list(detections_raw[0]['masks'][keep].detach().cpu().numpy())
        # scores are ordered from highest to lowest
        detections_scores = list(detections_raw[0]['scores'][keep].detach().cpu().numpy())

        # create Detections object for each detection
        detections = []
        for i, mask in enumerate(detections_masks):
            mask = np.transpose(mask, (1, 2, 0))
            
            # create mask detection object, polygon, centroid information, etc 
            # should be automatically populated
            maskdetections = MaskDetections(label = detections_class[i],
                                            score = detections_scores[i],
                                            mask = mask,
                                            mask_threshold = mask_threshold)
            detections.append(maskdetections)

        # TODO calibrate model confidence scores!

        # apply confidence threshold to detection scores
        detections = [d for d in detections if detections.score >= confidence_threshold]

        # return detections
        return detections
        

    # @abstractmethod
    # def run(self, image):
    #     raise NotImplementedError
        # must return 'classes', 'scores', 'boxes'
        # classes: a list of class labels (int)
        # scores: a list of confidence scores
        # boxes: a list of coords (format?)
    def run(self, image):

        detections = self.detect(image)

        classes = [d[i].label for d, i in enumerate(detections)]
        scores = [d[i].score for d, i in enumerate(detections)]
        boxes = [d[i].box for d, i in enumerate(detections)]

        return classes, scores, boxes

