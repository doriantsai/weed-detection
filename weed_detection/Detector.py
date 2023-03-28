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
# import json
from shutil import ReadError
from subprocess import call

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image as PIL_Image
import cv2 as cv
from typing import Tuple
# from shapely.geometry import Polygon
# from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

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

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model = self.load_model(model_file)
        self.species = self.load_class_names(class_names_file=names_file)
        self.image_width = image_input_size[0]
        self.image_height = image_input_size[1]
        

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


    @staticmethod
    def download_model(url, model_file, model_dir = 'model'):
        """ download model """
        move_model_file = os.path.join(model_dir, model_file)
        if not os.path.exists(move_model_file):
            os.makedirs(model_dir, exist_ok=True)
            call(['wget', '-O', model_file, url])   
            
            os.rename(model_file, move_model_file)
            print(f'model downloaded to {move_model_file}')

        else:
            print('Model_file already exists. Model already downloaded.')
            print(model_file)
        
        return True


    def build_model(self, 
                    num_classes: int = 2,
                    pre_trained: bool = True):
        """ build maskrcnn model for set number of classes (num_classes)
        loads pre-trained model on coco image database
        """
        # load instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='MaskRCNN_ResNet50_FPN_Weights.DEFAULT')

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

        if isinstance(image, np.ndarray):
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
            # image = torch.from_numpy(image)
            transform = transforms.ToTensor()
            image = transform(image)
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
            image = image.to(self.device)

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
                                            mask_confidence= mask,
                                            mask_threshold = mask_threshold)
            detections.append(maskdetections)

        # TODO calibrate model confidence scores!

        # apply confidence threshold to detection scores
        detections = [d for d in detections if d.score >= confidence_threshold]

        # return detections
        return detections


    def load_image_from_string(self, image_filename: str):
        """show_image
        Given an image filename, output the image as a numpy array 
        """
        img = cv.imread(image_filename) # openCV default reads in BGR format
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB) # however, the rest of the world works with RGB
        return cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U) # just in case, normalize image from 0,1 to 0,255


    def show_detections(self, image, detections, save_image_filename, SAVE=False):
        """show_detections
        Given an RGB image (numpy array), output a new image with detections drawn ontop of the input image

        Args:
            image (_type_): _description_

        Returns:
            _type_: _description_
        """
        # TODO assert valid image, numpy array, format, number of channels, size, etc
        

        # get detections
        # detections = self.detect(image)

        # draw on detetections - just bounding boxes
        # plotting parameters:
        box_thick = 3
        font_scale = 1 # TODO font scale should be function of image size
        font_thick = 1
        detection_colour = [255, 0, 0] # RGB red - TODO should setup a library of colours for multiple classes

        for d in detections:
            bb = np.array(d.box, dtype=np.float32)
            # draw box
            image = cv.rectangle(image, 
                                 (int(bb[0]), int(bb[1])),
                                 (int(bb[2]), int(bb[3])),
                                 color=detection_colour,
                                 thickness=box_thick)
            
            # add text to top left corner of box
            # class + confidence as a percent
            conf_str = format(d.score * 100.0, '.0f')
            cv.putText(image,
                       '{}: {}'.format(d.class_name, conf_str),
                       (int(bb[0] + 10), int(bb[1]) + 30),
                       fontFace=cv.FONT_HERSHEY_COMPLEX,
                       fontScale=font_scale,
                       color=detection_colour,
                       thickness=font_thick)
        
        if SAVE:
            self.save_image(image, save_image_filename)
        return image
    

    def save_image(self, image, image_filename):
        """ save_image
        write image to file, given image and image_filename
        """
        # TODO ensure that image_filename is valid and folder exists
        image_dir = os.path.dirname(image_filename)
        os.makedirs(image_dir, exist_ok=True)
        # TODO ensure that image is valid
        # assuming image is in RGB format, so convert back to BGR
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.imwrite(image_filename, image)
        return True


    # @abstractmethod
    # def run(self, image):
    #     raise NotImplementedError
        # must return 'classes', 'scores', 'boxes'
        # classes: a list of class labels (int)
        # scores: a list of confidence scores
        # boxes: a list of coords [xmin, ymin, xmax, ymax]
    def run(self, image): # TODO add parameters/options

        detections = self.detect(image)

        classes = [d.label for d in detections]
        scores = [d.score for d in detections]
        boxes = [d.box for d in detections]

        return classes, scores, boxes


if __name__ == "__main__":

    print('Detector.py')
    # set LOCAL to True if using a local file model, or False to automatically download a pre-existing model
    # NOTE pre-existing/trained model link may need to be updated for most recent data
    LOCAL = True
    
    # load a model
    # infer on a given image
    # print out the classes, scores, boxes (TODO plots later)
    # model_file = 'model/2021_Yellangelo_Tussock_v0_2022-10-12_10_35.pth'
    
    if LOCAL:
        model_file = '/home/agkelpie/Code/agkelpie_weed_detection/weed-detection/model/model_best.pth'
    else: # REMOTE
        model_file = '2021_Yellangelo_Tussock_v0_2022-10-12_10_35_epoch30.pth'
        url = 'https://cloudstor.aarnet.edu.au/plus/s/3XEnfIEoLEAP27o/download'
        Detector.download_model(url, model_file)
    detector = Detector(model_file = os.path.join('model', model_file)) # might have to retrain with correct image size

    # grab any images in the 'images' folder:
    # img_dir = 'images'
    img_dir = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_yellangelo_tussock/annotated_images'
    out_dir = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_yellangelo_tussock/detections'
    img_list = os.listdir(img_dir)
    # image_file = 'images/2021-10-13-T_13_50_55_743.png'
    max_image = 5
    for i, img_name in enumerate(img_list):
        if i > max_image:
            print(f'hit max images {i}')
            break
        print(f'{i}: image name = {img_name}')
        img = detector.load_image_from_string(os.path.join(img_dir, img_name))
        detections = detector.detect(img)
        save_img_name = os.path.join(out_dir, img_name[:-4] + '_det.png')
        img_out = detector.show_detections(img, detections, save_img_name, True)
        # use matplotlib to show image, since their image viewer is much more stable and user-friendly
        # fig, ax = plt.subplot()
        # plt.imshow(img_out)
        # plt.show()
        # image = cv.imread(os.path.join(img_dir, img_name))

        # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # classes, scores, boxes = detector.run(image)
        # print('classes:')
        # print(classes)
        # print('scores: ')
        # print(scores)
        # print('boxes: ')
        # print(boxes)