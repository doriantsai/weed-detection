#! /usr/bin/env/python3

import os
from shutil import ReadError
from subprocess import call
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image as PIL_Image
import cv2 as cv
from typing import Tuple
# import matplotlib.pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# from Detections import Detections
from weed_detection.MaskDetections import MaskDetections


""" Detector.py

Detector is a class to hold functions associated with loading the trained model,
performing detections (i.e. model inference), and plotting these detections on
the original image. See main for example usage.

As for integration with UNE detection pipeline, UNE's code should import this
Detector class: https://github.com/AgKelpie/une_weed_package
"""
class Detector:

    # parameters
    IMAGE_INPUT_SIZE_DEFAULT = (1028, 1232) # TODO should make a config file
    SPECIES_FILE_DEFAULT = os.path.join(os.getcwd(), 'model/names_yellangelo32.txt')
    MODEL_FILE_DEFAULT = os.path.join('model/Yellangelo32/model_best.pth')
    CONFIDENCE_THRESHOLD_DEFAULT = 0.5
    NMS_THRESHOLD_DEFAULT = 0.5
    MASK_THRESHOLD_DEFAULT = 0.5

    ORIGINAL_IMAGE_INPUT_SIZE_DEFAULT = (2052, 2460)


    def __init__(self, 
                 model_file: str = MODEL_FILE_DEFAULT,
                 names_file: str = SPECIES_FILE_DEFAULT,
                 image_input_size: Tuple[int, int] = IMAGE_INPUT_SIZE_DEFAULT,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD_DEFAULT,
                 nms_threshold: float = NMS_THRESHOLD_DEFAULT,
                 mask_threshold: float = MASK_THRESHOLD_DEFAULT,
                 original_image_input_size: Tuple[int, int]= ORIGINAL_IMAGE_INPUT_SIZE_DEFAULT):

        # device to run computation on (GPU or CPU)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # define list of available class names for detector, because otherwise
        # how can a detector know what classes it is finding (it only returns
        # numerical labels) 
        # must come before load_model, because load_model needs num_classes
        self.class_names = self.load_class_names(class_names_file=names_file)
        self.class_names.insert(0, 'background') # insert background at front of list

        self.num_classes = len(self.class_names)   #  +1 for background class "0"

        # load the maskrcnn model
        self.model = self.load_model(model_file)
        
        # load rescaled image input that the model was trained for due to computation reasons
        self.image_width = image_input_size[0]
        self.image_height = image_input_size[1]

        # confidence threshold is a value that is assigned by the model during
        # runtime used to determine the likelihood of an object being present in
        # an image. Specifically, it is the minimum prediction score that an
        # object detection algorithm must assign to a proposed object before
        # considering it a valid detection.
        self.confidence_threshold = confidence_threshold

        # non-maxima suppression threshold 
        # Non-maximum suppression (NMS) is a technique used in object detection
        # to remove redundant or overlapping bounding boxes that correspond to
        # the same object instance. The NMS algorithm selects the bounding box
        # with the highest confidence score, also known as the maximum score,
        # and suppresses all other boxes that overlap with it by a certain
        # threshold.
        
        # The NMS threshold is a hyperparameter that determines how much overlap is
        # allowed between bounding boxes before they are considered redundant and
        # suppressed. Specifically, it is the minimum intersection-over-union (IoU)
        # value that two bounding boxes need to have to be considered overlapping. The
        # IoU is the ratio between the intersection and union areas of two bounding
        # boxes. A higher NMS threshold will result in more aggressive suppression,
        # leading to fewer but more accurate detections, while a lower threshold will
        # produce more detections but may also lead to more false positives. The optimal
        # NMS threshold depends on the specific task and dataset and often requires
        # tuning through cross-validation or other techniques.

        # In practice, the NMS algorithm is typically applied after the object detection
        # model has generated a list of candidate bounding boxes and corresponding
        # confidence scores. The algorithm sorts the boxes by their confidence scores and
        # iteratively selects the box with the highest score and removes all other boxes
        # with an IoU greater than the NMS threshold. The process is repeated until no
        # more boxes remain or until a maximum number of boxes is reached.
        self.nms_threshold = nms_threshold

        # mask threshold is the threshold to determine the binary mask 0s and
        # 1s, given a confidence masks that has elements ranging from 0 to 1. A
        # lower mask_threshold means larger masks, where the edges of the mask
        # are less confident of being a part of the object
        self.mask_threshold = mask_threshold

        # to store detections in
        self.detections = []

        # to store original input image size for rescaling the predictions
        self.original_input_image_width = original_image_input_size[0]
        self.original_input_image_height = original_image_input_size[1]


    def load_class_names(self, class_names_file: str):
        """load_class_names

        Args:
            class_names_file (str): absolute filepath to a text file of class names

        Raises:
            ReadError: Unable to get any class_names from the file

        Returns:
            list: class_names, list of strings of the class names
        """
        class_names = []

        with open(class_names_file) as file:
            for class_name in file:
                class_names.append(class_name.strip())

        if len(class_names) > 0:
            return class_names
        else:
            raise ReadError(f'No class names read from class_names_file: {class_names_file}')


    @staticmethod
    def download_model(url: str, model_file: str, model_dir: str = 'model'):
        """download_model
        Download a trained model from a given url and save it to a directory
        (model_dir) and give the file a particular name (model_file)

        Args:
            url (str): url to download model 
            model_file (str): name of model to save as
            model_dir (str, optional): where to save the model. Defaults to 'model'.

        Returns:
            bool: True if model was downloaded successfully
        """
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


    def build_model(self):
        """build_model
        Build the MaskRCNN model for set number of classes (num_classes)

        Returns:
            torchvision MaskRCNN predictor model: see pytorch documentation
        """
        
        # load instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='MaskRCNN_ResNet50_FPN_Weights.DEFAULT')

        # get number of input features from the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # replace pretrained head with new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        # get number of input features for mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256

        # replace mask predictor with new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, self.num_classes)
        return model


    def load_model(self, 
                   model_file: str,
                   map_location: str = 'cuda:0'):
        """load_model
        Load model based on model file (absolute file path)

        Args:
            model_file (str): absolute filepath to model
            map_location (_type_, optional): where to put the model by default
            (onto the GPU). Defaults to 'cuda:0'.

        Returns:
            torchvision model: MaskRCNN model
        """
        
        model = self.build_model()
        model.load_state_dict(torch.load(os.path.abspath(model_file), map_location=map_location))
        print(f'loaded model: {model_file}')
        model.to(self.device)
        return model


    def convert_input_image(self, image, image_color_format: str = 'RGB'):
        """convert_input_image
        Convert the input image to the correct format and size. Also records the
        original image size input was for rescaling the output detections

        Args:
            image (numpy array, PIL image or Tensor): input image to give model
            image_color_format (str, optional): determine if numpy array is and
            RGB or BGR format. Defaults to 'RGB'.

        Returns:
            tensor: pytorch tensor of resized image in RGB format
        """        
        color_format = ['RGB', 'BGR']

        if isinstance(image, np.ndarray):
            # check color format/ordering of image, convert to RGB
            if image_color_format == color_format[1]:
                image = image[:, :, [2, 1, 0]]

            # check size of image, rescale to appropriate size
            hi, wi, ci = image.shape
            # record original image input size for rescaling output at end
            self.original_input_image_height = hi
            self.original_input_image_width = wi
            if (hi != self.image_height) or (wi != self.image_width):
                image = cv.resize(image, 
                                  (self.image_width, self.image_height),
                                  interpolation=cv.INTER_NEAREST)
            # transform to tensor
            transform = transforms.ToTensor()
            image = transform(image)
        elif isinstance(image, PIL_Image.Image):
            # record original image input size for rescaling output at end
            self.original_input_image_width, self.original_input_image_height = image.size
            # resize image to fit
            image = PIL_Image.resize((self.image_width, self.image_height))
            pil_transform = transforms.PILToTensor()
            image = pil_transform(image)
        else:
            transform = transforms.ToTensor()
            image = transform(image)
            # record original image input size for rescaling output at end # TODO check this
            self.original_input_image_width, self.original_input_image_height = image.get_image_size()

            resize = transforms.Resize()
            image = resize(image, (self.image_width, self.image_height))
        return image


    def detect(self, 
               image, 
               confidence_threshold = False, 
               nms_threshold = False,
               mask_threshold = False):
        """detect
        Infer model predictions given a single image.
        If thresholds are false, defaults to object's initialised thresholds.

        Args:
            image (numpy array, PIL image or tensor): input image
            confidence_threshold (bool, optional): confidence threshold. Defaults to False.
            nms_threshold (bool, optional): non-maxima-suppression threshold. Defaults to False.
            mask_threshold (bool, optional): mask threshold. Defaults to False.

        Returns:
            list of MaskDetections: list of detections that are defined by the MaskDetections class
        """ 
        
        # assign thresholds
        if not confidence_threshold:
            confidence_threshold = self.confidence_threshold
        if not nms_threshold:
            nms_threshold = self.nms_threshold
        if not mask_threshold:
            mask_threshold = self.mask_threshold

        # convert image to valid image tensor. also determines original input
        # image size and sets corresponding detector object properties
        image = self.convert_input_image(image)

        # perform model inference
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
        
        # rescale bbox to original image dimensions
        detections_boxes = self.rescale_boxes(detections_boxes)

        detections_masks = list(detections_raw[0]['masks'][keep].detach().cpu().numpy())

        # scores are ordered from highest to lowest
        detections_scores = list(detections_raw[0]['scores'][keep].detach().cpu().numpy())

        # create Detections object for each detection
        detections = []
        for i, mask in enumerate(detections_masks):
            mask = np.transpose(mask, (1, 2, 0))
            
            # rescale mask to original image dimensions
            mask = self.rescale_mask(mask)

            # create mask detection object, polygon, centroid information, etc 
            maskdetections = MaskDetections(label = detections_class[i],
                                            score = detections_scores[i],
                                            mask_confidence= mask,
                                            mask_threshold = mask_threshold,
                                            class_names = self.class_names)
            detections.append(maskdetections)

        # TODO calibrate model confidence scores! https://towardsdatascience.com/confidence-calibration-for-deep-networks-why-and-how-e2cd4fe4a086

        # apply confidence threshold to detection scores
        detections = [d for d in detections if d.score >= confidence_threshold]
        return detections


    def rescale_boxes(self, boxes):
        """rescale_boxes
        Helper function to rescale bounding boxes to original input image size

        Args:
            boxes (list of floats 1x4): coordinates [xmin, ymin, xmax, ymax]

        Returns:
            list of floats: same as boxes
        """
        arx = float(self.original_input_image_width) / float(self.image_width)
        ary = float(self.original_input_image_height) / float(self.image_height)
        for bb in boxes:
            bb[0] = bb[0] * arx
            bb[1] = bb[1] * ary
            bb[2] = bb[2] * arx
            bb[3] = bb[3] * ary
        return boxes


    def rescale_mask(self, mask):
        """rescale_mask
        rescale mask to orginal input image size 
        Args:
            mask (_type_): _description_

        Returns:
            _type_: _description_
        """       
        mask = cv.resize(mask, 
                         (self.original_input_image_width, self.original_input_image_height),
                         interpolation=cv.INTER_NEAREST)
        return mask


    def load_image_from_string(self, image_filename: str):
        """show_image
        Given an image filename, output the image as a numpy array 
        """
        img = cv.imread(image_filename) # openCV default reads in BGR format
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB) # however, the rest of the world works with RGB
        return cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U) # just in case, normalize image from 0,1 to 0,255


    def show_detections(self, 
                        image, 
                        detections, 
                        save_image_filename:str, 
                        POLY:bool=True, 
                        SAVE:bool=True):
        """show_detections
        Given an RGB image (numpy array), output a new image with detections
        drawn ontop of the input image 
        
        Args:
            image (numpy array): image to plot detections on
            detections (list of MaskDetections): detections from Detector
            save_image_filename (str): absolute filename to save image
            POLY (bool, optional): boolean to use bounding boxes (False) or
            polygons (True). Defaults to True.
            SAVE (bool, optional): if True, save image to file. Defaults to True.

        Returns:
            _type_: _description_
        """ 
        # TODO assert valid input
        
        # plotting parameters:
        line_thickness = int(np.ceil(0.002 * max(image.shape)))
        font_scale = max(1, 0.0005 * max(image.shape)) # font scale should be function of image size
        font_thick = int(np.ceil(0.00045 * max(image.shape)))
        detection_colour = [255, 0, 0] # RGB red 
        # TODO should setup a library of colours for multiple classes, see config/classes.json

        for d in detections:
            if POLY:
                image = self.plot_poly(image, d, line_thickness, font_scale, detection_colour, font_thick)
            else:
                image = self.plot_box(image, d, line_thickness, font_scale, detection_colour, font_thick)
        
        if SAVE:
            self.save_image(image, save_image_filename)
        return image
    

    def plot_box(self, 
                 image,
                 detection, 
                 line_thickness,
                 font_scale,
                 detection_colour,
                 font_thick):
        """plot_box
        plot bounding box onto image

        Args:
            image (_type_): _description_
            detection (_type_): _description_
            line_thickness (_type_): _description_
            font_scale (_type_): _description_
            detection_colour (_type_): _description_
            font_thick (_type_): _description_

        Returns:
            _type_: _description_
        """        
        bb = np.array(detection.box, dtype=np.float32)
        
        # draw box
        image = cv.rectangle(image, 
                            (int(bb[0]), int(bb[1])),
                            (int(bb[2]), int(bb[3])),
                            color=detection_colour,
                            thickness=line_thickness)
            
        # add text to top left corner of box
        # class + confidence as a percent
        conf_str = format(detection.score * 100.0, '.0f')
        detection_str = '{}: {}'.format(detection.class_name, conf_str) 
        image = Detector.draw_rectangle_with_text(image, 
                                                  text=detection_str,
                                                  xy=(int(bb[0]), int(bb[1])), 
                                                  font_scale=font_scale, 
                                                  font_thickness=font_thick, 
                                                  rect_color=detection_colour)
        return image


    def plot_poly(self,
                  image,
                  detection,
                  lines_thick,
                  font_scale,
                  lines_color,
                  font_thick):
        """plot_poly
        plot detection polygon onto image

        Args:
            image (_type_): _description_
            detection (_type_): _description_
            lines_thick (_type_): _description_
            font_scale (_type_): _description_
            lines_color (_type_): _description_
            font_thick (_type_): _description_

        Returns:
            _type_: _description_
        """        
        poly = np.array(detection.poly)
        poly = poly.transpose().reshape((-1, 1, 2))
        cv.polylines(image,
                     pts=[poly],
                     isClosed=True,
                     color=lines_color,
                     thickness=lines_thick,
                     lineType=cv.LINE_4)

        # find a place to put polygon class prediction and confidence we find
        # the x,y-pair that's closest to the top-left corner of the image, but
        # also need it to be ON the polygon, so we know which one it is and thus
        # most likely to actually be in the image
        distances = np.linalg.norm(detection.poly, axis=0)
        minidx = np.argmin(distances)
        xmin = detection.poly[0, minidx]
        ymin = detection.poly[1, minidx]

        # add text to top left corner of box
        # class + confidence as a percent
        conf_str = format(detection.score * 100.0, '.0f')
        detection_str = '{}: {}'.format(detection.class_name, conf_str) 
        
        image = Detector.draw_rectangle_with_text(image, 
                                                  text=detection_str,
                                                  xy=(xmin, ymin), 
                                                  font_scale=font_scale, 
                                                  font_thickness=font_thick, 
                                                  rect_color=lines_color)
        return image
    

    @staticmethod
    def draw_rectangle_with_text(image, 
                                 text, 
                                 xy, 
                                 font_scale, 
                                 font_thickness, 
                                 rect_color,
                                 text_color = (255, 255, 255)):  
        """draw_rectangle_with_text
        Draw rectangle with background text onto image, rectangle should neatly
        encompass the text, so the text is legible ontop of the image

        Args:
            image (_type_): _description_
            text (_type_): _description_
            xy (_type_): _description_
            font_scale (_type_): _description_
            font_thickness (_type_): _description_
            rect_color (_type_): _description_
            text_color (tuple, optional): _description_. Defaults to (255, 255, 255).

        Returns:
            _type_: _description_
        """              

        # determine size of the text
        text_size, _ = cv.getTextSize(text, cv.FONT_HERSHEY_COMPLEX, font_scale, font_thickness)

        # add buffers/padding to the rectangle
        width_padding = 5
        height_padding = 5
        rect_width = text_size[0] + width_padding
        rect_height = text_size[1] + height_padding
        rect_x1 = xy[0]  - width_padding
        rect_y1 = xy[1] - height_padding
        rect_x2 = rect_x1 + rect_width + width_padding
        rect_y2 = rect_y1 + rect_height + height_padding + 2

        # draw rectangle
        rect_thickness = -1 # rect thickness of -1 means rectangle is filled in
        cv.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), rect_color, rect_thickness)

        cv.putText(image,
                   text,
                   (int(xy[0]), int(xy[1] + text_size[1])),
                   fontFace=cv.FONT_HERSHEY_COMPLEX,
                   fontScale=font_scale,
                   color=text_color,
                   thickness=font_thickness)
        # NOTE: technically, because of pointers, don't actually have to return the image
        # should maybe just be a void function
        return image 
    

    def save_image(self, image, image_filename: str):
        """save_image
        write image to file, given image and image_filename, assume image in in
        RGB format

        Args:
            image (_type_): _description_
            image_filename (str): _description_

        Returns:
            _type_: _description_
        """     
        # make sure directory exists
        image_dir = os.path.dirname(image_filename)
        os.makedirs(image_dir, exist_ok=True)

        # assuming image is in RGB format, so convert back to BGR
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.imwrite(image_filename, image)
        return True


    def run(self, image):
        """run
        runs the detectors and outputs classes, scores and boxes can also output
        masks and polygons, but commented out for now since they are not yet used

        must return 'classes', 'scores', 'boxes'
        classes: a list of class labels (int)
        scores: a list of confidence scores
        boxes: a list of coords [xmin, ymin, xmax, ymax]

        Args:
            image (_type_): _description_

        Returns:
            _type_: _description_
        """        

        detections = self.detect(image)

        classes = [d.label for d in detections]
        scores = [d.score for d in detections]
        boxes = [d.box for d in detections]
        # masks = [d.mask for d in detections]
        # poly = [d.poly for d in detections]
        return classes, scores, boxes


if __name__ == "__main__":

    print('Detector.py')
    
    # set LOCAL to True if using a local file model, or False to automatically download a pre-existing model
    # NOTE pre-existing/trained model link may need to be updated for most recent data
    LOCAL = True
    
    # load a model
    # infer on a given image
    # print out the classes, scores, boxes
    if LOCAL:
        model_file = '/home/agkelpie/Code/agkelpie_weed_detection/weed-detection/model/Yellangelo32/model_best.pth'
    else: # REMOTE
        model_file = '2021_Yellangelo_Tussock_v0_2022-10-12_10_35_epoch30.pth'
        url = 'https://cloudstor.aarnet.edu.au/plus/s/3XEnfIEoLEAP27o/download'
        Detector.download_model(url, model_file)
    detector = Detector(model_file = os.path.join('model', model_file)) # might have to retrain with correct image size

    # grab any images in the 'images' folder:
    # img_dir = 'images'
    img_dir = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_yellangelo32_tussock/annotated_images'
    out_dir = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_yellangelo32_tussock/detections'
    img_list = os.listdir(img_dir)
    max_image = 5
    for i, img_name in enumerate(img_list):
        if i > max_image:
            print(f'hit max images {i}')
            break
        print(f'{i}: image name = {img_name}')
        img = detector.load_image_from_string(os.path.join(img_dir, img_name))
        detections = detector.detect(img)
        save_img_name = os.path.join(out_dir, img_name[:-4] + '_det.png')
        detector.show_detections(img, detections, save_img_name, SAVE=True, POLY=True)

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