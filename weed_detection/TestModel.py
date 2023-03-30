#! /usr/bin/env python3

"""
Evaluator

object that prints detections alongside groundtruth labels
"""

import os
from typing import Tuple
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from weed_detection.Annotations import Annotations as Ann
from weed_detection.Detector import Detector

class TestModel:
    
    # default parameters
    IMAGE_INPUT_SIZE_DEFAULT = (1028, 1232) # TODO should make a config file
    SPECIES_FILE_DEFAULT = os.path.join(os.getcwd(), 'model/names_yellangelo32.txt')
    MODEL_FILE_DEFAULT = os.path.join('/home/agkelpie/Code/agkelpie_weed_detection/weed-detection/model/Yellangelo32/model_best.pth')
    CONFIDENCE_THRESHOLD_DEFAULT = 0.5
    NMS_THRESHOLD_DEFAULT = 0.5
    MASK_THRESHOLD_DEFAULT = 0.5

    # annotation data defaults, a list of dictionaries, defining the
    # annotation file, correesponding image directory and mask directory
    ANNOTATION_DATA_DEFAULT = {'annotation_file': '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_yellangelo_tussock/dataset.json',
                               'image_dir': '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_yellangelo_tussock/annotated_images',
                               'mask_dir': '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_yellangelo_tussock/masks'}
    
    # default output directory, where models and progress checkpoints are saved
    OUTPUT_DIR_DEFAULT = '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_yellangelo_tussock/images_test'

    # default image save name if none are given
    DEFAULT_IMAGE_SAVE_NAME = os.path.join(os.getcwd(), 'image_test.png')


    def __init__(self,
                 model_file: str = MODEL_FILE_DEFAULT,
                 annotation_data: dict = ANNOTATION_DATA_DEFAULT,
                 output_dir: str = OUTPUT_DIR_DEFAULT,
                 names_file: str = SPECIES_FILE_DEFAULT,
                 image_input_size: Tuple [int, int] = IMAGE_INPUT_SIZE_DEFAULT,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD_DEFAULT,
                 nms_threshold: float = NMS_THRESHOLD_DEFAULT,
                 mask_threshold: float = MASK_THRESHOLD_DEFAULT):   
         
        # annotation data
        self.annotation_data = annotation_data
        self.annotation_object = Ann(filename=annotation_data['annotation_file'],
                                     img_dir=annotation_data['image_dir'],
                                     mask_dir=annotation_data['mask_dir'])
         
        # detector
        self.detector = Detector(model_file, 
                                 names_file, 
                                 image_input_size,
                                 confidence_threshold,
                                 nms_threshold,
                                 mask_threshold)
        
        # output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        

    def show_annotations(self, image, image_annotation, save_image_filename=None, SAVE=False):
        # image as numpy array on CPU
        # groundtruth as an annotations object, indexed to the relevant image
        # SAVE as boolean flag to save figure or not
        # for each region in groundtruth
        # draw the bounding box + class name in the bottom left corner (so doesn't overlap with detections text)

        lines_colour = (0, 0, 255) # RGB 
        line_thickness = int(np.ceil(0.004 * max(image.shape)))
        font_scale = max(1, 0.0005 * max(image.shape))
        font_thick = int(np.ceil(0.00045 * max(image.shape)))

        for regions in image_annotation.regions:
            if regions.shape_type == 'polygon':
                xy = np.array(regions.shape.exterior.coords.xy, dtype=np.int32)

                # find a place to put polygon class prediction and confidence we find
                # the x,y-pair that's closest to the top-left corner of the image, but
                # also need it to be ON the polygon, so we know which one it is and thus
                # most likely to actually be in the image
                distances = np.linalg.norm(xy, axis=0)
                minidx = np.argmin(distances)
                xmin = xy[0, minidx]
                ymin = xy[1, minidx]

                # xy2 = xy.reshape((-1, 1, 2))
                xy = xy.transpose().reshape((-1, 1, 2))

                cv.polylines(image, 
                             pts=[xy], 
                             isClosed=True, 
                             color=lines_colour, 
                             thickness=line_thickness,
                             lineType=cv.LINE_4) # vs cv.FILLED

                image = Detector.draw_rectangle_with_text(image,
                                                          text=regions.class_name,
                                                          xy=(xmin, ymin),
                                                          font_scale=font_scale,
                                                          font_thickness=font_thick,
                                                          rect_color=lines_colour)

                

        if SAVE:
            if save_image_filename is None:
                save_image_filename = self.DEFAULT_IMAGE_SAVE_NAME
            self.detector.save_image(image, save_image_filename)

        return image
         

    def show_image_test(self, image_filename, POLY=True):
        # given image filename
        # find the corresponding image in the annotations as groundtruth
        # run detector
        # run show_groundtruth
        # run show_detections
        # save image

        # find the corresponding image in the annotations as groundtruth
        # since ann.imgs corresponds to the ordering of annotations, the index is the same
        img_name = os.path.basename(image_filename)
        idx = self.annotation_object.imgs.index(img_name)
        image_annotation = self.annotation_object.annotations[idx]

        # load the image
        image = self.detector.load_image_from_string(image_filename)

        # show the gt on the image
        # save_name = img_name[:-4] + '_ann.png'
        # save_image_filename_ann = os.path.join(self.output_dir, save_name) 
        image_ann = self.show_annotations(image, image_annotation)

        # run detector on the image
        detections = self.detector.detect(image)

        # show detections as overlay
        save_name = img_name[:-4] + '_test.png'
        save_image_filename = os.path.join(self.output_dir, save_name)
        image_det = self.detector.show_detections(image_ann, detections, save_image_filename, POLY=POLY, SAVE=True)

        return image_det



if __name__ == "__main__":

    print('TestModel.py')

    Test = TestModel() # use defaults

    ANNOTATION_DATA = {'annotation_file': '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_yellangelo_tussock/dataset.json',
                        'image_dir': '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_yellangelo_tussock/annotated_images',
                        'mask_dir': '/home/agkelpie/Code/agkelpie_weed_detection/agkelpiedataset_yellangelo_tussock/masks'}
    
    img_list = os.listdir(ANNOTATION_DATA['image_dir'])
    
    max_img = 5
    for i, img_name in enumerate(img_list):
        if i > max_img:
            print(f'reached max images: {max_img}')
            break
        print(f'{i}: {os.path.basename(img_name)}')
        Test.show_image_test(os.path.join(ANNOTATION_DATA['image_dir'], img_name))

    print(f'saved images in {Test.output_dir}')
    print('Complete TestModel.py')

# functions:
# show_groundtruth
# setup test_model 



# as a script,
# setup image directory, annotation file
# setup model file
# load annotations
# from annotations, specify image
# run detection on image
# run show_detections for image painted with detections
# run show_groundtruth for image painted with groundtruth (might swap the order of det/gt)

