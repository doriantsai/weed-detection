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




class Detector:

    # parameters

    def __init__(self, params):
        self.detections = []

        self.model = []

    def load_class_names(self, class_names_file):
        class_names = class_names_file
        return class_names

    def build_model(self, model_file):
        model = model_file
        return model

    def threshold_predictions(self, pred, thresh):
        return pred
        
    def run(self, image, thresholds):

        classes, scores, boxes = self.model.detect(image, thresholds)
        return classes, scores, boxes