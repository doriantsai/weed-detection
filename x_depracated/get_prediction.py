#! /usr/bin/env python
import torch
import torchvision
import numpy


def get_prediction_batch(model, images, confidence_threshold, iou_threshold, device, class_names):
    """ take in model, image batch and confidence threshold,
    return bbox predictions for scores > threshold """
    """ TODO incomplete """


    # send all images in batch of images to device
    images = list(image.to(device) for image in images)

    # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    model.to(device)
    pred = model(images)

    # import code
    # code.interact(local=dict(globals(), **locals()))

    # TODO update to handle batches (need to index over the 0's)
    # do non-maxima suppression
    keep = torchvision.ops.nms(pred[0]['boxes'], pred[0]['scores'], iou_threshold)


    # TODO may not have to keep on cpu, might be faster on gpu
    pred_class = [class_names[i] for i in list(pred[0]['labels'][keep].cpu().numpy())]
    pred_boxes = [[bb[0], bb[1], bb[2], bb[3]] for bb in list(pred[0]['boxes'][keep].detach().cpu().numpy())]
    # scores are ordered from highest to lowest
    pred_score = list(pred[0]['scores'][keep].detach().cpu().numpy())

    if len(pred_score) > 0:
        if max(pred_score) < confidence_threshold: # none of pred_score > thresh, then return empty
            pred_thresh = []
            pred_boxes = []
            pred_class = []
            pred_score = []
        else:
            pred_thresh = [pred_score.index(x) for x in pred_score if x > confidence_threshold][-1]
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

    return predictions, keep


def get_prediction_image(model,
                         image,
                         confidence_threshold,
                         nms_iou_threshold,
                         device):
    """ take in model, image and confidence threshold,
    return bbox predictions for scores > threshold """

    # image in should be a tensor, because it's coming from a dataloader
    # for now, assume it is a single image, as opposed to a batch of images
    model.eval()
    if torch.cuda.is_available():
        image = image.to(device)
    pred = model([image])

    # do non-maxima suppression
    keep = torchvision.ops.nms(pred[0]['boxes'], pred[0]['scores'], nms_iou_threshold)
    # import code
    # code.interact(local=dict(globals(), **locals()))

    # TODO may not have to keep on cpu, might be faster on gpu
    # pred_class = [class_names[i] for i in list(pred[0]['labels'][keep].cpu().numpy())]
    pred_class = [i for i in list(pred[0]['labels'][keep].cpu().numpy())]
    pred_boxes = [[bb[0], bb[1], bb[2], bb[3]] for bb in list(pred[0]['boxes'][keep].detach().cpu().numpy())]
    # scores are ordered from highest to lowest
    pred_score = list(pred[0]['scores'][keep].detach().cpu().numpy())
    pred_labels = pred_class  # TODO temp - can't tensor strings, and coco api/tools requires key/value labels

    predictions = {}
    predictions['boxes'] = pred_boxes
    predictions['classes'] = pred_class
    predictions['scores'] = pred_score
    predictions['labels'] = pred_labels

    # confidence threshold
    predictions = threshold_predictions(predictions, confidence_threshold)

    return predictions, keep


def threshold_predictions(pred, thresh):
    """ apply confidence threshold to predictions """
    # unpack predictions
    # apply threshold
    # apply to all relevant key-value pairs
    # pack predictions/return
    pred_boxes = pred['boxes']
    pred_class = pred['classes']
    pred_score = pred['scores']
    pred_labels = pred['labels']

    if len(pred_score) > 0:
        if max(pred_score) < thresh: # none of pred_score > thresh, then return empty
            pred_thresh = []
            pred_boxes = []
            pred_class = []
            pred_score = []
            pred_labels = []
        else:
            pred_thresh = [pred_score.index(x) for x in pred_score if x > thresh][-1]
            pred_boxes = pred_boxes[:pred_thresh+1]
            pred_class = pred_class[:pred_thresh+1]
            pred_score = pred_score[:pred_thresh+1]
            pred_labels = pred_class[:pred_thresh+1]
    else:
        pred_thresh = []
        pred_boxes = []
        pred_class = []
        pred_score = []
        pred_labels = []

    predictions = {}
    predictions['boxes'] = pred_boxes
    predictions['classes'] = pred_class
    predictions['scores'] = pred_score
    predictions['labels'] = pred_labels

    return predictions

