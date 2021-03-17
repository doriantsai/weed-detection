#! /usr/bin/env python

# run model on entirety of new dataset
# print the results and save them as a .json file

import os
import torch
import torchvision
import torch.utils.data
import pickle
import numpy as np
import cv2 as cv
import json

from PIL import Image
from train import build_model
from inference import show_groundtruth_and_prediction_bbox
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as tvtransfunc
from SerratedTussockDataset import SerratedTussockDataset, RandomHorizontalFlip, Rescale, ToTensor, Compose

from get_prediction import get_prediction_image

""" get bounding boxes dictionary """
def get_regions_dict(predictions):
    reg_dict = {}
    for ib, b in enumerate(predictions['boxes']):
        reg_dict[str(ib)] = {
            'shape_attributes': {
                'name': 'rect',
                'x': int(b[0]),
                'y': int(b[1]),
                'width': int(b[2] - b[0]),
                'height': int(b[3] - b[1]),
            },
            'region_attributes': {
                'serrated tussock': 1
            }
        }

    return reg_dict


""" bbox unscale """
def unscale_bounding_box(image_orig, image_resc, pred_resc):
    # image_orig - original image scale (numpy array)
    # image_resc - rescaled image for model (numpy array)
    # bounding box - predictions are with respect to image_resc size (numpy array)
    # need to adjust bbox to iamge_orig size
    # print(type(image_orig))
    # print(type(image_resc))
    # print(type(pred_resc))

    w_orig, h_orig, _ = image_orig.shape
    w_resc, h_resc, _ = image_resc.shape
    xChange = float(w_orig) / float(w_resc)
    yChange = float(h_orig) / float(h_resc)

    bbox = np.array(pred_resc['boxes'])
    if len(bbox) == 0:
        print('rescale bbox: no boxes here')
    else:


        bbox[:, 0] = bbox[:, 0] * yChange
        bbox[:, 1] = bbox[:, 1] * xChange
        bbox[:, 2] = bbox[:, 2] * yChange
        bbox[:, 3] = bbox[:, 3] * xChange

        # import code
        # code.interact(local=dict(globals(), **locals()))

        pred_resc['boxes'] = bbox.tolist()

    return pred_resc

# """ get image dictionary """
# def get_image_dict(img_name, img_path, predictions):
#     # save as one big dictionary
#     # filename
#     ann_dict = {}

#     # get file size
#     file_size = os.path.getsize(os.path.join(img_path, img_name))
#     ann_dict = {img_name + str(file_size): {
#                     'fileref': "",
#                     'size': file_size,
#                     'filename': img_name,
#                     'file_attributes': {},
#                     'regions': get_regions_dict(predictions)}
#                 }

#     return ann_dict



# -----------------------------------------------------------------------------#

if __name__ == "__main__":

    # import model
    # set device -> GPU
    # set new dataset location
    # create dataloader for new dataset
    # iterate over dataloader, do predictions/save bboxes

    # setup device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load model
    save_name = 'fasterrcnn-serratedtussock-4'
    save_model_path = os.path.join('output', save_name, save_name + '.pth')
    print('load model: {}'.format(save_model_path))
    model = build_model(num_classes=2)
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # print('in_features = {}'.format(in_features))
    # num_classes = 2
    class_names = ["_background_", "serrated tussock"]
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(save_model_path))
    model.to(device)
    model.eval()

    # setup new dataset to be labelled/bootstrapped
    root_dir = os.path.join('/home', 'dorian', 'Data', 'SerratedTussockDataset_v2')
    json_file = os.path.join('Annotations', 'via_region_data_bootstrap.json')
    save_file = os.path.join(root_dir, json_file)


    # TODO
    # grab list of all files in folder
    # for each image, convert frame to PIL image
    # transform to 800 pix using tform
    # convert to tensor
    # send to device
    # set thresholds
    # get prediction
    # show_groundtruth and prediction box
    # convert from RGB2BGR
    # save image and add to json/file (via append?)?

    tform_rsc = Rescale(800)


    files = sorted(os.listdir(os.path.join(root_dir, 'Images')))
    # sort files in alphabetical order
    i = 0
    ann_dict = {}
    for f in files:
        # TODO assert f is an image

        # prep image for use with model
        img = Image.open(os.path.join(root_dir, 'Images', f))
        # import code
        # code.interact(local=dict(globals(), **locals()))
        # c, h, w = img.size()
        # if c > 3:
            # 4 or more layers (eg, alpha layer), just take the first 3
            # img = img[0:3, :, :]
        # if img.mode == 'RGBA':
        #     # remove alpha
        #     img = Ima


        img_rs = tform_rsc(img)
        img_rs = tvtransfunc.to_tensor(img_rs)
        img_rs.to(device)

        # difficult to remove alpha in PIL image form, I think more trivial as a tensor
        c, h, w = img_rs.size()
        if c > 3:
            img_rs = img_rs[0:3, :, :]

        # do inference
        print('inference on {}: {}'.format(i, f))
        conf_thresh = 0.6
        iou_thresh = 0.5

        pred, keep = get_prediction_image(model, img_rs, conf_thresh, iou_thresh, device, class_names)
        img_rs = show_groundtruth_and_prediction_bbox(img_rs, predictions=pred, keep=keep)

        # save image
        img_filename = os.path.join(root_dir, 'Bootstrap_Images', f[:-4] + '_pred.png')
        img_rs = cv.cvtColor(img_rs, cv.COLOR_RGB2BGR)
        cv.imwrite(img_filename, img_rs)

        # TODO save the bbox into json file
        # save_file = os.path.join(root_dir, json_file)
        # first save as a list, then write to text later

        # TODO resize bboxes to appropriate output image size


        img_orig_np = np.array(img)
        pred_rs = pred
        pred_orig = unscale_bounding_box(img_orig_np, img_rs, pred_rs)

        # img_dict = get_image_dict(f, os.path.join(root_dir, 'Images'), pred)

        file_size = os.path.getsize(os.path.join(os.path.join(root_dir, 'Images'), f))
        ann_dict[f + str(file_size)] = {
                    'fileref': "",
                    'size': file_size,
                    'filename': f,
                    'file_attributes': {},
                    'regions': get_regions_dict(pred_orig)
                    }


        i += 1
        # TEMP just iterate 4x for now
        # if i >= 4:
        #     break


    with open(save_file, 'w') as ann_file:
        json.dump(ann_dict, ann_file, indent=4)



    import code
    code.interact(local=dict(globals(), **locals()))







