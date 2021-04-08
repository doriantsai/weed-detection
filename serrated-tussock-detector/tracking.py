#! /usr/bin/env python

import os
import numpy as np
import cv2 as cv
import time

import torch
import torchvision
# import torchvision.transforms as tvtrans
from torchvision.transforms import functional as tvtransfunc
from PIL import Image

from SerratedTussockDataset import SerratedTussockDataset, Rescale, ToTensor, Compose, Blur
from train import build_model
from webcam import grab_webcam_video
from inference import show_groundtruth_and_prediction_bbox
from get_prediction import get_prediction_image
from sort import *

"""
use SORT to demo tracking weeds
read in webcam image
apply our normal detection
apply sort
"""

def show_bbox_ids(image,
                  track_bbs_ids,
                  colors,
                  frame=None,
                  transpose_channels=True):
    """
    append image with tracked bounding boxes and ids
    """
    imgnp = image.cpu().numpy()

    if transpose_channels:
        # ax.imshow(np.transpose(imgnp, (1, 2, 0)))
        imgnp = np.transpose(imgnp, (1, 2, 0))

    imgnp = cv.normalize(imgnp,
                         None,
                         alpha=0,
                         beta=255,
                         norm_type=cv.NORM_MINMAX,
                         dtype=cv.CV_8U)

    # track_bbs_ids is a np array where each row contains a valid
    # bounding box and track_id (last column)
    for d in track_bbs_ids:
        # print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
            d = d.astype(np.int32)
            id_color = colors[d[4] % 32, :]
            id_color = (int(id_color[0]), int(id_color[1]), int(id_color[2]))

            imgnp = cv.rectangle(imgnp,
                                 (d[0], d[1]),
                                 (d[2], d[3]),
                                 color=id_color,
                                 thickness=2)
            # sc = format(d[4])
            cv.putText(imgnp,
                       '{}'.format(d[4]),
                       (int(d[0] + 10), int(d[1] + 30)),
                       fontFace=cv.FONT_HERSHEY_COMPLEX,
                       fontScale=1,
                       color=id_color,
                       thickness=2)
            if frame is not None:
                cv.putText(imgnp,
                           '{}'.format(frame),
                           (int(10), int(60)),
                           fontFace=cv.FONT_HERSHEY_COMPLEX,
                           fontScale=1,
                           color=(250, 250, 250),
                           thickness=2)

    return imgnp


def convert_predictions_to_detections(predictions):
    """
    convert model predictions to detections for SORT
    The output should be:
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """

    boxes = predictions['boxes']
    scores = predictions['scores']
    if len(boxes) > 0:
        dets = []
        for i, b in enumerate(boxes):
            # print(i)
            # print(b)
            # z = convert_bbox_to_z(b)
            z = np.array([b[0], b[1], b[2], b[3], scores[i]])
            # z = np.append(z, scores[i])
            # print(z)
            dets.append(z)
        dets = np.array(dets)
    else:
        dets = np.empty((0, 5))

    return dets

# parameter settings:
confidence_thresh = 0.6
iou_thresh = 0.5  # use same iou threshold for model and tracker?
tracker_iou_thresh = 0.5
max_age = 200  # max # frames keep detection alive without current detection
min_hits = 3 # min # of frames of associated detections b4 track initialised
colors = np.random.randint(0, 255, size=(32, 3), dtype='uint8') # used for display

# load model
save_name = 'fasterrcnn-serratedtussock-4'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = build_model(num_classes=2)
class_names = ['_background_', 'tussock']
save_path = os.path.join('output', save_name, save_name + '.pth')
model.load_state_dict(torch.load(save_path))
model.to(device)

 # setup webcam feed
fps = 5
cap = cv.VideoCapture(0)
w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

v_name_out = os.path.join('output', 'webcam', 'video_tracking.avi')
vid_w = cv.VideoWriter_fourcc(*'XVID')
vid_out = cv.VideoWriter(v_name_out,
                         fourcc=vid_w,
                         fps=fps,
                         frameSize=(int(1066), int(800)))
tform_rsc = Rescale(800)

# create instance of tracker
weed_tracker = Sort(max_age=max_age,
                    min_hits=min_hits,
                    iou_threshold=tracker_iou_thresh)

# iterate over webcam feed

MAX_FRAMES = 10000
i = 0
while (cap.isOpened() and i < MAX_FRAMES):
    # get frame from capture device
    ret, frame = cap.read()

    if ret:
        start_time = time.time()

        # convert image from BGR to RGB
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # convert to PIL Image (for tform)
        frame_p = Image.fromarray(frame)
        # rescale to 800 (for model)
        frame_800 = tform_rsc(frame_p)
        # convert to tensor (for model)
        frame_t = tvtransfunc.to_tensor(frame_800)
        # send frame to gpu
        frame_t.to(device)

        # detection on frame
        print('frame detection {}'.format(i))
        pred, keep = get_prediction_image(model,
                                          frame_t,
                                          confidence_thresh,
                                          iou_thresh,
                                          device,
                                          class_names)

        # convert predictions to detections for tracking:
        detections = convert_predictions_to_detections(pred)
        print(detections)

        tracks = weed_tracker.update(detections)
        print(tracks)
        # import code
        # code.interact(local=dict(globals(), **locals()))
        # track_bbs_ids is a np array where each row contains a valid
        # bounding box and track_id (last column)

        # TODO show bbs and ids from SORT on images
        img = show_bbox_ids(image=frame_t,
                            track_bbs_ids=tracks,
                            colors=colors,
                            frame=i,)
        # img = show_groundtruth_and_prediction_bbox(image=frame_t,
        #                                            predictions=pred,
        #                                            keep=keep)

        # convert image from RGB to BGR for display purposes
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        # write to video
        vid_out.write(img)
        # increment frame counter
        i += 1

        end_time = time.time()
        sec = end_time - start_time
        print('cycle time: {} sec = {} Hz'.format(sec, 1.0 / sec))

        # show image
        cv.imshow('frame', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('ERROR: ret is not True')
        break

cap.release()
vid_out.release()
cv.destroyAllWindows()

print('end of tracking.py')
import code
code.interact(local=dict(globals(), **locals()))