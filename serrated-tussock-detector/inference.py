#! /usr/bin/env python


import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import matplotlib.pyplot as plt
import os
import utils
import pickle
import json

from train import build_model
from SerratedTussockDataset import SerratedTussockDataset, RandomHorizontalFlip, Rescale, ToTensor, Compose
import matplotlib.patches as mpp
import torchvision.transforms as tvtrans
from torchvision.transforms import functional as tvtransfunc
from webcam import grab_webcam_video
import cv2 as cv
from PIL import Image
import time
from get_prediction import get_prediction_image
from engine_st import evaluate

# def matplotlib_imshow(img):
#     # image in, probably as a tensor
#     # want to show image using plt.imshow(img)
#     # img = img / 2 + 0.5     # unnormalize
#     imgnp = img.cpu().numpy()
#     plt.imshow(np.transpose(imgnp, (1, 2, 0)))
#     return plt

def cv_imshow(img, winname, wait_time=2000):
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.namedWindow(winname, cv.WINDOW_GUI_NORMAL)
    cv.imshow(winname, img)
    cv.waitKey(wait_time)
    cv.destroyWindow(winname)


# def show_single_bbox(img, bb, color='red'):
#     # put bounding box onto image
#     # bb is a numpy array or tensor?
#     # TODO convert to opencv -> cv.rectangle
#     bb = np.array(bb.cpu(), dtype=np.float32)
#     rect = mpp.Rectangle((bb[0], bb[1]),
#                           bb[2] - bb[0],
#                           bb[3] - bb[1],
#                           color=color,
#                           fill=False,
#                           linewidth=3)


# def show_image_bbox(image, sample, color='blue'):
#     # show image and bounding box together
#     # matplotlib_imshow(image)
#     # TODO convert to opencv -> cv.rectangle
#     imgnp = image.cpu().numpy()

#     fig, ax = plt.subplots(1)

#     ax.imshow(np.transpose(imgnp, (1, 2, 0)))

#     boxes = sample['boxes']
#     nbb, _ = boxes.size()

#     print(imgnp.shape)

#     for i in range(nbb):
#         print('plot box {}'.format(i))
#         bb = np.array(boxes[i, :].cpu(), dtype=np.float32)
#         print(bb)  # [xmin, ymin, xmax, ymax]
#         rect = mpp.Rectangle((bb[0], bb[1]),
#                              bb[2] - bb[0],
#                              bb[3] - bb[1],
#                              color=color,
#                              fill=False,
#                              linewidth=3)
#         ax.add_patch(rect)

#         # plt.gca().add_patch(show_single_bbox(image, boxes[i, :]))

#     return fig, ax


def show_groundtruth_and_prediction_bbox(image,
                         sample=None,
                         predictions=None,
                         keep=None,
                         outcomes=None,
                         sample_color=(0, 0, 255), #RGB
                         predictions_color=(255, 0, 0), #RGB
                         iou_color=(255, 255, 255),  #RGB
                         transpose_channels=True,
                         transpose_color_channels=False):
    # show image, sample/gt bounding box, and predictions bounding box together

    gt_box_thick = 12
    dt_box_thick = 6
    out_box_thick = 3

    if transpose_color_channels:
        print('swap colour channels in tensor format')
        # if read in through opencv, then format is bgr
        # solution: just read in as RGB
        # RGB vs BGR
        image = image[(2, 0, 1), :, :]

    imgnp = image.cpu().numpy()
    # fig, ax = plt.subplots(1)
    if transpose_channels:
        # ax.imshow(np.transpose(imgnp, (1, 2, 0)))
        imgnp = np.transpose(imgnp, (1, 2, 0))
    # else:
    #     ax.imshow(imgnp)

    # with opencv, need to normalize the image to 0,255 range:
    # assuming it's coming in
    imgnp = cv.normalize(imgnp,
                         None,
                         alpha=0,
                         beta=255,
                         norm_type=cv.NORM_MINMAX,
                         dtype=cv.CV_8U)

    # first, plot the groundtruth bboxes
    if sample is None:
        print('no groundtruth')
    else:
        boxes_gt = sample['boxes']
        if len(boxes_gt) == 0:
            print('show groundtruth: no boxes here')
        else:
            nbb_gt, _ = boxes_gt.size()
            for i in range(nbb_gt):
                bb = np.array(boxes_gt[i,:].cpu(), dtype=np.float32)

                # rect = mpp.Rectangle((bb[0], bb[1]),
                #                     bb[2] - bb[0],
                #                     bb[3] - bb[1],
                #                     color=sample_color,
                #                     fill=False,
                #                     linewidth=3)
                # import code
                # code.interact(local=dict(globals(), **locals()))

                imgnp = cv.rectangle(imgnp,
                                     (int(bb[0]), int(bb[1])),
                                     (int(bb[2]), int(bb[3])),
                                     color=sample_color,
                                     thickness=gt_box_thick)
                # ax.add_patch(rect)

    # second, plot predictions
    if not predictions is None:
        boxes_pd = predictions['boxes']
        boxes_score = predictions['scores']

        if len(boxes_pd) > 0:
            for i in range(len(boxes_pd)):
                bb = np.array(boxes_pd[i], dtype=np.float32)  # TODO might just specify np.int16?
                imgnp = cv.rectangle(imgnp,
                                    (int(bb[0]), int(bb[1])),
                                    (int(bb[2]), int(bb[3])),
                                    color=predictions_color,
                                    thickness=dt_box_thick)

                # add text to top left corner of bounding box
                sc = format(boxes_score[i], '.2f') # show 4 decimal places
                cv.putText(imgnp,
                          '{}: {}'.format(i, sc),
                          (int(bb[0] + 10), int(bb[1]) + 30),
                          fontFace=cv.FONT_HERSHEY_COMPLEX,
                          fontScale=1,
                          color=predictions_color,
                          thickness=2)

        # third, add iou
        if (outcomes is not None) and (predictions is not None):
            iou = outcomes['dt_iou']

            # iou should be a list or array with iou values for each boxes_pd
            boxes_pd = predictions['boxes']
            if len(iou) > 0 and len(boxes_pd) > 0:
                for i  in range(len(iou)):
                    bb = np.array(boxes_pd[i], dtype=np.float32)
                    # needed to get top/left corner of bbox
                    iou_str = format(iou[i], '.2f')
                    cv.putText(imgnp,
                               'iou: {}'.format(iou_str),
                               (int(bb[0]+ 10), int(bb[1] + 60)), # place iou under confidence score
                               fontFace=cv.FONT_HERSHEY_COMPLEX,
                               fontScale=1,
                               color=iou_color,
                               thickness=2)

        # fourth, add outcome (TP, FP, FN, TN)
        if (outcomes is not None) and \
            (predictions is not None) and \
            (sample is not None):
            # for each prediction, there is an outcome, which is an array with 1-4
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
            boxes_pd = predictions['boxes']
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
                    # replot the detection boxes' colour based on outcome
                    bb = np.array(boxes_pd[i], dtype=np.float32)  # TODO might just specify np.int16?
                    imgnp = cv.rectangle(imgnp,
                                        (int(bb[0]), int(bb[1])),
                                        (int(bb[2]), int(bb[3])),
                                        color=outcome_color[dt_outcome[i]],
                                        thickness=out_box_thick)

                # add text to top left corner of bounding box
                    sc = format(boxes_score[i], '.2f') # show 4 decimal places
                    cv.putText(imgnp,
                            '{}: {}/{}'.format(i, sc, outcome_list[dt_outcome[i]]),
                            (int(bb[0]+ 10), int(bb[1]) + 30),
                            fontFace=cv.FONT_HERSHEY_COMPLEX,
                            fontScale=1,
                            color=outcome_color[dt_outcome[i]],
                            thickness=2)

            # falseneg negatives cases wrt ground-truth bounding boxes
            boxes_gt = sample['boxes']
            fn_gt = outcomes['fn_gt']
            if len(fn_gt) > 0 and len(boxes_gt) > 0:
                for j in range(len(boxes_gt)):
                    # groundtruth boxes already plotted, so only replot them
                    # if false negative case
                    if fn_gt[j]:
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
                            fontScale=1,
                            color=outcome_color[2],
                            thickness=2)

    return imgnp


@torch.no_grad()
def infer_dataset(model,
                  subdataset,
                  confidence_threshold,
                  iou_threshold,
                  save_folder_name,
                  device,
                  class_names,
                  output_folder=None,
                  dataset=None,
                  wait_time=1000,
                  imshow=True,
                  img_name_suffix=None):
    """
    infer model on entire dataset
    """

    model.eval()
    model.to(device)
    out = []
    for image, sample in subdataset:
        image_id = sample['image_id'].item()
        if dataset is not None:
            img_name = dataset.annotations[image_id]['filename'][:-4]
        else:
            img_name = str(image_id)

        pred, keep = get_prediction_image(model,
                                          image,
                                          confidence_threshold,
                                          iou_threshold,
                                          device,
                                          class_names)
        image_marked = show_groundtruth_and_prediction_bbox(image,
                                                            sample=sample,
                                                            predictions=pred,
                                                            keep=keep)
        if not os.path.isdir(os.path.join('output', save_folder_name)):
            os.mkdir(os.path.join('output', save_folder_name))

        if output_folder is None:
            subfolder = 'dataset'
        else:
            subfolder = output_folder
        if not os.path.isdir(os.path.join('output', save_folder_name, subfolder)):
            os.mkdir(os.path.join('output', save_folder_name, subfolder))
        # if output_folder is None:
            # dsname = f'{dataset=}'.split('=')[0]

        if img_name_suffix is None:
            image_name = os.path.join('output',
                                    save_folder_name,
                                    subfolder,
                                    img_name + '.png')
        else:
            image_name = os.path.join('output',
                                    save_folder_name,
                                    subfolder,
                                    img_name + img_name_suffix + '.png')

        image_marked = cv.cvtColor(image_marked, cv.COLOR_RGB2BGR)
        cv.imwrite(image_name, image_marked)
        if imshow:
            winname = 'testing'
            cv.namedWindow(winname, cv.WINDOW_NORMAL)
            cv.imshow(winname, image_marked)
            cv.waitKey(wait_time)
            cv.destroyWindow(winname)

        # save output
        # TODO HACK until output keep bug is solved:
        outtensor = model([image.to(device)])

        # convert from tensor gpu to tensor cpu
        outnumpy = [{k: v.cpu().numpy() for k, v in t.items()} for t in outtensor]

        # import code
        # code.interact(local=dict(globals(), **locals()))
        # outnumpy[0]["image_id"] = image_id
        # need to zip w/ targets
        # res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        # res = {sample["image_id"].item(): outnumpy}

        out.append(outnumpy)

    return out



# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    # test the detector model by plotting a sample image

    # setup device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load model

    # model = build_modegit statl(nclasses)

    model = build_model(num_classes=2)
    CLASS_NAMES = ["_background_", "serrated tussock"]

    # save_name = 'fasterrcnn-serratedtussock-4'
    save_name = 'Tussock_v0_8'
    save_path = os.path.join('output', save_name, save_name + '.pth')
    model.load_state_dict(torch.load(save_path))

    # setup dataset
    root_dir = os.path.join('SerratedTussockDataset')
    json_file = os.path.join('Annotations', 'via_region_data.json')

    # load stuff:
    data_save_path = os.path.join('.', 'output', save_name, save_name + '.pkl')
    with open(data_save_path, 'rb') as f:
        dataset_tform_test = pickle.load(f)
        dataset_tform_train = pickle.load(f)
        dataset_train = pickle.load(f)
        dataset_val = pickle.load(f)
        dataset_test = pickle.load(f)
        dataloader_test = pickle.load(f)
        dataloader_train = pickle.load(f)
        dataloader_val = pickle.load(f)
        hp = pickle.load(f)

    INFER_ON_TRAINING = False
    INFER_ON_TEST = False
    INFER_ON_SINGLE_IMAGE = True
    INFER_ON_JOH = False
    INFER_ON_VIDEO = False
    INFER_ON_VAL = False

    if INFER_ON_TRAINING:
        print('training set')
        # infer on entire dataset + save images
        confidence_thresh = 0.5
        iou_thresh = 0.5
        out_test = infer_dataset(model,
                                 dataset_train,
                                 confidence_thresh,
                                 iou_thresh,
                                 save_name,
                                 device,
                                 CLASS_NAMES,
                                 output_folder='train',
                                 dataset=dataset_train.dataset.dataset,
                                 imshow=False,
                                 img_name_suffix='_train')

    # test model inference on a single image to see if the predictions are changing
    # should be consistent/not change

    if INFER_ON_SINGLE_IMAGE:
        print('single test image')

        with torch.no_grad():
            # model.eval()
            model.to(device)
            imgs, smps = next(iter(dataloader_test))
            model_conf = 0.5
            model_iou = 0.5
            bs = 1
            for i in range(bs):
                img = imgs[i]
                smp = smps[i]
                image_id = smp['image_id']
                print('image_id = ', str(image_id))

                for j in range(10):
                    pred, keep = get_prediction_image(model,
                                                    img,
                                                    model_conf,
                                                    model_iou,
                                                    device,
                                                    CLASS_NAMES)

                    print('iter: {} :: {}'.format(j, pred))

                imgname = os.path.join('output', save_name, 'single_image_model_infer.png')
                img_out = show_groundtruth_and_prediction_bbox(img, smp, pred)
                cv.imwrite(imgname, img_out)
                cv_imshow(img_out, winname='single image')

                print(pred)

    if INFER_ON_VAL:
        print('validation set')

        confidence_thresh = 0.5
        iou_thresh = 0.5
        out_test = infer_dataset(model,
                                 dataset_val,
                                 confidence_thresh,
                                 iou_thresh,
                                 save_name,
                                 device,
                                 CLASS_NAMES,
                                 output_folder='validation',
                                 dataset=dataset_val.dataset.dataset,
                                 imshow=True,
                                 img_name_suffix='_val')

        # save output
        save_output_test = os.path.join('output', save_name, 'predictions_test.json')
        with open(save_output_test, 'w') as out_file:
            json.dump(out_test, ann_file, indent=4)

    if INFER_ON_TEST:
        print('testing set')

        confidence_thresh = 0.8
        iou_thresh = 0.5
        out_test = infer_dataset(model,
                                 dataset_test,
                                 confidence_thresh,
                                 iou_thresh,
                                 save_name,
                                 device,
                                 CLASS_NAMES,
                                 output_folder='dataset-test',
                                 dataset=dataset_test.dataset)

        # save output
        save_output_test = os.path.join('output', save_name, 'predictions_test.json')
        with open(save_output_test, 'w') as out_file:
            json.dump(out_test, ann_file, indent=4)


    # --------------- #
    # try joh's images
    if INFER_ON_JOH:
        print('joh image set')
        # create a new dataset, run inference
        joh_folder = os.path.join('/home', 'dorian', 'Data', 'JohImagesDataset')
        json_file = os.path.join('Annotations', 'via_region_data.json')
        tforms = Compose([Rescale(800), RandomHorizontalFlip(0.5), ToTensor()])
        dataset_joh = SerratedTussockDataset(joh_folder, json_file, tforms)
        bs = 6
        dataloader_joh = torch.utils.data.DataLoader(dataset_joh,
                                                    batch_size=bs,
                                                    shuffle=False,
                                                    num_workers=0,
                                                    collate_fn=utils.collate_fn)
        infer_dataset(model,
                      dataset_joh,
                      confidence_thresh,
                      iou_thresh,
                      save_name,
                      device,
                      CLASS_NAMES,
                      output_folder='dataset-joh',
                      dataset=dataset_joh)


    if INFER_ON_VIDEO:
        # read in video using webcam.py's grab_webcam_video
        # run on a single frame within the video

        # get video
        # save video
        # read in video
        # iterate through video frames
        # infer on video frame

        print('getting video')
        # video_name_in = os.path.join('output', 'webcam', 'video_raw.avi')
        fps = 5
        # grab_webcam_video(outpath=video_name_in, fps=fps)

        # vc = cv.VideoCapture(video_name_in)
        cap = cv.VideoCapture(0)
        #  # get width/height of frame
        w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

        # TODO set webcam settings (eg, shutter speed, white balance, etc)

        # TODO save as video, but for now just plot out images
        video_name_out = os.path.join('output', 'webcam', 'video_inference.avi')
        # fourccc = cv.VideoWriter_fourcc('M','P','E','G')
        vid_w = cv.VideoWriter_fourcc(*'XVID')
        # import code
        # code.interact(local=dict(globals(), **locals()))

        # vw = cv.VideoWriter_fourcc(*'XVID')
        # out = cv.VideoWriter(outpath, vw, fps, (int(w), int(h)))
        vid_out = cv.VideoWriter(video_name_out,
                                 fourcc=vid_w,
                                 fps=fps,
                                 frameSize=(int(1066), int(800)))  # images get resized for the model

        # need to resize incoming frame to expected model image size, so..
        tform_rsc = Rescale(800)

        # for now, read video and just run model inference on each frame
        # then write back into video format
        # TODO consider looking into video datasets and torchvision.io stuff (but requires install from source)

        # read in the video,

        i = 0
        MAX_VIDEO_FRAMES = 10000
        # send model to GPU for speed
        model.to(device)

        while (cap.isOpened() and i < MAX_VIDEO_FRAMES):  # when we reach the end of the video, need to shut down, how?


            # how long to go through this cycle
            start_time = time.time()

            ret, frame = cap.read()
            # TODO consider how to skip frames

            if ret:
                # convert image from BGR to RGB
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                # first convert frame to PIL Image, which the tform expects
                frame_p = Image.fromarray(frame)

                # send frame to gpu

                # resize numpy array to size/shape expected by model:
                # 800 pix min
                frame_800 = tform_rsc(frame_p)

                # convert frame to a tensor
                # frame_t = frame.transpose((2, 0, 1))
                # frame_t = torch.from_numpy(frame_t)
                frame_t = tvtransfunc.to_tensor(frame_800)
                frame_t.to(device)

                # do inference on the frame
                print('frame inference {}'.format(i))
                confidence_thresh = 0.6
                iou_thresh = 0.5
                model.eval()

                pred, keep = get_prediction_image(model, frame_t, confidence_thresh, iou_thresh, device, CLASS_NAMES)
                img = show_groundtruth_and_prediction_bbox(image=frame_t,
                                                           predictions=pred,
                                                           keep=keep)
                # save image for now
                # imgname = os.path.join('output', 'webcam','fasterrcnn-serratedtussock-video-' + str(i).zfill(3) + '.png')

                # write image to video
                img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                vid_out.write(img)

                # increment frame counter
                i += 1

                cv.imshow('frame', img)

                end_time = time.time()

                sec = end_time - start_time
                print('cycle time: {} sec'.format(sec))

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                print('Error: ret is not True')  # TODO should probably be an assert or Raise
                break

        cap.release()
        vid_out.release()
        cv.destroyAllWindows()



    print('end of inference.py')
    import code
    code.interact(local=dict(globals(), **locals()))