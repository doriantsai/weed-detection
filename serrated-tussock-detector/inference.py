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

def matplotlib_imshow(img):
    # image in, probably as a tensor
    # want to show image using plt.imshow(img)
    # img = img / 2 + 0.5     # unnormalize
    imgnp = img.cpu().numpy()
    plt.imshow(np.transpose(imgnp, (1, 2, 0)))
    return plt


def show_single_bbox(img, bb, color='red'):
    # put bounding box onto image
    # bb is a numpy array or tensor?
    # TODO convert to opencv -> cv.rectangle
    bb = np.array(bb.cpu(), dtype=np.float32)
    rect = mpp.Rectangle((bb[0], bb[1]),
                          bb[2] - bb[0],
                          bb[3] - bb[1],
                          color=color,
                          fill=False,
                          linewidth=3)


def show_image_bbox(image, sample, color='blue'):
    # show image and bounding box together
    # matplotlib_imshow(image)
    # TODO convert to opencv -> cv.rectangle
    imgnp = image.cpu().numpy()

    fig, ax = plt.subplots(1)

    ax.imshow(np.transpose(imgnp, (1, 2, 0)))

    boxes = sample['boxes']
    nbb, _ = boxes.size()

    print(imgnp.shape)

    for i in range(nbb):
        print('plot box {}'.format(i))
        bb = np.array(boxes[i, :].cpu(), dtype=np.float32)
        print(bb)  # [xmin, ymin, xmax, ymax]
        rect = mpp.Rectangle((bb[0], bb[1]),
                             bb[2] - bb[0],
                             bb[3] - bb[1],
                             color=color,
                             fill=False,
                             linewidth=3)
        ax.add_patch(rect)

        # plt.gca().add_patch(show_single_bbox(image, boxes[i, :]))

    return fig, ax


def show_groundtruth_and_prediction_bbox(image,
                         sample=None,
                         predictions=None,
                         keep=None,
                         sample_color=(0, 0, 255), #RGB
                         predictions_color=(255, 0, 0), #RGB
                         transpose_channels=True,
                         transpose_color_channels=False):
    # show image, sample/gt bounding box, and predictions bounding box together

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
                                     thickness=15)
                # ax.add_patch(rect)

    # second, plot predictions
    if not predictions is None:

            # if not(keep is None):
            #     boxes_pd = predictions['boxes']
            #     import code
            #     code.interact(local=dict(globals(), **locals()))
            #     boxes_pd = boxes_pd[keep]
            # else:
        boxes_pd = predictions['boxes']
        boxes_score = predictions['scores']

        # import code
        # code.interact(local=dict(globals(), **locals()))

        if len(boxes_pd) > 0:
            for i in range(len(boxes_pd)):
                bb = np.array(boxes_pd[i], dtype=np.float32)  # TODO might just specify np.int16?
                imgnp = cv.rectangle(imgnp,
                                    (int(bb[0]), int(bb[1])),
                                    (int(bb[2]), int(bb[3])),
                                    color=predictions_color,
                                    thickness=5)
                # rect = mpp.Rectangle((bb[0], bb[1]),
                #                       bb[2] - bb[0],
                #                       bb[3] - bb[1],
                #                     color=predictions_color,
                #                     fill=False,
                #                     linewidth=1)
                # ax.add_patch(rect)

                # add text to top left corner of bounding box
                sc = format(boxes_score[i], '.4f') # show 4 decimal places
                # ax.annotate('{}: {}'.format(i, sc),
                #              (bb[0], bb[1]),
                #              color=predictions_color,
                #              fontsize=12)
                cv.putText(imgnp,
                          '{}: {}'.format(i, sc),
                          (int(bb[0]), int(bb[1])),
                          fontFace=cv.FONT_HERSHEY_COMPLEX,
                          fontScale=2,
                          color=predictions_color,
                          thickness=3)

    return imgnp





def infer_dataset(model,
                  dataset,
                  confidence_threshold,
                  iou_threshold,
                  save_folder_name,
                  device,
                  class_names,
                  save_image_name=None):
    """
    infer model on entire dataset
    """

    model.eval()
    model.to(device)
    for image, sample in dataset:
        image_id = sample['image_id'].item()
        pred, keep = get_prediction_image(model,
                                          image,
                                          confidence_threshold,
                                          iou_threshold,
                                          device,
                                          class_names)
        image_marked = show_groundtruth_and_prediction_bbox(image,
                                                            sample,
                                                            pred,
                                                            keep)
        if not os.path.isdir(os.path.join('output', save_folder_name)):
            os.mkdir(os.path.join('output', save_folder_name))
        if save_image_name is None:
            # dsname = f'{dataset=}'.split('=')[0]
            image_name = os.path.join('output', save_folder_name,
                                      'fasterrcnn-serratedtussock-' + 'dataset-' + str(image_id) + '.png')
        else:
            image_name = os.path.join('output', save_folder_name, save_image_name + '-' + str(image_id) + '.png')

        image_marked = cv.cvtColor(image_marked, cv.COLOR_RGB2BGR)
        cv.imwrite(image_name, image_marked)
        winname = 'testing'
        cv.namedWindow(winname, cv.WINDOW_NORMAL)
        cv.imshow(winname, image_marked)
        cv.waitKey(2000)
        cv.destroyWindow(winname)





# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    # test the detector model by plotting a sample image

    # setup device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load model

    # model = build_model(nclasses)

    num_classes = 2
    CLASS_NAMES = ["_background_", "serrated tussock"]
    # load instance segmentation model pre-trained on coco:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    save_name = 'fasterrcnn-serratedtussock-4'
    save_path = os.path.join('output', save_name, save_name + '.pth')
    # model.load_state_dict(torch.load(save_path))
    model.load_state_dict(torch.load(save_path))

    # setup dataset
    root_dir = os.path.join('SerratedTussockDataset')
    json_file = os.path.join('Annotations', 'via_region_data.json')

    # tforms = Compose([Rescale(800),
    #                     RandomHorizontalFlip(0.5),
    #                     ToTensor()])

    # dataset = SerratedTussockDataset(root_dir=root_dir,
    #                                     json_file=json_file,
    #                                     transforms=tforms)

    # split into training, validation and testing
    # nimg = len(dataset)
    # ntrain = 20
    # TODO obviously, we'll need more data/images in future runs,
    # but we're just gunning for the pipeline at the moment
    # dataset_train, dataset_test = torch.utils.data.random_split(dataset, [ntrain, nimg - ntrain])

    # batch_size = 1
    # num_workers = 0
    # dataloader_test = torch.utils.data.DataLoader(dataset_test,
    #                                                 batch_size=batch_size,
    #                                                 shuffle=False,
    #                                                 num_workers=num_workers,
    #                                                 collate_fn=utils.collate_fn)

    # load stuff:
    data_save_path = os.path.join('.', 'output', save_name, save_name + '.pkl')
    with open(data_save_path, 'rb') as f:
        dataset = pickle.load(f)
        dataset_train = pickle.load(f)
        dataset_val = pickle.load(f)
        dataset_test = pickle.load(f)
        dataloader_test = pickle.load(f)
        dataloader_train = pickle.load(f)
        dataloader_val = pickle.load(f)
        hp = pickle.load(f)


    # print up filenames in dataset_test, so we can print them
    # for image, sample in dataset_test:



    # first, plot a sample from the training set (should be overfit)
    # model.to(device)
    INFER_ON_TRAINING = False
    if INFER_ON_TRAINING:
        print('training set')
        model.eval()
        imgs_train, smps_train = next(iter(dataloader_train))
        confidence_thresh = 0.9
        iou_thresh = 0.5
        bs = len(imgs_train)
        model.to(device)
        # imgs_train.to(device)
        for i in range(bs):
            print(i)
            # figi, axi = show_image_bbox(imgs_train[i], smps_train[i])
            pred, keep = get_prediction_image(model, imgs_train[i], confidence_thresh, iou_thresh, device, CLASS_NAMES)
            img = show_groundtruth_and_prediction_bbox(imgs_train[i],
                                                             smps_train[i],
                                                             pred,
                                                             keep)
            imgname = os.path.join('output', 'fasterrcnn-serratedtussock-train-' + str(i) + '.png')
            # plt.savefig()
            # plt.show()
            # time.sleep(1) # sleep for 1 second
            # plt.close(figi)
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            cv.imwrite(imgname, img)
            winname = 'training'
            cv.namedWindow(winname, cv.WINDOW_GUI_NORMAL)
            # img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            cv.imshow(winname, img)
            cv.waitKey(2000)  # wait for 2 sec
            cv.destroyWindow(winname)


    # finally, evaulate on the whole dataset
    confidence_thresh = 0.8
    iou_thresh = 0.5

    # test loading images from dataloader_test, dataloader_train and dataloader_val

    # import code
    # code.interact(local=dict(globals(), **locals()))
    # mt_eval = evaluate(model,
    #                    dataloader_test,
    #                    device=device,
    #                    conf=confidence_thresh,
    #                    iou=iou_thresh,
    #                    class_names=CLASS_NAMES)

    INFER_ON_TEST = True
    if INFER_ON_TEST:
        print('testing set')
        # model.eval()

        # this code snippet only runs on a single iteration of the dataset
        # imgs_test, smps_test = next(iter(dataloader_test))
        confidence_thresh = 0.8
        iou_thresh = 0.5
        infer_dataset(model,
                      dataset_test,
                      confidence_thresh,
                      iou_thresh,
                      save_name,
                      device,
                      CLASS_NAMES,
                      'dataset-test')
        # bs = len(imgs_test)
        # model.to(device)
        # imgs_test.to(device)
        # for i in range(bs):
        #     pred, keep = get_prediction(model, imgs_test[i], confidence_thresh, iou_thresh, device, CLASS_NAMES)
        #     img = show_groundtruth_and_prediction_bbox(imgs_test[i], smps_test[i], pred, keep)
        #     imgname = os.path.join('output', save_name, 'fasterrcnn-serratedtussock-test-' + str(i) + '.png')
        #     img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        #     cv.imwrite(imgname, img)
        #     winname = 'testing'
        #     cv.namedWindow(winname, cv.WINDOW_NORMAL)

        #     cv.imshow(winname, img)
        #     cv.waitKey(2000)  # wait for 2 sec
        #     cv.destroyWindow(winname)

        # this code runs on the entire test dataset
        # for image, sample in dataset_test:
        #     image_id = sample['image_id']
        #     print('image_id: {}'.format(image_id))
        #     pred, keep = get_prediction(model,
        #                                 image,
        #                                 confidence_thresh,
        #                                 iou_thresh,
        #                                 device,
        #                                 CLASS_NAMES)

        #     image_marked = show_groundtruth_and_prediction_bbox(image,
        #                                                         sample,
        #                                                         pred,
        #                                                         keep)

        #     if not os.path.isdir(os.path.join('output', save_name)):
        #         os.mkdir(os.path.join('output', save_name))
        #     image_name = os.path.join('output', save_name, 'fasterrcnn-serratedtussock-test-' + str(image_id) + '.png')
        #     image_marked = cv.cvtColor(image_marked, cv.COLOR_RGB2BGR)
        #     cv.imwrite(image_name, image_marked)
        #     winname = 'testing'
        #     cv.namedWindow(winname, cv.WINDOW_NORMAL)
        #     cv.imshow(winname, image_marked)
        #     cv.waitKey(2000)
        #     cv.destroyWindow(winname)





   # --------------- #
   # try joh's images

    INFER_ON_JOH = False
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
                      'dataset-joh')

        # imgs_joh, smps_joh = next(iter(dataloader_joh))
        # confidence_thresh = 0.6
        # iou_thresh = 0.5
        # model.eval()
        # model.to(device)
        # for i in range(bs):
        #     pred, keep = get_prediction(model, imgs_joh[i], confidence_thresh, iou_thresh, device, CLASS_NAMES)
        #     img = show_groundtruth_and_prediction_bbox(imgs_joh[i], smps_joh[i], pred, keep)
        #     # plt.savefig(os.path.join('output', 'fasterrcnn-serratedtussock-testjoh-' + str(i) + '.png'))
        #     imgname = os.path.join('output', 'fasterrcnn-serratedtussock-testjoh-' + str(i) + '.png')
        #     # plt.savefig()
        #     # plt.show()
        #     # time.sleep(1) # sleep for 1 second
        #     # plt.close(figi)
        #     img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        #     cv.imwrite(imgname, img)
        #     winname = 'testingjoh'
        #     cv.namedWindow(winname, cv.WINDOW_NORMAL)

        #     cv.imshow(winname, img)
        #     cv.waitKey(2000)  # wait for 2 sec
        #     cv.destroyWindow(winname)



    # read in video using webcam.py's grab_webcam_video
    # run on a single frame within the video
    INFER_ON_VIDEO = False
    if INFER_ON_VIDEO:
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

                # import code
                # code.interact(local=dict(globals(), **locals()))
                # if torch.cuda.is_available():
                #     imgs = imgs.to(device)
                #     lbls = lbls.to(device)

                pred, keep = get_prediction_image(model, frame_t, confidence_thresh, iou_thresh, device, CLASS_NAMES)
                img = show_groundtruth_and_prediction_bbox(image=frame_t,
                                                           predictions=pred,
                                                           keep=keep)
                # save image for now
                # imgname = os.path.join('output', 'webcam','fasterrcnn-serratedtussock-video-' + str(i).zfill(3) + '.png')

                img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

                # cv.imwrite(imgname, img)



                # TODO string must have 0's in front - format the number, so that image files are ordered sequentially
                # plt.savefig()
                # plt.close(figi)
                # plt.show()
                # TODO close the figure somehow

                # TODO # write to a new video

                vid_out.write(img)

                # increment frame counter
                i += 1

                # convert imshow frame to BGR (because OpenCV)
                # TODO show annotations from fig i/axi?

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




    import code
    code.interact(local=dict(globals(), **locals()))