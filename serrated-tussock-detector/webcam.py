#! /usr/bin/env python

# code to read in images from webcam and save it as a video

import os
import cv2 as cv
import time

# import torchvision


""" grab image, show it, save it """
def grab_webcam_image(outpath, outresolution=None, num_images=5):
    device_id = 0
    win_name = 'webcam image'

    cv.namedWindow(win_name)
    vc = cv.VideoCapture(device_id)

    if outresolution is not None:
        # set custom out resolution 
        print('TODO: setting resolution')

    # get first frame
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    # TODO iterate for num_images?

    while rval:
        # read image
        rval, frame = vc.read()

        # show image
        cv.imshow(win_name, frame)
        key = cv.waitKey(20)
        if key == 27:  # exit on ESC
            break

    # save image to outpath
    im_name = os.path.join(outpath, 'image0.png')
    cv.imwrite(im_name, frame)
    
    cv.destroyWindow(win_name)
    vc.release()


""" capture video, replay it, save it """
def grab_webcam_video(outpath, fps=3, mirror=False):
    device_id = 0
    win_name = 'webcam video'

    cv.namedWindow(win_name)
    vc = cv.VideoCapture(0)
    
    # get width/height of frame
    w = vc.get(cv.CAP_PROP_FRAME_WIDTH)
    h = vc.get(cv.CAP_PROP_FRAME_HEIGHT)

    # define video writer object
    vw = cv.VideoWriter_fourcc(*'XVID')

    out = cv.VideoWriter(outpath, vw, fps, (int(w), int(h)))

    MAX_FRAMES = 100
    iframe = 0
    start = time.time()
    while (vc.isOpened() and iframe < MAX_FRAMES):
        # capture frame-by-frame?

        # start time
        
        ret, frame = vc.read()
        

        if ret is True:
            if mirror is True:
                # mirror output of video frame
                frame = cv.flip(frame, 1)

            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # save for video
            out.write(frame_rgb)

            # display resulting frame
            cv.imshow(win_name, frame)
        else:
            break

        if cv.waitKey(1) & 0xFF == ord('q'):  # if q is pressed then stop
            break

        # to stop duplicate images
        iframe += 1

    # when everything is done, release capture


    vc.release()
    out.release()
    # should be able to destroy just win_name, but got an error?
    cv.destroyAllWindows()

    # end time 
    end = time.time()

    # time elapsed:
    sec = end-  start
    print('time in {0} seconds'.format(sec))

    # fps
    fps = iframe / sec
    print('fps = {}'.format(fps))

 


  


# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    outpath = os.path.join('output', 'webcam')


    # grab_webcam_image_cv(outpath)

    vid_name = os.path.join(outpath, 'video.avi')
    grab_webcam_video(vid_name, fps=27)


# def grab_webcam_video(outpath, fps, outresolution, mirror=False):

#     device_id = 0
#     win_name = 'webcam preview'

    # reader = torchvision.io.VideoReader(outpath, "video")
    # reader_md = reader.get_metadata()

    # print(reader_md["video"]["fps"])

    # video.set_current_stream("video:0")

    

# window_name = 'webcam preview'
# cv.namedWindow(window_name)
# vc = cv.VideoCapture(0)

# if vc.isOpened():  # try to get first frame
#     rval, frame = vc.read()
# else:
#     rval = False


# while rval:
#     cv.imshow(window_name, frame)
#     rval, frame = vc.read()
#     key = cv.waitKey(20)
#     if key == 27:  # exit on ESC
#         break

# cv.destroyWindow(window_name)
# vc.release()

