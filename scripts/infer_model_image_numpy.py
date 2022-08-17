
# test with numpy array input

import cv2 as cv
import numpy
import os
from weed_detection.WeedModel import WeedModel


# setup weed model
Tussock = WeedModel()

model_name = '2021-03-25_MFS_Tussock_v0_2021-09-16_08_55'
model_path = os.path.join('/home/agkelpie/Code/weed-detection/scripts/output', model_name, model_name + '.pth')
Tussock.load_model(model_path)
Tussock.set_model_name(model_name)
Tussock.set_snapshot(25)

img_dir = os.path.join('/home/agkelpie/Data/2021-03-25_MFS_Tussock_v0/images')
img_list = os.listdir(img_dir)
img_name = img_list[0]
img_path = os.path.join(img_dir, img_name)

img = cv.imread(img_path)
img_out, pred = Tussock.infer_image(img, imshow=False, imsave=True, save_dir='.')

import code
code.interact(local=dict(globals(), **locals()))
