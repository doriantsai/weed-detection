#! /usr/bin/env python

"""

Test script to export Pytorch model to Onnx format
https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
"""

 
import torch
import torch.onnx
import os
from weed_detection.WeedModel import WeedModel
from weed_detection.WeedDatasetPoly import WeedDatasetPoly
from weed_detection.PreProcessingToolbox import PreProcessingToolbox
import numpy as np


# pytorch model location:
model_path = os.path.join('/home/agkelpie/Code/weed-detection/output',
                          '2021-03-25_MFS_Tussock_v0_2021-09-16_08_55',
                          '2021-03-25_MFS_Tussock_v0_2021-09-16_08_55_epoch25.pth')
# 2021-03-26_MFS_Multiclass_v1_2022-08-17_16_19/2021-03-26_MFS_Multiclass_v1_2022-08-17_16_19.pth'

# load model
WM = WeedModel()
torch_model = WM.load_model(model_path, annotation_type='poly', num_classes=2)
model_name = '2021-03-25_MFS_Tussock_v0_2021-09-16_08_55'
WM.set_model_name(model_name)
WM.set_model_path(model_path)
WM.set_model_folder(model_name)

# note: these numbers are from pipeline_multiclass.py
batch_size = 1
model_input = torch.randn(batch_size, 3, int(1028), int(1232), requires_grad=False)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_input = model_input.to(device)

# import code
# code.interact(local=dict(globals(), **locals()))

# export model to onnx
torch.onnx.export(torch_model, 
                  model_input, 
                  (model_name + ".onnx"), 
                  export_params=True,
                  opset_version=13, # worked with 11, trying 13
                  do_constant_folding=False,
                  input_names=['images'], 
                  output_names = ['predictions'])

# test model runs in Onnx?

print('model checker?')

import onnx
onnx_model = onnx.load((model_name + ".onnx"))
onnx.checker.check_model(onnx_model)




# test model conversion from pytorch to Onnx
# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
# following the above link

import onnxruntime

model_name = '2021-03-25_MFS_Tussock_v0_2021-09-16_08_55.onnx'
ort_session = onnxruntime.InferenceSession(model_name)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(model_input)}
ort_outs = ort_session.run(None, ort_inputs)


# ==========================================================================
# compare ONNX runtime and pytorch results:
# np.testing.assert_allclose(to_numpy())


print('test with real input: our model (pytorch)')


# some actual test input:
img_name = '/home/agkelpie/Code/weed-detection/examples/images/mako___2021-3-26___10-53-40-932.png'
import cv2 as cv
img = cv.imread(img_name, cv.IMREAD_COLOR)

img_out, pred = WM.infer_image(img, 
                               imsave=True,
                               save_dir='.',
                               image_name=(os.path.basename(img_name)[:-4] + '_pred'),
                               image_color_format='BGR')

print(f'img_out type = {type(img_out)}')
print(f'img_out size = {img_out.shape}')
print(f'img_out path = {os.path.basename(img_name)}') # assuming save_dir, image_name same as above

print(pred)

print('test with real input: converted model (onnx))')

import code
code.interact(local=dict(globals(), **locals()))

# setup transformation, img resize, tensor
from PIL import Image
import torchvision.transforms as transforms
img_resize = transforms.Resize([1028, 1232])
to_tensor = transforms.ToTensor()

img_pil = Image.fromarray(img)
img_on = img_resize(img_pil) # only works on PIL images
img_on = to_tensor(img_on)
img_on = img_on.unsqueeze(0) # add in the batch size front element


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
img_on = img_on.to(device)
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_on)}
ort_outs = ort_session.run(None, ort_inputs)

print(ort_outs)

import code
code.interact(local=dict(globals(), **locals()))
