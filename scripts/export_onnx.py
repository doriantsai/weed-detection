#! /usr/bin/env python

"""

Test script to export Pytorch model to Onnx format
https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
"""

 
import torch
import torch.onnx
import os
from weed_detection.WeedModel import WeedModel
from weed_detection.PreProcessingToolbox import PreProcessingToolbox


# pytorch model location:
model_path = '/home/agkelpie/Code/weed-detection/output/2021-03-26_MFS_Multiclass_v1_2022-08-17_16_19/2021-03-26_MFS_Multiclass_v1_2022-08-17_16_19.pth'

# load model
WM = WeedModel()
torch_model = WM.load_model(model_path, annotation_type='poly', num_classes=3)
model_name = '2021-03-26_MFS_Multiclass_v1_2022-08-17_16_19'
WM.set_model_name(model_name)
WM.set_model_path(model_path)
WM.set_model_folder(model_name)

# note: these numbers are from pipeline_multiclass.py
batch_size = 10
model_input = torch.randn(batch_size, 3, int(1024), int(1024), requires_grad=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_input = model_input.to(device)

# import code
# code.interact(local=dict(globals(), **locals()))

# export model to onnx
torch.onnx.export(torch_model, 
                  model_input, 
                  "model_onnx.onnx", 
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=False,
                  input_names=['images'], 
                  output_names = ['predictions'])

# test model runs in Onnx?

