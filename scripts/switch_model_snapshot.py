#! /usr/bin/env python

""" script to switch snapshots """

import os
from weed_detection.WeedModel import WeedModel as WM

# init object
# model path
# model name
# load model/etc

# manually specify new epoch
# call object/function

Tussock = WM()
model_name = 'tussock_test_2021-05-16_16_13'
save_model_path = os.path.join('output',
                               model_name,
                               model_name + '.pth')
Tussock.load_model(save_model_path)
Tussock.set_model_name(model_name)
Tussock.set_model_path(save_model_path)
Tussock.set_model_epoch(100)

print('old model path = {}'.format(Tussock.get_model_path()))
epoch = 60
Tussock.set_snapshot(epoch)
print('new model path = {}'.format(Tussock.get_model_path()))
print(Tussock.get_model_epoch())