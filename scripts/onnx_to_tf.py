# /usr/bin/env python

# text to convert onnx model to tensorflow

import onnx

from onnx_tf.backend import prepare

onnx_model_name = '2021-03-25_MFS_Tussock_v0_2021-09-16_08_55.onnx'
onnx_model = onnx.load(onnx_model_name) # load onnx model
tf_rep  = prepare(onnx_model) # prepare tf representation
tf_rep.export_graph('test.pb')
print('done exporting onnx model to tf')


