import os
import numpy as np
import onnx
import nnvm
import tvm
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import glob
import time

# load model
onnx_model = onnx.load('alexnet.proto')

# load data
img_list = np.load('input_list.npy')
val = np.loadtxt('target_list.txt')

input_shape = (1, 3, 224, 224)
output_shape = (1, 1000)

# NNVM compiler
sym, params = nnvm.frontend.from_onnx(onnx_model)
import nnvm.compiler
target = 'cuda'
input_name = sym.list_input_names()[0]
shape_dict = {input_name: input_shape}
graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)

# TVM compiler
from tvm.contrib import graph_runtime
ctx = tvm.gpu(0)
dtype = 'float32'
m = graph_runtime.create(graph, lib, ctx)


# compute accuracy and throughput
accum_time = 0.
top1_correct_count = 0
top5_correct_count = 0

#for i, (input, target) in enumerate(val_loader):
for i in range(len(val)):
    img_ = img_list[i]
    #m.set_input('input_1', tvm.nd.array(img_.astype('float32')))
    m.set_input(**params)
    start_ = time.time()
    m.set_input(0, tvm.nd.array(img_.astype('float32')))
    m.run()
    tvm_out = m.get_output(0, tvm.nd.empty(output_shape, 'float32')).asnumpy()
    end_ = time.time()
    accum_time = accum_time + (end_ - start_)

    if int(val[i]) == tvm_out.argmax():
        top1_correct_count = top1_correct_count + 1
    if int(val[i]) in (-tvm_out).argsort()[0][:5]:
        top5_correct_count = top5_correct_count + 1
    #tvm_out = m.get_output(0, tvm.nd.empty(output_shape, 'float32')).asnumpy()

print('top1 accuracy : ', top1_correct_count * 1.0 / len(img_list))
print('top5 accuracy : ', top5_correct_count * 1.0 / len(img_list))
print('throughput per sec : ', len(img_list) * 1.0 / accum_time)




# test ONNX converter
from onnx_tf.backend import prepare
tf_rep = prepare(onnx_model)
top1 = 0
top5 = 0
for i in range(len(val)):
    img_ = img_list[i]
    pred_ = tf_rep.run(img_)._0
    if int(val[i]) == pred_.argmax():
        top1 = top1 + 1
    if int(val[i]) in (-pred_).argsort()[0][:5]:
        top5 = top5 + 1
print('onnx-tf top1 accuracy : ', top1 * 1.0 / len(img_list))
print('onnx-tf top5 accuracy : ', top5 * 1.0 / len(img_list))
