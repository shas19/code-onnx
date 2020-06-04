
# This is to create Lenet CIFAR-10 onnx model


import onnx
from onnx import helper, numpy_helper
import unittest
from onnx import TensorProto
import numpy as np
import subprocess
import common
from datetime import date
import time
import hashlib


def _get_rnd_float32(low=-1.0, high=1.0, shape=None):
	output = np.random.uniform(low, high, shape)
	cnt = 1
	for val in shape: cnt*=val
	if shape == None:
		return np.float32(output)
	else:
		return output.astype(np.float32).reshape(cnt).tolist()

def save_model(graph, name):
	model = onnx.helper.make_model(graph, producer_name='onnx-compiler-test')
	onnx.save(model, 'models/' + name + '.onnx')

def add_param(name, shape, param_list):
	param_val = np.load(name+'.npy')
	param = helper.make_tensor(name, TensorProto.FLOAT, shape, param_val)
	param_list.append(param)
	return param_list

# lenet on cifar-10
name = "lenet"
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3072, 1])
state_out  = helper.make_tensor_value_info('state_out', TensorProto.FLOAT, [])

param_list = []
add_param('Wc1', [5,5,3,6], param_list)
add_param('Wc2', [5,5,6,12], param_list)
add_param('Wf1', [768,120], param_list)
add_param('Wf2', [120,84], param_list)
add_param('Wf3', [84,10], param_list)
add_param('Bc1', [6], param_list)
add_param('Bc2', [12], param_list)
add_param('Bf1', [120], param_list)
add_param('Bf2', [84], param_list)
add_param('Bf3', [10], param_list)


X2 = helper.make_node("Reshape", ['X'], ['X2'], shape=[1, 32, 32, 3]) 

Hc1_conv = helper.make_node("Conv", ['X2', 'Wc1', 'Bc1'], ['Hc1_conv']) 
Hc1 = helper.make_node("Relu", ['Hc1_conv'], ['Hc1'])
Hc1P =  helper.make_node("Maxpool", ['Hc1'], ['Hc1P'])

Hc2_conv = helper.make_node("Conv", ['Hc1P', 'Wc2', 'Bc2'], ['Hc2_conv'])
Hc2 = helper.make_node("Relu", ['Hc2_conv'], ['Hc2P'])
Hc2p = helper.make_node("Maxpool", ['Hc2'], ['Hc2P'])

# this is not just rehape but also reorder
Hc2PP = helper.make_node("Reshape", ['Hc2P'], ['Hc2'])


Hf1_mul = helper.make_node("Gemm", ['Hc2PP', 'Wf1', 'Bf1'], ['Hf2'])
Hf1 = helper.make_node("Relu", ["Hf1_mul"], ["Hf1"])

Hf2_mul = helper.make_node("Gemm", ['Hf1', 'Wf2', 'Bf2'], ['Hf2'])
Hf2 = helper.make_node("Relu", ["Hf2_mul"], ["Hf2"]) 

Hf3 = helper.make_node("Gemm", ['Hf2', 'Wf3', 'Bf3'], ['Hf3'])
out = helper.make_node("Argmax", ['Hf3'], ['out'])

node_list = [X2, Hc1_conv, Hc1, Hc1P, Hc2_conv, Hc2, Hc2P, Hc2PP, Hf1_mul, Hf1, Hf2_mul, Hf2, Hf3, out]

graph = helper.make_graph(
        node_list,
        name,
        [X],
        [state_out],
        param_list
    )

save_model(graph, name)

# let X2    = reshape(X, (1, 32, 32, 3), (1, 2))     in
# let Hc1   = relu   ((X2    conv Wc1) <+> Bc1) in
# let Hc1P  = maxpool(Hc1, 2)              in
# let Hc2   = relu   ((Hc1P conv Wc2) <+> Bc2) in
# let Hc2P  = maxpool(Hc2, 2)              in
# let Hc2PP = reshape(Hc2P, (1, 768), (1, 4, 2, 3))       in
# let Hf1   = relu   ((Hc2PP * Wf1) <+> Bf1) in
# let Hf2   = relu   ((Hf1   * Wf2) <+> Bf2) in
# let Hf3   =        ((Hf2   * Wf3) <+> Bf3) in
# argmax(Hf3)