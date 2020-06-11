
# This is to create Lenet CIFAR-10 onnx model


import onnx
from onnx import helper, numpy_helper, checker
import unittest
from onnx import TensorProto
import numpy as np
import subprocess
from datetime import date
import time
import hashlib
import onnx.shape_inference
from onnx.helper import make_tensor_value_info


model_name = "lenet"

def _get_rnd_float32(low=-1.0, high=1.0, shape=None):
	output = np.random.uniform(low, high, shape)
	cnt = 1
	for val in shape: cnt*=val
	if shape == None:
		return np.float32(output)
	else:
		return output.astype(np.float32).reshape(cnt).tolist()

def proto_val_to_dimension_tuple(proto_val):
	return tuple([dim.dim_value for dim in proto_val.type.tensor_type.shape.dim])		

def save_model(graph, model_name):
	model = onnx.helper.make_model(graph, producer_name='onnx-compiler-test')



	model.graph.value_info.append(make_tensor_value_info(model.graph.input[0].name, TensorProto.FLOAT, proto_val_to_dimension_tuple(model.graph.input[0])))
	model.graph.value_info.append(make_tensor_value_info(model.graph.output[0].name, TensorProto.FLOAT, proto_val_to_dimension_tuple(model.graph.output[0])))	

	for init_vals in model.graph.initializer:
		model.graph.value_info.append(make_tensor_value_info(init_vals.name, TensorProto.FLOAT, tuple(init_vals.dims)))	

	# print(model.graph.value_info)
	# print("**************************************************************************")
	inferred_model = onnx.shape_inference.infer_shapes(model)

	print(inferred_model.graph.value_info)

	checker.check_model(model)

	onnx.save(model, 'models/' + model_name + '.onnx')

def add_param(name, shape, param_list, type=TensorProto.FLOAT, value=None, permute=None):
	cnt = 1
	for val in shape: cnt*=val

	if value==None:
		value = np.load('models/'+ model_name +'/'+ name + '.npy')
		
		# reorder the input param e.g. in case of conv filter
		if permute!= None:
			# [3, 2, 0, 1] for filter
			stored_shape = [0,0,0,0]
			for i in range(len(shape)): stored_shape[permute[i]] = shape[i]
			value = value.reshape(stored_shape)
			value = np.transpose(value, permute)

		value = value.reshape(cnt).tolist()
	
	param = helper.make_tensor(name, type, shape, value)
	param_list.append(param)
	return param_list

# lenet on cifar-10
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3072, 1])
out  = helper.make_tensor_value_info('out', TensorProto.INT64, [1])

param_list = []

#[5,5,3,6] spatial_filter_shape + [in_channels, out_channels].
# [CO CI H W]
add_param('Wc1', [6,3,5,5], param_list, permute=(3, 2, 0, 1))
# [5,5,6,12]
add_param('Wc2', [12,6,5,5], param_list, permute=(3, 2, 0, 1))
add_param('Wf1', [768,120], param_list)
add_param('Wf2', [120,84], param_list)
add_param('Wf3', [84,10], param_list)
add_param('Bc1', [6], param_list)
add_param('Bc2', [12], param_list)
add_param('Bf1', [120], param_list)
add_param('Bf2', [84], param_list)
add_param('Bf3', [10], param_list)

# shape params
add_param('X2_shape', [4], param_list, type=TensorProto.INT64, value=[1,3,32,32])
add_param('Hc2PP_shape', [2], param_list, type=TensorProto.INT64, value=[1,768])


X2 = helper.make_node("Reshape", ['X', 'X2_shape'], ['X2']) 

Hc1_conv = helper.make_node("Conv", ['X2', 'Wc1', 'Bc1'], ['Hc1_conv'], pads=[2,2,2,2]) 

Hc1 = helper.make_node("Relu", ['Hc1_conv'], ['Hc1'])
Hc1P =  helper.make_node("MaxPool", ['Hc1'], ['Hc1P'], kernel_shape=[2,2], strides=[2,2])

Hc2_conv = helper.make_node("Conv", ['Hc1P', 'Wc2', 'Bc2'], ['Hc2_conv'], pads=[2,2,2,2])
Hc2 = helper.make_node("Relu", ['Hc2_conv'], ['Hc2'])
Hc2P = helper.make_node("MaxPool", ['Hc2'], ['Hc2P'], kernel_shape=[2,2], strides=[2,2])

# this is not just rehape but also reorder
Hc2PP = helper.make_node("Reshape", ['Hc2P', 'Hc2PP_shape'], ['Hc2PP'])

Hf1_mul = helper.make_node("Gemm", ['Hc2PP', 'Wf1', 'Bf1'], ['Hf1_mul'])
Hf1 = helper.make_node("Relu", ["Hf1_mul"], ["Hf1"])

Hf2_mul = helper.make_node("Gemm", ['Hf1', 'Wf2', 'Bf2'], ['Hf2_mul'])
Hf2 = helper.make_node("Relu", ["Hf2_mul"], ["Hf2"]) 

Hf3 = helper.make_node("Gemm", ['Hf2', 'Wf3', 'Bf3'], ['Hf3'])
out_node = helper.make_node("ArgMax", ['Hf3'], ['out'], axis=1, keepdims=0)

node_list = [X2, Hc1_conv, Hc1, Hc1P, Hc2_conv, Hc2, Hc2P, Hc2PP, Hf1_mul, Hf1, Hf2_mul, Hf2, Hf3, out_node]

graph = helper.make_graph(
        node_list,
        model_name,
        [X],
        [out],
        param_list,
    )

save_model(graph, model_name)
