# just conv



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


model_name = "conv"

def proto_val_to_dimension_tuple(proto_val):
	return tuple([dim.dim_value for dim in proto_val.type.tensor_type.shape.dim])		

def save_model(graph, model_name):
	model = onnx.helper.make_model(graph, producer_name='onnx-compiler-test')

	model.graph.value_info.append(make_tensor_value_info(model.graph.input[0].name, TensorProto.FLOAT, proto_val_to_dimension_tuple(model.graph.input[0])))
	model.graph.value_info.append(make_tensor_value_info(model.graph.output[0].name, TensorProto.FLOAT, proto_val_to_dimension_tuple(model.graph.output[0])))	
	print(model.graph.value_info)

	checker.check_model(model)
	onnx.save(model, 'models/' + model_name + '.onnx')

	# load and infer
	nm = onnx.load('models/' + model_name + '.onnx')
	inferred_nm = onnx.shape_inference.infer_shapes(nm)
	print("**************************************************************************")
	print(inferred_nm.graph.value_info)


def add_param(name, shape, param_list, type=TensorProto.FLOAT, value=None):
	cnt = 1
	for val in shape: cnt*=val

	if value==None:
		value = np.load('models/'+ model_name +'/'+ name + '.npy').reshape(cnt).tolist()
	
	param = helper.make_tensor(name, type, shape, value)
	param_list.append(param)

	return param_list


X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 32, 32])
out  = helper.make_tensor_value_info('out', TensorProto.FLOAT, [])

param_list = []
add_param('Wc', [6,3,5,5], param_list)

out_node = helper.make_node("Conv", ['X', 'Wc'], ['out']) 

graph = helper.make_graph(
        [out_node],
        model_name,
        [X],
        [out],
        param_list
    )

save_model(graph, model_name)
