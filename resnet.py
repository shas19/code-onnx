
import onnx
from onnx import helper, numpy_helper, checker
from onnx import TensorProto
import onnx.shape_inference
from onnx.helper import make_tensor_value_info

model_name = 'resnet-18'

def proto_val_to_dimension_tuple(proto_val):
	return tuple([dim.dim_value for dim in proto_val.type.tensor_type.shape.dim])	

def save_model(model):

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

model = onnx.load('models/' + model_name + '/' + model_name + '.onnx')

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [150528, 1])
data_shape = helper.make_tensor_value_info('data_shape', TensorProto.INT64, [4])
data_shape_param = helper.make_tensor('data_shape', TensorProto.INT64, [4], [1,3,224,224])

model.graph.input.pop(0) 
model.graph.input.insert(0, X) 
model.graph.input.insert(1, data_shape)

model.graph.initializer.extend([data_shape_param])

print(model.graph.input[0])

print(model.graph.input[1])

data = helper.make_node("Reshape", ['X', 'data_shape'], ['data'])

model.graph.node.insert(0, data)

# print(type(model.graph.node))
# print(dir(model.graph.node))

##############################################################
# new code for debugging

remove_list = []
for node in model.graph.node:
	print(node.op_type)
	# print(dir(node))
	if(not (node.op_type=='Reshape' or node.name=='resnetv15_conv0_fwd' or node.name=='resnetv15_batchnorm0_fwd' 
		or node.name=='resnetv15_relu0_fwd' or node.name=='resnetv15_pool0_fwd' or node.name=='resnetv15_stage1_conv0_fwd' or node.name=='resnetv15_stage1_relu0_fwd'
		or node.name=='resnetv15_stage1_batchnorm0_fwd' or node.name=='resnetv15_stage1_conv1_fwd' or node.name=='resnetv15_stage1_batchnorm1_fwd'
		or node.name=='resnetv15_stage1_activation0' or node.name=='resnetv15_stage1__plus0' or node.name=='resnetv15_stage1_conv2_fwd' or node.name=='resnetv15_stage1_batchnorm2_fwd'
		or node.name=='resnetv15_stage1_relu0_fwd' or node.name=='resnetv15_stage1_relu1_fwd' or node.name=='resnetv15_stage1_conv3_fwd'
		or node.name=='resnetv15_stage1_batchnorm3_fwd' or node.name=='resnetv15_stage1__plus1' or node.name=='resnetv15_stage1_activation1'
		or node.name=='resnetv15_stage2_conv0_fwd' or node.name=='resnetv15_stage2_batchnorm0_fwd' or node.name=='resnetv15_stage2_relu0_fwd'
		or node.name=='resnetv15_stage2_conv1_fwd' or node.name=='resnetv15_stage2_batchnorm1_fwd' or node.name=='resnetv15_stage2_conv2_fwd' 
		or node.name=='resnetv15_stage2_batchnorm2_fwd' or node.name=='resnetv15_stage2__plus0' or node.name=='resnetv15_stage2_activation0'
		or node.name=='resnetv15_stage2_conv3_fwd' or node.name=='resnetv15_stage2_batchnorm3_fwd'
		or node.name=='resnetv15_stage2_relu1_fwd' or node.name=='resnetv15_stage2_conv4_fwd' or node.name=='resnetv15_stage2_batchnorm4_fwd' or node.name=='resnetv15_stage2__plus1')):
		remove_list.append(node)

for node in remove_list:
	print(node.op_type)
	model.graph.node.remove(node)


model.graph.output.pop(0)
model.graph.output.insert(0, helper.make_tensor_value_info('resnetv15_stage2__plus1', TensorProto.FLOAT, [1,128,28,28]))

print(model.graph.output)

# new code for debugging
#############################################################


save_model(model)

# params are present in both model.input and model.init_val
