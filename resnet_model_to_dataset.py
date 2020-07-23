import numpy as np
import onnx
import os
import glob
import onnxruntime

from onnx import numpy_helper

model = onnx.load('input.onnx')
test_data_dir = 'test_data_set_0'

# Load inputs
inputs = []
inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
for i in range(inputs_num):
    input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
    tensor = onnx.TensorProto()
    with open(input_file, 'rb') as f:
        tensor.ParseFromString(f.read())
    inputs.append(numpy_helper.to_array(tensor))
    # print(inputs[0])

# Load reference outputs
ref_outputs = []
ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
for i in range(ref_outputs_num):
    output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
    tensor = onnx.TensorProto()
    with open(output_file, 'rb') as f:
        tensor.ParseFromString(f.read())
    ref_outputs.append(numpy_helper.to_array(tensor))

# Run the model on the backend
input_name = model.graph.input[0].name

inp = inputs[0].reshape(-1,1)

sess = onnxruntime.InferenceSession('input.onnx') 
pred = sess.run(None, {input_name: inp})

# outputs = list(backend.run_model(model, inputs))

#(10, 150529)
print(pred[0].reshape(1,-1).shape)
print(inputs[0].reshape(1, -1).shape)

print((np.min(inputs[0]),np.max(inputs[0])))

train = np.concatenate([pred[0].reshape(1,-1), inputs[0].reshape(1, -1)], 1)
print("Shape of training data is" + str(train.shape))
np.save('train_onnx.npy', train)

# Compare the results with reference outputs.
# for ref_o, o in zip(ref_outputs, pred):
#     np.testing.assert_almost_equal(ref_o, o, decimal=4)