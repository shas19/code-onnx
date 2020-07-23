# transform dataset

import numpy as np

filepath = 'train.npy'

data = np.load(filepath)
inp = data[:,1:]
out = data[:,:1]

total_inp = out.shape[0]
sdt_shape = [1,32,32,3]

inp = inp.reshape(total_inp, sdt_shape[0], sdt_shape[1], sdt_shape[2], sdt_shape[3]).transpose(0,1,4,2,3).reshape(total_inp, -1)

fin = np.zeros(data.shape)
fin[:,1:] = inp
fin[:,:1] = out

np.save('train_onnx.npy', fin)
