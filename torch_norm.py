## This code is built on https://github.com/yxlijun/S3FD.pytorch

 

#-*- coding:utf-8 -*-

 

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

 


import torch
import torch.nn as nn
import torch.nn.init as init
# from torch.autograd import Function
# from torch.autograd import Variable

 

 

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.ones(self.n_channels))
        # print(self.n_channels)
        # print(torch.ones(self.n_channels))
        # print(nn.Parameter(torch.ones(self.n_channels)))
        # print(self.weight)
        self.reset_parameters()

 

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

 

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        # print(norm.shape)
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

 

L2Norm3_3 = L2Norm(32, 1)
x = torch.randn(1, 32, 30, 40)

out = L2Norm3_3(x)

sum_square = 0
for i in range(30):
    for j in range(40):
        sum_square += (x[0,0,i,j]*x[0,0,i,j])

print(sum_square)
import math
print(x[0,0,0,0])
print(x[0,0,0,0]/math.sqrt(sum_square))        
print(1/math.sqrt(sum_square))


print(out[0,0,0,0])
