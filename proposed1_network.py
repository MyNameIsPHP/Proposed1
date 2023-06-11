#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os
import math
import matplotlib.pyplot as plt


class Dense_Layer(nn.Module):
    def __init__(self, in_channels, growthrate, bn_size):
        super(Dense_Layer, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, bn_size * growthrate, kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(bn_size * growthrate)
        self.conv2 = nn.Conv2d(
            bn_size * growthrate, growthrate, kernel_size=3, padding=1, bias=False
        )

    def forward(self, prev_features):
        out1 = torch.cat(prev_features, dim=1)
        out1 = F.relu(out1)
        out1 = self.conv1(out1)
        out1 = F.relu(out1)
        out2 = self.conv2(out1)
        return out2

class Dense_Block(nn.ModuleDict):
    def __init__(self, n_layers, in_channels, growthrate, bn_size):
        """
        A Dense block consists of `n_layers` of `Dense_Layer`
        Parameters
        ----------
            n_layers: Number of dense layers to be stacked 
            in_channels: Number of input channels for first layer in the block
            growthrate: Growth rate (k) as mentioned in DenseNet paper
            bn_size: Multiplicative factor for # of bottleneck layers
        """
        super(Dense_Block, self).__init__()

        layers = dict()
        for i in range(n_layers):
            layer = Dense_Layer(in_channels + i * growthrate, growthrate, bn_size)
            layers['dense{}'.format(i)] = layer
        
        self.block = nn.ModuleDict(layers)
    
    def forward(self, features):
        if(isinstance(features, torch.Tensor)):
            features = [features]
        
        for _, layer in self.block.items():
            new_features = layer(features)
            features.append(new_features)
        
        return torch.cat(features, dim=1)


# dense_block = dense_block.cuda()


class Proposed1(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(Proposed1, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.dense_block = nn.Sequential(
            Dense_Block(6, 32, growthrate=32, bn_size=4),
            )
        self.conv = nn.Sequential(
            # nn.Conv2d(224, 192, 3, 1, 1),
            # nn.ReLU(),
            # nn.Conv2d(192, 160, 3, 1, 1),
            # nn.ReLU(),
            # nn.Conv2d(160, 128, 3, 1, 1),
            # nn.ReLU(),
            # nn.Conv2d(128, 96, 3, 1, 1),
            # nn.ReLU(),
            # nn.Conv2d(96, 64, 3, 1, 1),
            # nn.ReLU(),
            # nn.Conv2d(64, 32, 3, 1, 1),
            # nn.ReLU(),
            # nn.Conv2d(32, 3, 3, 1, 1),
            nn.Conv2d(224, 3, 3, 1, 1),
            )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)
            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)
            x = h

            x = self.dense_block(x)

            x = self.conv(x)
            x = x + input
            x_list.append(x)
        return x, x_list