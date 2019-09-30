# The file for reconstruction of VGG structure-wised features 
# and style representations.
# Main purpose is to speed up the training process and reduce number of iterations.

import os
import sys
import torch
import torch.nn as nn
import math
import argparse
import tensorboardX

from tqdm import tqdm
from VGG import myVGG
from dataset import ST_dataset
from options import FeatureOptions
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from modify import ModifyMap, OverAllLoss

# Convolution block.
class doubleConv(torch.nn.Module):
    def __init__(self, input_, output):
        super(doubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_, output, 3, 1, 1),
            #nn.InstanceNorm2d(output, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv2d(output, output, 3, 1, 1),
            #nn.InstanceNorm2d(output, track_running_stats=False),
            nn.ReLU(True),
        )
    def forward(self, x):
        return self.conv(x)

# The reconstruction network.
# Default: all *_1 layers in VGG19 network.
class FeatureRC(nn.Module):
    def __init__(self, opt, requires_grad=True):
        super(FeatureRC, self).__init__()
        self.FeatureGenerator = myVGG(layers=opt.layers)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.upscale_1 = nn.ConvTranspose2d(512, 512, 4, 2, 1)
        self.upscale_2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.upscale2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.upscale_3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)

        self.conv1 = doubleConv(1024, 512)
        self.conv2 = doubleConv(512, 256)
        self.conv3 = doubleConv(256, 128)
        self.conv4 = doubleConv(128, 64)
        self.net_output = nn.Conv2d(64, 3, 1, 1, 0)

    def forward(self, input, style):
        res = 0
        
        return res