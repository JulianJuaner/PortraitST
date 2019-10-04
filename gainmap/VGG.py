
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import init
from torchvision import models

# The VGG19_bn style generator presented in the paper.
# As two layer features could be extracted continuely, the model is divided into two continuous parts.
# (start-conv3_1; conv3_2-conv4_1) to enhance performance.

class myVGG(nn.Module):
    def __init__(self, requires_grad=False, layers=None, BN=True):
        super(myVGG, self).__init__()

        if BN:
            self.original_model = models.vgg19_bn(pretrained=False)
            self.original_model.load_state_dict(torch.load('../weights/vgg19_bn-c79401a0.pth'))
            self.checkpoints = [3, 7, 10, 14, 17, 20, 23, 27, 30, 33, 36, 40, 43, 46, 49, 53]
        else:
            self.original_model = models.vgg19(pretrained=False)
            self.original_model.load_state_dict(torch.load('../weights/vgg19-dcbb9e9d.pth'))
            self.checkpoints = [2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34, 37]

        #if vgg_bn: 17, 17-30; vgg: 12, 12-21
        self.layers = layers
        self.convs = [
                       'conv1_1','conv1_2',
                       'conv2_1','conv2_2',
                       'conv3_1','conv3_2','conv3_3','conv3_4',
                       'conv4_1','conv4_2','conv4_3','conv4_4',
                       'conv5_1','conv5_2','conv5_3','conv5_4',
                       ]
        self.features = []
        self.CreateCheckPoint()

        if requires_grad == False:
            for param in self.parameters():
                param.requires_grad = False

    # seperate sequensial model into parts.
    def CreateCheckPoint(self):
        start = 0
        self.layers.sort()
        for layer in self.layers:
            checkpoint = self.checkpoints[self.convs.index(layer)]
            self.features += [nn.Sequential(
                        *list(self.original_model.features.children())[start:checkpoint])]
            start = checkpoint

    def forward(self, x):
        result = []
        for f in self.features:
            x = f(x)
            result += [x]
        return result

if __name__ == '__main__':
    myVGG()