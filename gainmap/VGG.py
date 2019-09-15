
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
    def __init__(self, requires_grad=False, layer='conv4_1'):
        super(myVGG, self).__init__()
        original_model = models.vgg19(pretrained=False)
        original_model.load_state_dict(torch.load('../weights/vgg19-dcbb9e9d.pth'))
        #original_model.load_state_dict(torch.load('../weights/vgg19_bn-c79401a0.pth'))
        #if vgg_bn: 17, 17-30; vgg: 12, 12-21
        if 'conv4' in layer:
            self.features = nn.Sequential(
                        *list(original_model.features.children())[12:21])
        else:
            self.features = nn.Sequential(
                        *list(original_model.features.children())[:12])
        #print(self.features)

        if requires_grad == False:
            for param in self.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        return self.features(x)

if __name__ == '__main__':
    myVGG()