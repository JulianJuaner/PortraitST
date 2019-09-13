
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from torch.nn import init

class myVGG(nn.Module):
    def __init__(self, requires_grad=False):
        super(myVGG, self).__init__()
        original_model = models.vgg19_bn(pretrained=False)
        original_model.load_state_dict(torch.load('models/vgg19_bn-c79401a0.pth'))
        self.features = nn.Sequential(
                    # stop at conv4
                    *list(original_model.features.children())[:-4]
                )
        if requires_grad == False:
            for param in self.parameters():
                param.requires_grad = False
    def forward(self, x):
        return self.features(x)