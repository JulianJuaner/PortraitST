import os
import cv2
import sys
import math
import numpy as np
import torch
import tensorboardX
sys.path.insert(1, '../gainmap')
sys.path.insert(1, '../options')
from style_option import StyleOptions
from dataset import CN_dataset
from utils import ResnetGenerator

# Patch based texture Synthesis for Neural Network Outputs
class CoordinateNet():
    def __init__(self, opt):
        self.zoom_up = opt.zoom_up
        self.patch_size = opt.patch_size
        self.model = ResnetGenerator(opt.input, 2)
        self.opt = opt

    def load_img(self, style, face_target):
        # Read Images.
        style = cv2.cvtColor(cv2.imread(style), cv2.COLOR_BGR2RGB)
        self.face_target = cv2.cvtColor(cv2.imread(face_target), cv2.COLOR_BGR2RGB)
        self.style_target = cv2.reshape(face_target.shape, style)
        
        self.origin_size = face_target.shape[:2]
        self.after_size = self.zoom_up*self.origin_size
        self.style = cv2.reshape(self.after_size, style)

        self.output = np.zeros((self.after_size[0], self.after_size[1], 3))
        
    def get_guide_channel(self):
        self.style_guide = []
        self.input_guide = []

    def get_result(self, inputs):
        return self.model(inputs)

def train():


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    OptionInit = StyleOptions(parser)
    parser = OptionInit.initialize(parser)
    opt = parser.parse_args()
    train(opt)    