import os
import cv2
import sys
import math
import numpy as np
import torch
import tensorboardX

# Patch based texture Synthesis for Neural Network Outputs
class StyLit():
    def __init__(self, opt):
        self.threshold = opt.threshold
        self.Level = opt.num_level
        self.zoom_up = opt.zoom_up
        self.patch_size = opt.patch_size
        self.opt = opt

    def load_img(self, style, face_target):
        # Read Images.
        style = cv2.cvtColor(cv2.imread(style), cv2.COLOR_BGR2RGB)
        self.face_target = cv2.cvtColor(cv2.imread(face_target), cv2.COLOR_BGR2RGB)
        self.style_target = cv2.reshape(face_target.shape, style)
        
        self.origin_size = face_target.shape[:2]
        self.after_size = self.zoom_up*self.origin_size
        self.style = cv2.reshape(after_size, style)

        self.output = np.zeros((after_size[0], after_size[1], 3))
        
    def 


    