import os
import time
import sys
import torch
import cv2
import math
import argparse
import tensorboardX

from tqdm import tqdm
from VGG import myVGG
from dataset import ST_dataset, RC_dataset, de_norm, decolor
sys.path.insert(1, '../options')
from gainmap_option import FeatureOptions
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def window_stdev(X, window_size, kernel):
    X = F.pad(X, [window_size//2,window_size//2,window_size//2,window_size//2], mode='reflect')
    c1 = F.conv2d(X, kernel)
    c2 = F.conv2d(torch.pow(X,2), kernel)
    t = c2 - c1*c1
    return torch.sqrt(torch.clamp_min_(t,0))

def GaborFilter(img, gkernel):
    winsize = gkernel.shape[2]
    #input = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).cuda()
    input = img#.cuda()
    weight = torch.from_numpy(gkernel).unsqueeze(1).float().cuda()
    kernel = (torch.ones((1,1,winsize,winsize)).float() / (winsize * winsize)).cuda()
    gaborfeats = F.conv2d(F.pad(input, [winsize//2,winsize//2,winsize//2,winsize//2], mode='reflect'),weight)
    gaborfeats = gaborfeats.reshape(48,2,gaborfeats.shape[2],gaborfeats.shape[3])
    tmp = torch.sqrt(torch.sum(torch.pow(gaborfeats, 2), 1, keepdim=True))
    mean = F.conv2d(F.pad(tmp, [winsize//2,winsize//2,winsize//2,winsize//2], mode='reflect'), kernel)
    stddev = window_stdev(tmp, winsize, kernel)
    return torch.cat((mean, stddev), 0)#.cpu()#.numpy()

class GaborWavelet(nn.Module):
    def __init__(self, gkern=None, winsize=13):
        super(GaborWavelet, self).__init__()
        if gkern is None:
            gkern = torch.from_numpy(np.load("../weights/kernel2.npy")).float().unsqueeze(1)
            winsize = gkern.shape[2]
        self.pad = nn.ReflectionPad2d(winsize//2)
        # self.gabor = nn.Conv2d(1,96,kernel_size=winsize, padding=winsize//2, padding_mode='reflect', bias=False)
        self.gabor = nn.Conv2d(1,96,kernel_size=winsize, bias=False)
        self.gabor.weight.data = gkern
        print('load weight:', gkern.shape)
        self.wind = nn.Conv2d(1, 1,kernel_size=winsize, bias=False)
        nn.init.constant_(self.wind.weight.data, 1.0/(winsize*winsize))
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        feats = self.gabor(self.pad(x))
        feats = feats.reshape(-1, 2, feats.shape[2], feats.shape[3])
        tmp = torch.sqrt(torch.sum(torch.pow(feats, 2), 1, keepdim=True))
        mean = self.wind(self.pad(tmp)).reshape(-1,48,feats.shape[2], feats.shape[3])
        stddev = torch.sqrt(torch.clamp(self.wind(self.pad(torch.pow(tmp,2)))-torch.pow(self.wind(self.pad(tmp)),2),min=0)).reshape(-1,48,feats.shape[2], feats.shape[3])
        return torch.cat((mean, stddev), 1)

def ModifyMap(Style, Input, opt):
    Gain = torch.div(Style, Input+1e-4)
    Gain = torch.clamp(Gain, min=opt.gmin, max=opt.gmax)
    Modified = Input*Gain

    # for the Gaty's implementation, return the input feature as gainmap.
    if opt.optimode == 'dataset':
        return Input

    return Modified

class FeatureMap(torch.nn.Module):
    def __init__(self, opt):
        super(FeatureMap, self).__init__()
        self.featureList = []
        self.opt = opt
    def mapload(self, input_feats, style_feats, length):
        for i in range(length):
            self.featureList += [ModifyMap(style_feats[i], input_feats[i], self.opt)]

class OverAllLoss():
    def __init__(self, opt):
        self.L1 = torch.nn.L1Loss().cuda()
        self.L2 = torch.nn.MSELoss(reduction='sum').cuda()
        self.compare = torch.nn.MSELoss().cuda()

        # set parameters
        self.alpha_3 = opt.alpha_3
        self.alpha_4 = opt.alpha_4
        self.beta_3 = opt.beta_3
        self.beta_4 = opt.beta_4
        self.gT = opt.gT
        self.gmin = opt.gmin
        self.gmax = opt.gmax
        self.sigma = 1e-4
        self.opt = opt

    # get the style representation.
    def transposed_mul(self, Input, Style):
        output = 0

        Input = Input.reshape(Input.shape[1], -1)
        Style = Style.reshape(Style.shape[1], -1)

        mul_input = torch.mm(Input, Input.transpose(1, 0))
        mul_style = torch.mm(Style, Style.transpose(1, 0))

        output = self.L2(mul_input, mul_style)

        return output


    def forward(self, Style, Input, Map=None, mode='convN'):
        # A, B: 4-D tensors.
        if Map is None:
            print('new map needed.')
            Gain = torch.div(Style, Input+self.sigma)
            Gain = torch.clamp(Gain, min=self.gmin, max=self.gmax)
            Map = torch.mul(Input, Gain)

        if 'conv3_1' in mode:
            alpha = self.alpha_3
            beta = self.beta_3
        elif 'conv4_1' in mode:
            alpha = self.alpha_4
            beta = self.beta_4
        else:
            alpha = 1.0
            beta = 1.0

        Gain_loss = alpha * self.L2(Input, Map) / (2*Style.shape[1]*Style.shape[2]*Style.shape[3])

        Style_loss = self.gT * beta /(4*math.pow(Style.shape[1], 2))\
                    * self.transposed_mul(Input, Style)
        
        return Gain_loss, Style_loss

def StyleTransfer(opt):
    print('loading VGG models......')
    model = myVGG(layers=opt.layers.split(',')).cuda()
    
    totalLoss = OverAllLoss(opt)
    DN = de_norm()
    os.makedirs("./log/%s/"%opt.outf, exist_ok=True)
    os.makedirs("./checkpoints/%s/"%opt.outf, exist_ok=True)
    train_writer = tensorboardX.SummaryWriter("./log/%s/"%opt.outf)

    modefactor = 1 # number of face images
    counter = 0 #count for dataset.

    if opt.optimode == 'dataset':
        os.makedirs("./checkpoints/%s/0/"%opt.outf, exist_ok=True)
        os.makedirs("./checkpoints/%s/1/"%opt.outf, exist_ok=True)
        dataloader = DataLoader(
                RC_dataset(root=opt.root, name='train', mode='paired'),
                batch_size=opt.batch_size, 
                shuffle=False,
                num_workers=0,
            )
        modefactor = 2
    
    else:
        dataloader = DataLoader(
                ST_dataset(root=opt.root, name=opt.name, mode='paired'),
                batch_size=opt.batch_size, 
                shuffle=False,
                num_workers=0,
        )

    images = -1

    # Trigger of the gabor wavelet loss.
    if opt.gabor == True:
        Gabor = GaborWavelet().cuda()
        Gabor_target = Gabor(decolor(DN(data[1][0].cuda()).unsqueeze(0))/255)

    print('start processing.')
    for epoch in range(opt.epoch):
        for k, data in enumerate(dataloader):
            for imgs in range(0, modefactor):

                if opt.optimode == 'dataset':
                    input_feats = model(data[imgs].cuda())
                    style_feats = model(data[2].cuda())
                    # Initialize the output.
                    output = data[imgs].clonr().cuda()
                else:
                    input_feats = model(data[1].cuda())
                    style_feats = model(data[0].cuda())
                    output = data[1].clone().cuda()

                Maps = []
                for i in range(len(model.layers)):
                    Maps += [ModifyMap(style_feats[i], input_feats[i], opt)]
                
                index = 0
                view_shape = (4, int(Maps[index].shape[1]/4), Maps[index].shape[2],  Maps[index].shape[3])
                print(view_shape)
                temp_image = make_grid(Maps[index].reshape(view_shape)[:,:3,:,:], nrow=4, padding=0, normalize=True)
                train_writer.add_image('Gain Map', temp_image, 0)
                temp_image = make_grid(style_feats[index].reshape(view_shape)[:,:3,:,:], nrow=4, padding=0, normalize=True)
                train_writer.add_image('style_feat_4', temp_image, 0)
                temp_image = make_grid(input_feats[index].reshape(view_shape)[:,:3,:,:], nrow=4, padding=0, normalize=True)
                train_writer.add_image('input_feat_4', temp_image, 0)

                # Initialize the output.
                #output = data[1].cuda()
                #output.requires_grad = True
                output = torch.nn.Parameter(output, requires_grad=True)
                images += 1
                # Set optimizer.
                optimizer = torch.optim.LBFGS([output], lr=opt.lr)
                optimizer.zero_grad()

                # Iteration for 300 times.
                pbar = tqdm(total=opt.iter)

                for iters in range(opt.iter+1):

                    def closure():
                        input_feats = model(output)
                        optimizer.zero_grad()
                        Loss_gain = 0
                        Loss_style = 0
                        Loss = 0
                        for i in range(len(model.layers)):
                            loss_gain_item, loss_style_item = totalLoss.forward(style_feats[i], input_feats[i],
                                                                Map=Maps[i], mode=model.layers[i])
                            Loss_gain += loss_gain_item
                            Loss_style += loss_style_item

                        if opt.gabor == True:
                            # decolorization
                            Loss_gabor = totalLoss.compare(
                                Gabor(decolor(DN(output[0]).unsqueeze(0))/255),
                                Gabor_target)
                            Loss_gain += Loss_gabor

                        # loss term
                        Loss = Loss_gain + Loss_style
                        Loss.backward(retain_graph=True)
                        return Loss

                    if iters%opt.iter_show == 0:
                        # record result pics.
                        temp_image = make_grid(torch.clamp(DN(output[0]).unsqueeze(0),0,1), nrow=opt.batch_size, padding=0, normalize=False)
                        train_writer.add_image('temp result', temp_image, iters+images*opt.iter)

                        if iters%(20) == 0 and iters!=0:
                            if opt.optimode == 'dataset':
                                print(imgs)
                                if imgs == 0:
                                    save_image(temp_image, "./checkpoints/%s/0/%d.png"%(opt.outf, counter))
                                else:
                                    save_image(temp_image, "./checkpoints/%s/1/%d.png"%(opt.outf, counter))
                                    counter += 1
                            else:
                                save_image(temp_image, "./checkpoints/%s/%d_%d.png"%(opt.outf, k, iters))

                    # Updates.
                    #Loss.backward(retain_graph=True)
                    optimizer.step(closure)
                    #optimizer.zero_grad()
                    pbar.update(1)

                pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    OptionInit = FeatureOptions(parser)
    parser = OptionInit.initialize(parser)
    opt = parser.parse_args()
    StyleTransfer(opt)
    