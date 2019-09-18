import os
import time
import sys
import torch
import math
import argparse
import tensorboardX

from tqdm import tqdm
from VGG import myVGG
from dataset import ST_dataset
from options import FeatureOptions
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

def ModifyMap(Style, Input, opt):
    Gain = torch.div(Style, Input+1e-4)
    Modified = Input*Gain
    Modified = torch.clamp(Modified, min=opt.gmin, max=opt.gmax)
    return Modified

class OverAllLoss():
    def __init__(self, opt):
        self.L1 = torch.nn.L1Loss().cuda()
        self.L2 = torch.nn.MSELoss().cuda()
        self.compare = torch.nn.MSELoss(reduction='sum').cuda()

        #set parameters
        self.alpha_3 = opt.alpha_3
        self.alpha_4 = opt.alpha_4
        self.beta_3 = opt.beta_3
        self.beta_4 = opt.beta_4
        self.gT = opt.gT
        self.gmin = opt.gmin
        self.gmax = opt.gmax
        self.sigma = 1e-4
        

    def forward(self, Style, Input, Map='none', mode='conv3_1'):
        # A, B: 4-D tensors.
        if Map is not None:
            Gain = torch.div(Style, Input+self.sigma)
            Map = torch.mul(Input, Gain)
            Map = torch.clamp(Map, min=self.gmin, max=self.gmax)

        if 'conv3' in mode:
            alpha = self.alpha_3
            beta = self.beta_3
        elif 'conv4' in mode:
            alpha = self.alpha_4
            beta = self.beta_4
        else:
            alpha = 1.0
            beta = 1.0

        Gain_loss = alpha * self.L2(Input, Map) / 2
        Style_loss = self.gT * beta /(2*math.pow(Style.shape[1], 2))\
                    * self.compare(Input, Style)

                #****The point that confuced me****#
                #* self.compare(torch.mul(Input, Input.transpose(2,3)),
                #               torch.mul(Style, Style.transpose(2,3)))
        
        return Gain_loss, Style_loss

def StyleTransfer(opt):
    print('loading VGG models......')
    model = myVGG(layers=opt.layers.split(',')).cuda()
    
    totalLoss = OverAllLoss(opt)

    os.makedirs("./log/%s/"%opt.name, exist_ok=True)
    os.makedirs("./checkpoints/%s/"%opt.name, exist_ok=True)
    train_writer = tensorboardX.SummaryWriter("./log/%s/"%opt.name)

    dataloader = DataLoader(
            ST_dataset(root=opt.root, name=opt.name, mode='unpaired'),
            batch_size=opt.batch_size, 
            shuffle=False,
            num_workers=0,
        )
    images = -1

    print('start processing.')
    for _, data in enumerate(dataloader):
        style_feats = model(data[0].cuda())
        input_feats = model(data[1].cuda())

        Maps = []
        for i in len(model.layers):
            Maps += [ModifyMap(style_feats[i], input_feats[i], opt)]

        
        temp_image = make_grid((Maps[1][:,:3,:,:]-opt.gmin)/(opt.gmax-opt.gmin), nrow=opt.batch_size, padding=0, normalize=False)
        train_writer.add_image('Gain Map', temp_image, 0)
        temp_image = make_grid(style_feats[1][:,:3,:,:], nrow=opt.batch_size, padding=0, normalize=False)
        train_writer.add_image('style_feat_4', temp_image, 0)
        temp_image = make_grid(input_feats[1][:,:3,:,:], nrow=opt.batch_size, padding=0, normalize=False)
        train_writer.add_image('input_feat_4', temp_image, 0)

        # Initialize the output.
        output = data[1].cuda()
        #output.requires_grad = True
        output = torch.nn.Parameter(output, requires_grad=True)
        images += 1
        # Set optimizer.
        optimizer = torch.optim.Adam([output], lr=opt.lr)
        optimizer.zero_grad()

        # Iteration for 300 times.
        pbar = tqdm(total=opt.iter)

        for iters in range(opt.iter+1):

            for i in len(model.layers):
                if 'conv3_1' in model.layers[i]:
                    loss_gain_item, loss_style_item = totalLoss.forward(style_feats[i], input_feats[i],
                                                                        Map=Maps[i], mode='conv3_1')
                elif 'conv4_1' in model.layers[i]:
                    loss_gain_item, loss_style_item = totalLoss.forward(style_feats[i], input_feats[i],
                                                                        Map=Maps[i], mode='conv4_1')
                else:
                    loss_gain_item, loss_style_item = totalLoss.forward(style_feats[i], input_feats[i],
                                                                        Map=Maps[i])
                Loss_gain += loss_gain_item
                Loss_style += loss_style_item
            
            Loss = Loss_gain + Loss_style
            
            if iters%opt.iter_show == 0:
                # record result pics.
                temp_image = make_grid(output, nrow=opt.batch_size, padding=0, normalize=True)
                train_writer.add_image('temp result', temp_image, iters+images*opt.iter)
                
            if iters%10 == 0:
                # record loss items variation
                train_writer.add_scalar("total_loss", Loss.item(), iters+images*opt.iter)
                train_writer.add_scalar("loss_gain", Loss_gain.item(), iters+images*opt.iter)
                train_writer.add_scalar("loss_style", Loss_style.item(), iters+images*opt.iter)

            # Updates.
            Loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            input_feats = model(output)
            pbar.update(1)

        pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    OptionInit = FeatureOptions(parser)
    parser = OptionInit.initialize(parser)
    opt = parser.parse_args()
    StyleTransfer(opt)
    pass