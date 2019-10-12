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
from dataset import ST_dataset, de_norm
from options import FeatureOptions
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

def ModifyMap(Style, Input, opt):
    Gain = torch.div(Style, Input+1e-4)
    Gain = torch.clamp(Gain, min=opt.gmin, max=opt.gmax)
    Modified = Input*Gain
    return Modified

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

    dataloader = DataLoader(
            ST_dataset(root=opt.root, name=opt.name, mode='paired'),
            batch_size=opt.batch_size, 
            shuffle=False,
            num_workers=0,
        )
    images = -1

    print('start processing.')
    for k, data in enumerate(dataloader):
        style_feats = model(data[0].cuda())
        input_feats = model(data[1].cuda())
        
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
        output = data[1].cuda()
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

                for i in range(len(model.layers)):
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
                # loss term
                Loss = Loss_gain + Loss_style
                Loss.backward(retain_graph=True)
                return Loss

            if iters%opt.iter_show == 0:
                # record result pics.
                temp_image = make_grid(torch.clamp(DN(output[0]).unsqueeze(0),0,1), nrow=opt.batch_size, padding=0, normalize=False)
                train_writer.add_image('temp result', temp_image, iters+images*opt.iter)

                if iters%(opt.iter_show*30) == 0:
                    save_image(temp_image, "./checkpoints/%s/%d_%d.png"%(opt.outf, k, iters))
                    
                
                
            #if iters%10 == 0:
                # record loss items variation
                #train_writer.add_scalar("total_loss", Loss.item(), iters+images*opt.iter)
                #train_writer.add_scalar("loss_gain", Loss_gain.item(), iters+images*opt.iter)
                #train_writer.add_scalar("loss_style", Loss_style.item(), iters+images*opt.iter)

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
    