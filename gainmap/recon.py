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
from dataset import RC_dataset, de_norm
from options import FeatureOptions
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision.utils import make_grid
from modify import ModifyMap, OverAllLoss

# Convolution block.
class doubleConv(torch.nn.Module):
    def __init__(self, input_, output):
        super(doubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_, output, 3, 1, 1),
            nn.InstanceNorm2d(output),
            nn.ReLU(True),
            nn.Conv2d(output, output, 3, 1, 1),
            nn.InstanceNorm2d(output),
            nn.ReLU(True),
        )
    def forward(self, x):
        return self.conv(x)

class VGGRC(nn.Module):
    def __init__(self, opt, requires_grad=True):
        super(VGGRC, self).__init__()
        self.VGG = myVGG(layers=opt.layers.split(','))
        self.pool = nn.MaxPool2d(2, stride=2)

        self.loss = torch.nn.MSELoss().cuda()
        self.loss_cycle = torch.nn.L1Loss().cuda()

        self.upscale_1 = nn.ConvTranspose2d(512, 512, 4, 2, 1)
        self.upscale_2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.upscale_3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.upscale_4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)

        self.conv1 = doubleConv(512*2, 512)
        self.conv2 = doubleConv(256*2, 256)
        self.conv3 = doubleConv(128*2, 128)
        self.conv4 = doubleConv(64*2, 64)
        self.net_output = nn.Conv2d(64, 3, 1, 1, 0)
        self.opt = opt
        

    def net_forward(self, Input):
        x = self.upscale_1(Input[4])
        x = self.conv1(torch.cat((x, Input[3]), 1))
        x = self.upscale_2(x)
        x = self.conv2(torch.cat((x, Input[2]), 1))
        x = self.upscale_3(x)
        x = self.conv3(torch.cat((x, Input[1]), 1))
        x = self.upscale_4(x)
        x = self.conv4(torch.cat((x, Input[0]), 1))
        x = self.net_output(x)

        return x

    def forward(self, face):
        res = 0
        face_feat = self.VGG(face)
        #Maps = []
        out = self.net_forward(face_feat)
        out_feat = self.VGG(out)
        Loss_rc = self.loss(out, face)
        Loss_cycle = 0

        for i in range(len(self.VGG.layers)):

            Loss_cycle += self.loss_cycle(out_feat[i], face_feat[i])

        Loss = 50*Loss_rc + Loss_cycle
        return Loss, out

# The reconstruction network.
# Default: all *_1 layers in VGG19 network.
class FeatureRC(nn.Module):
    def __init__(self, opt, requires_grad=True):
        super(FeatureRC, self).__init__()
        self.VGG = myVGG(layers=opt.layers.split(','))
        self.pool = nn.MaxPool2d(2, stride=2)
        self.loss = OverAllLoss(opt)

        self.upscale_1 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.upscale_2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.upscale_3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.upscale_4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)

        self.conv1 = doubleConv(512*3, 512)
        self.conv2 = doubleConv(256*3, 256)
        self.conv3 = doubleConv(128*3, 128)
        self.conv4 = doubleConv(64*3, 64)
        self.net_output = nn.Conv2d(64, 3, 1, 1, 0)
        self.opt = opt
        

    def net_forward(self, Input, Style):
        x = self.upscale_1(torch.cat((Input[4], Style[4]), 1))
        x = self.conv1(torch.cat((x, Input[3], Style[3]), 1))
        x = self.upscale_2(x)
        x = self.conv2(torch.cat((x, Input[2], Style[2]), 1))
        x = self.upscale_3(x)
        x = self.conv3(torch.cat((x, Input[1], Style[1]), 1))
        x = self.upscale_4(x)
        x = self.conv4(torch.cat((x, Input[0], Style[0]), 1))
        x = self.net_output(x)

        return x

    def forward(self, face, style):
        res = 0
        face_feat = self.VGG(face)
        style_feat = self.VGG(style)

        out = self.net_forward(face_feat, style_feat)
        out_feat = self.VGG(out)

        Maps = []
        
        for i in range(len(self.VGG.layers)):
            Loss_gain = 0
            Loss_style = 0
            Maps += [ModifyMap(style_feat[i], face_feat[i], self.opt)]
            
            if 'conv3_1' in self.VGG.layers[i]:
                loss_gain_item, loss_style_item = self.loss.forward(style_feat[i], 
                out_feat[i], Map=Maps[i], mode='conv3_1')
            elif 'conv4_1' in self.VGG.layers[i]:
                loss_gain_item, loss_style_item = self.loss.forward(style_feat[i], 
                out_feat[i], Map=Maps[i], mode='conv4_1')
            else:
                loss_gain_item, loss_style_item = self.loss.forward(style_feat[i],
                 out_feat[i], Map=Maps[i])

            Loss_gain += loss_gain_item
            Loss_style += loss_style_item

        Loss = Loss_gain + Loss_style
        return Loss, out

# train function.
def trainRC(opt):
    print('loading VGG models......')
    DN = de_norm()
    model = VGGRC(opt).cuda()
    if opt.start>=1:
        model.load_state_dict(torch.load('checkpoints/rec/%s/model_%d.pth' % (opt.outf, opt.start)))
    os.makedirs("./log/rec/%s/"%opt.outf, exist_ok=True)
    os.makedirs("./checkpoints/rec/%s/"%opt.outf, exist_ok=True)
    train_writer = tensorboardX.SummaryWriter("./log/rec/%s/"%opt.outf)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-8)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    traindataset  = RC_dataset(root=opt.root, name='train', mode='unpaired')
    dataloader = DataLoader(
            traindataset, 
            batch_size=opt.batch_size, 
            shuffle=False,
            num_workers=0,
        )

    testloader = DataLoader(
            RC_dataset(root=opt.root, name='test', mode='unpaired'),
            batch_size=1, 
            shuffle=False,
            num_workers=0,
    )

    iters = 0 + opt.start*len(traindataset)
    print('start processing.')
    for epoch in range(opt.start, opt.epoch):
        pbar = tqdm(total=len(dataloader))
        for k, data in enumerate(dataloader):
            iters += opt.batch_size
            Loss, out = model(data[1].cuda())

            if iters%50 == 0:
                train_writer.add_scalar("total_loss", Loss.item(), iters)

            if iters%100 == 0:
                index = 0

                for k2, test in enumerate(testloader):
                    if k2 == 0:
                        _, out = model(test[0].cuda())

                        #temp_image = make_grid(test[1], nrow=1, padding=0, normalize=True)
                        #train_writer.add_image('style', temp_image, iters+k2)
                        temp_image = make_grid(torch.clamp((DN(test[0][0])/255).unsqueeze(0),0, 1), nrow=1, padding=0, normalize=False)
                        train_writer.add_image('face', temp_image, iters+k2)
                        
                        temp_image = make_grid(torch.clamp((DN(out[0])/255).unsqueeze(0), 0, 1), nrow=1, padding=0, normalize=False)
                        train_writer.add_image('out', temp_image, iters+k2)

            Loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            
            pbar.update(1)

        pbar.close()
        torch.save(model.cpu().state_dict(), 'checkpoints/rec/%s/model_%d.pth' % (opt.outf, epoch))
        model.cuda()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    OptionInit = FeatureOptions(parser)
    parser = OptionInit.initialize(parser)
    opt = parser.parse_args()
    trainRC(opt)
