import os
import cv2
import sys
import math
import numpy as np
import torch
import tensorboardX
import torch.nn as nn
import math
import argparse

sys.path.insert(1, '../gainmap')
sys.path.insert(1, '../options')
from tqdm import tqdm
from style_option import StyleOptions
from dataset import CN_dataset, de_norm
from utils import ResnetGenerator
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision.utils import make_grid
from VGG import conv1_1

# Traditional StyLit Algorithm.
class StyLit():
    pass

# Patch based texture Synthesis for Neural Network Outputs
class CoordinateNet():
    def __init__(self, opt):
        self.zoom_up = opt.zoom_up
        self.patch_size = opt.patch_size
        self.model = ResnetGenerator(opt.inchannel, 3).cuda()
        self.opt = opt
        self.detail = conv1_1().cuda()
        # Loss functions.
        self.L1 = torch.nn.L1Loss().cuda()
        self.L2 = torch.nn.MSELoss(reduction='sum').cuda()
        self.compare = torch.nn.MSELoss().cuda()
        self.new_img = Variable(torch.ones((3, 512, 512), dtype=torch.float32), requires_grad=True)

    def load_img(self, style, face_target):
        # Read Images.
        style = cv2.cvtColor(cv2.imread(style), cv2.COLOR_BGR2RGB)
        self.face_target = cv2.cvtColor(cv2.imread(face_target), cv2.COLOR_BGR2RGB)
        self.style_target = cv2.reshape(face_target.shape, style)
        
        self.origin_size = face_target.shape[:2]
        self.after_size = self.zoom_up*self.origin_size
        self.style = cv2.reshape(self.after_size, style)

        self.output = np.zeros((self.after_size[0], self.after_size[1], 3))

    # default: the rgb blurred channel.
    def get_guide_channel(self):
        self.style_guide = []
        self.input_guide = []

    def transposed_mul(self, Input, Style):
        output = 0

        Input = Input.reshape(Input.shape[1], -1)
        Style = Style.reshape(Style.shape[1], -1)

        mul_input = torch.mm(Input, Input.transpose(1, 0))
        mul_style = torch.mm(Style, Style.transpose(1, 0))

        output = self.compare(mul_input, mul_style)

        return output

    def get_result(self, inputA, inputB):
        return self.model(torch.cat((inputA, inputB), dim = 1))

    def loss_func(self, data, res):
        
        #for i in range(512):
        #    for j in range(512):
        #        pos_x = int(torch.clamp(res[0][0][i][j] + i, 0, 512))
        #        pos_y = int(torch.clamp(res[0][0][i][j] + j, 0, 512))
        #        #print(data[2].shape)
        #        self.new_img[:, i, j] = data[2][0, :, pos_x, pos_y]
        loss_change = self.compare(res, data[2].cuda())
        loss_detail = 5*self.compare(self.detail(res), self.detail(data[2].cuda()))
        Style_loss = 0.01/(4*math.pow(data[2].shape[1], 2))\
                    * self.transposed_mul(self.detail(res), self.detail(data[2].cuda()))
        return loss_change + loss_detail

# train the network
def trainCN(opt):
    print('loading models......')
    CoorNet = CoordinateNet(opt)
    
    DN = de_norm()
    if opt.start>=1:
        CoorNet.model.load_state_dict(torch.load('checkpoints/rec/%s/model_%d.pth' % (opt.outf, opt.start)))
    os.makedirs("./log/rec/%s/"%opt.outf, exist_ok=True)
    os.makedirs("./checkpoints/rec/%s/"%opt.outf, exist_ok=True)
    
    dataloader = DataLoader(
                CN_dataset(root=os.path.join(opt.root, opt.name), name='train', mode='paired', length = 1200),
                batch_size=opt.batch_size, 
                shuffle=False,
                num_workers=0,
            )

    train_writer = tensorboardX.SummaryWriter("./log/rec/%s/"%opt.outf)
    optimizer = torch.optim.Adam(CoorNet.model.parameters(), lr=1e-5, weight_decay=1e-8)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    iters = 0 + opt.start*1200
    print('start processing.')

    for epoch in range(opt.start, opt.epoch):
        #pbar = tqdm(total=len(dataloader))
        for k, data in enumerate(dataloader):
            iters += opt.batch_size
            res = CoorNet.get_result(data[0].cuda(), data[3].cuda())
            Loss = CoorNet.loss_func(data, res)
            
            Loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #pbar.update(1)

            if iters%(opt.iter_show*5) == 0:
                # record result pics.
                print(Loss.item())
                temp_image = make_grid(torch.clamp(DN(res[0]),0,1), nrow=opt.batch_size, padding=0, normalize=False)
                train_writer.add_image('temp result', temp_image, iters)
                temp_image = make_grid(torch.clamp(DN(data[0][0]),0,1), nrow=opt.batch_size, padding=0, normalize=False)
                train_writer.add_image('example', temp_image, iters)
                temp_image = make_grid(torch.clamp(DN(data[2][0]),0,1), nrow=opt.batch_size, padding=0, normalize=False)
                train_writer.add_image('target style', temp_image, iters)

        #pbar.close()
        torch.save(CoorNet.model.cpu().state_dict(), 'checkpoints/rec/%s/model_%d.pth' % (opt.outf, epoch))
        CoorNet.model.cuda()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    OptionInit = StyleOptions(parser)
    parser = OptionInit.initialize(parser)
    opt = parser.parse_args()
    trainCN(opt)    