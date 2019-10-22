import os
import cv2
import sys
import torch
import torch.nn as nn
import math
import copy
import argparse
import tensorboardX

from tqdm import tqdm
from VGG import myVGG
from dataset import RC_dataset, ST_dataset, de_norm
from options import FeatureOptions
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision.utils import make_grid, save_image
from modify import ModifyMap, OverAllLoss, FeatureMap
from recon import VGGRC

def modifedStyleTransfer(opt):
    print('loading VGG models......')
    model = VGGRC(opt).cuda()
    model.load_state_dict(torch.load('checkpoints/rec/%s/model_%d.pth' % ('VGGRC_style', opt.start)))
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
        style_feats = model.VGG(data[0].cuda())
        input_feats = model.VGG(data[1].cuda())
        totalLoss = OverAllLoss(opt)
        Maps = FeatureMap(opt)
        Maps.mapload(input_feats, style_feats, len(model.VGG.layers))
        Maplist = Maps.featureList

        out = model.net_forward(Maplist)
        temp_image = make_grid(torch.clamp(DN(out[0]).unsqueeze(0), 0, 1), nrow=1, padding=0, normalize=False)
        train_writer.add_image('outimg', temp_image, k)

        index = 0
        view_shape = (4, int(Maplist[index].shape[1]/4), Maplist[index].shape[2],  Maplist[index].shape[3])
        print(view_shape)
        temp_image = make_grid(Maplist[index].reshape(view_shape)[:,:3,:,:], nrow=4, padding=0, normalize=True)
        train_writer.add_image('Gain Map', temp_image, 0)
        temp_image = make_grid(style_feats[index].reshape(view_shape)[:,:3,:,:], nrow=4, padding=0, normalize=True)
        train_writer.add_image('style_feat_4', temp_image, 0)
        temp_image = make_grid(input_feats[index].reshape(view_shape)[:,:3,:,:], nrow=4, padding=0, normalize=True)
        train_writer.add_image('input_feat_4', temp_image, 0)

        images += 1
        total_iter = opt.iter+1
            
        # Set optimizer.
        if 'image' in opt.optimode:
            # Initialize the output.
            output = out.cuda()
            #output.requires_grad = True
            output = torch.nn.Parameter(output, requires_grad=True)
            optimizer = torch.optim.LBFGS([output], lr=opt.lr)
        else:
            total_iter *= 30
            for layer in range(5):
                Maplist[layer] = torch.nn.Parameter(Maplist[layer], requires_grad=True)
            optimizer = torch.optim.Adam(Maplist, lr=1e-2, weight_decay=1e-8)

        optimizer.zero_grad()

        # define total iteration number. 
        pbar = tqdm(total=total_iter)

        for iters in range(total_iter):
            
            def closure():
                input_feats = model.VGG(output)
                optimizer.zero_grad()
                Loss_gain = 0
                Loss_style = 0
                Loss = 0
                for i in range(len(model.VGG.layers)):
                    loss_gain_item, loss_style_item = totalLoss.forward(style_feats[i], input_feats[i],
                                                        Map=Maps.featureList[i], mode=model.VGG.layers[i])

                    Loss_gain += loss_gain_item
                    Loss_style += loss_style_item
                # loss term
                Loss = Loss_gain + Loss_style
                Loss.backward(retain_graph=True)
                return Loss

            # Updates.
            #Loss.backward(retain_graph=True)
            if 'image' in opt.optimode:
                optimizer.step(closure)
                if iters%opt.iter_show == 0:
                    # record result pics.
                    temp_image = make_grid(torch.clamp(DN(output[0]).unsqueeze(0),0,1), nrow=opt.batch_size, padding=0, normalize=False)
                    train_writer.add_image('temp result', temp_image, iters+images*opt.iter)

                if iters%(10) == 0:
                    save_image(temp_image, "./checkpoints/%s/%d_%d.png"%(opt.outf, k, iters))
            else:
                optimizer.zero_grad()
                input_feats = Maplist
                Loss_gain = 0
                Loss_style = 0
                Loss = 0
                for i in range(len(model.VGG.layers)):
                    loss_gain_item, loss_style_item = totalLoss.forward(style_feats[i], input_feats[i],
                                                        Map=Maps.featureList[i], mode=model.VGG.layers[i])

                    Loss_gain += loss_gain_item
                    Loss_style += loss_style_item
                # loss term
                Loss = Loss_gain + Loss_style
                Loss.backward(retain_graph=True)
                optimizer.step()
                if iters%(opt.iter_show*10) == 0:
                    # record result pics.
                    output = model.net_forward(Maplist)
                    temp_image = make_grid(torch.clamp(DN(output[0]).unsqueeze(0),0,1), nrow=opt.batch_size, padding=0, normalize=False)
                    train_writer.add_image('temp result', temp_image, iters+images*total_iter)

                if iters%(100) == 0:
                    save_image(temp_image, "./checkpoints/%s/%d_%d.png"%(opt.outf, k, iters))

            #optimizer.zero_grad()
            pbar.update(1)

        pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    OptionInit = FeatureOptions(parser)
    parser = OptionInit.initialize(parser)
    opt = parser.parse_args()
    modifedStyleTransfer(opt)