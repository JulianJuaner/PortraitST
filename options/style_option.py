import argparse
import os

class StyleOptions():
    def __init__(self, parser):
        self.initialized = False

    def initialize(self, parser):

        allLayer = 'conv1_1,conv1_2,conv2_1,conv2_2,conv3_1,conv3_2,conv3_3,conv3_4,conv4_1,conv4_2,conv4_3,conv4_4,conv5_1,conv5_2,conv5_3,conv5_4'
        partialLayer = 'conv1_1,conv2_1,conv3_1,conv4_1,conv5_1'
        
        # Dataset options
        parser.add_argument('--root', type=str, default='../datasets',      help='root path of test images')
        parser.add_argument('--name', type=str, default='newset',         help='name of the dataset')
        parser.add_argument('--outf', type=str, default='image',            help='name of the dataset')
        parser.add_argument('--zoom_up', type=int, default=4,               help='scale to zoom up')
        parser.add_argument('--patch_size', type=int, default=1,            help='the patch size to optimize')
    
        # Training options
        parser.add_argument('--inchannel', type=int, default=6,             help='number of inpur channel in the network')
        parser.add_argument('--batch_size', type=int, default=4,            help='input batch size')
        parser.add_argument('--lr', type=float, default=1,                  help='learning rate')
        parser.add_argument('--iter_show', type=int, default=10,            help='iters to show the midate results')
        parser.add_argument('--layers', type=str, default=partialLayer)
        parser.add_argument('--epoch', type=int, default=50)
        parser.add_argument('--start', type=int, default=0)

        self.initialized = True
        return parser