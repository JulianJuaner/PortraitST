import argparse
import os

class FeatureOptions():
    def __init__(self, parser):
        self.initialized = False

    def initialize(self, parser):

        # Dataset options
        parser.add_argument('--root', type=str, default='', required=True, help='root path of test images')
        parser.add_argument('--name', type=str, default='', required=True, help='')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--max_size', type=tuple, default=(1024, 1024), help='the maximum size of test images')

        # Weight parameters
        parser.add_argument('--alpha_3', type=float, default=0.5, help='layer preference of conv3 first term')
        parser.add_argument('--alpha_4', type=float, default=0.5, help='layer preference of conv4 first term')
        parser.add_argument('--beta_3', type=float, default=0.5, help='layer preference of conv3 second term')
        parser.add_argument('--beta_4', type=float, default=0.5, help='layer preference of conv4 second term')

        parser.add_argument('--iter', type=int, default=300, help='iterations of feed-forward and back-propagation')
        parser.add_argument('--gmin', type=float, default=0.7, help='lower bound clamp gain map')
        parser.add_argument('--gmax', type=float, default=5.0, help='upper bound clamp gain map')
        parser.add_argument('--gT', type=int, default=100, help='balance two terms in the total loss')

        self.initialized = True
        return parser