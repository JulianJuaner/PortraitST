import argparse
import os

class FeatureOptions():
    def __init__(self, parser):
        self.initialized = False
    def initialize(self, parser):
        parser.add_argument('--root', required=True, help='root path of test images')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--max_size', type=tuple, default=(1024, 1024), help='the maximum size of test images')
        self.initialized = True
        return parser