import os
import sys 
import argparse

sys.path.insert(1, './gainmap')
sys.path.insert(1, './alignment')
sys.path.insert(1, './options')

from dataset import *
from modify import *
from test_option import TestOptions
from alignment import load_image_and_morph

parser = argparse.ArgumentParser()
OptionInit = TestOptions(parser)

load_image_and_morph() 