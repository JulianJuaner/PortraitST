from __future__ import print_function
import torch.utils.data as data
import os
import torch
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
import cv2

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.bmp', '.BMP',
]

# normalization step before puts in VGG.
def make_trans():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    return normalize

# test file by extension
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# find all image files and store paths as a list.
def make_dataset(dir):
    images = []
    print(dir)
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    images.sort()
    return images

# open and get a image tensor.
def getImage(filename):
    src = cv2.imread(filename,1)
    img = src.transpose(2,0,1).astype(np.float32)
    img = Variable(torch.from_numpy(img), requires_grad=False)
    return img

# The Style-Transfer Dataset.
# mode = Paired: ONE style example with ONE paired inputs;
#        Others: ONE style example with MULTIPLE inputs;
class ST_dataset(data.Dataset):
    def __init__(self, root, name='testing', mode='paired'):
        self.file_list = file_list
        self.feat = make_dataset(os.path.join(root,name,'style'))
        self.input = make_dataset(os.path.join(root,name,'example'))
        self.mode = mode
        self.name = name
        self.trans = make_trans()

    def __getitem__(self, index):
        if self.mode == 'paired':
            return [get_image(feat[index%len(self)]), get_image(input[index%len(self)])]
        else:
            return [get_image(feat[0]), get_image(input[index%len(self)])]

    def __len__(self):
        return len(self.input)