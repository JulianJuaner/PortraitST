from __future__ import print_function
import torch.utils.data as data
import os
import torch
import random
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
import cv2

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.bmp', '.BMP',
]
# decolorization:
def decolor(input_):
    RGB2G = torch.FloatTensor([0.299, 0.587, 0.114]).view(1, -1, 1, 1)
    summation =  torch.sum(input_*RGB2G.cuda(), 1).unsqueeze(1)
    #print(summation, input_.shape)
    return summation
    
# denormalize
def de_norm():
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    return invTrans

# normalization step before puts in VGG.
def make_trans():
    trans = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    return trans

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


# The Style-Transfer Dataset.
# mode = Paired: ONE style example with ONE paired inputs;
#        Others: ONE style example with MULTIPLE inputs;
class ST_dataset(data.Dataset):
    def __init__(self, root, name='testing', mode='paired'):
        self.feat = make_dataset(os.path.join(root,name,'style'))
        self.input = make_dataset(os.path.join(root,name,'input'))
        self.mode = mode
        self.name = name
        self.trans = make_trans()

    def __getitem__(self, index):
        if 'paired' in self.mode:
            return [self.get_image(self.feat[index%len(self)], mode='style'),
             self.get_image(self.input[index%1], mode='input')]
        else:
            return [self.get_image(self.feat[0], mode='style'),
             self.get_image(self.input[index%len(self)], mode='input')]

    def __len__(self):
        return len(self.feat)

    # open and get a image tensor.
    def get_image(self, filename, mode='input'):
        src = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        
        if 'VGG' in self.mode:
            src = cv2.resize(src, (512, 512))
        else:
            src = cv2.resize(src, (256, 336))

        img = Image.fromarray(src)
        #tensor = torch.from_numpy(img.transpose(2,0,1))
        tensor = self.trans(img)
        if 'input' in mode:
            img = Variable(tensor, requires_grad=False)#, device='cuda')
        elif 'style' in mode:
            img = Variable(tensor, requires_grad=False)
        return img

# Dataset for reconstruction network.
class RC_dataset(data.Dataset):
    def __init__(self, root, name='train', mode='unpaired'):
        self.feat = make_dataset(os.path.join(root,'styles'))
        self.input = make_dataset(os.path.join(root,'ffhq'))
        self.mode = mode
        self.name = name
        self.trans = make_trans()

    def __getitem__(self, index):
        if 'train' in self.name:
            style = self.feat[(index)%(len(self.feat)-100)]
            #style = self.feat[random.randint(0, len(self.feat)-100)]
            face1 = self.input[random.randint(0, 67000)]
            face2 = self.input[random.randint(0, 67000)]
            print(face1, face2)
        elif 'test' in self.name:
            style = self.feat[random.randint(len(self.feat)-99, len(self.feat)-1)]
            face1 = self.input[random.randint(68000, 69000)]
            face2 = self.input[random.randint(68000, 69000)]
        #print(face, style)
        return [self.get_image(face1, 'input'), self.get_image(face2, 'input'),
                self.get_image(style, 'style')]

    def get_image(self, filename, mode):
        src = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        src = cv2.resize(src, (512, 512))
        img = Image.fromarray(src)

        tensor = self.trans(img)
       
        img = Variable(tensor, requires_grad=False) 
        return img

    def __len__(self):
        if 'train' in self.name: 
            return len(self.feat)-100
        else:
            return 10

# Dataset for coordinate prediction network.
class CN_dataset(data.Dataset):
    def __init__(self, root, name='train', mode='paired', length=1200):
        self.target = make_dataset(os.path.join(root,'target'))
        self.example = make_dataset(os.path.join(root,'example'))
        self.mode = mode
        self.name = name
        self.trans = make_trans()
        self.length = length

    def __getitem__(self, index):
        if 'train' in self.name:
            seed = random.randint(0, self.length - 1)
            target = self.target[seed]
            example = self.example[seed]
            target_resize = target.replace('target', 'target_resize')
            example_resize = example.replace('example', 'example_resize')
            
        elif 'test' in self.name:
            target = self.target[seed]
            example = self.example[seed]
            target_resize = target.replace('target', 'target_resize')
            example_resize = example.replace('example', 'example_resize')
        
        return [
                    self.get_image(target_resize, 'blur'),
                    self.get_image(example_resize, 'blur'),
                    self.get_image(target),
                    self.get_image(example)
                ]

    def get_image(self, filename, mode = 'noblur'):
        #print(filename)
        src = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        src = cv2.resize(src, (512, 512))
        #if mode == 'blur':
        #    src = cv2.GaussianBlur(src,(3,3),0)
        img = Image.fromarray(src)

        tensor = self.trans(img)
       
        img = Variable(tensor, requires_grad=False) 
        return img

    def __len__(self):
        return self.length