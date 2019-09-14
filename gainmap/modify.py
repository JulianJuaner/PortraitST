import time
import tensorboardX

from VGG import myVGG
from options import FeatureOptions
from torch.utils.data import DataLoader

class OverAllLoss():
    def __init__(self, opt):
        self.L1 = torch.nn.L1Loss().cuda()
        self.L2 = torch.nn.MSELoss().cuda()
        self.compare = torch.nn.MSELoss(reduction='sum').cuda()

        #set parameters
        self.alpha_3 = opt.alpha_3
        self.alpha_4 = opt.alpha_4
        self.beta_3 = opt.beta_3
        self.beta_4 = opt.beta_4
        self.gT = opt.gT
        self.gmin = opt.gmin
        self.gmax = opt.gmax
        self.sigma = 1e-4

    def forward(self, Style, Input, mode='conv3_1'):
        # A, B: 4-D tensors.
        Gain = torch.div(Style, Input+self.sigma)
        Mediate = torch.mul(Input, Gain)
        Mediate = torch.clamp(Mediate, min=gmin, max=gmax)

        if 'conv3' in mode:
            alpha = self.alpha_3
            beta = self.beta_3
        elif 'conv4' in mode:
            alpha = self.alpha_4
            beta = self.beta_4
        else:
            print('no correct mode.')

        Gain_loss = self.alpha_3 * self.L2(Input, Mediate) / 2
        Style_loss = self.gT * self.beta_3 /(2*math.pow(Style.shape[1], 2))\
                   * self.compare(torch.mul(Input, Input,transpose(0,1,3,2)),
                                   torch.mul(Style, Style,transpose(0,1,3,2)))
        
        return Gain_loss + Style_loss

def StyleTransfer(opt):
    conv3_model = myVGG(layer='conv3_1')
    conv4_model = myVGG(layer='conv4_1')
    
    dataloader = DataLoader(
            ImageDataset("../../data/%s" % opt.dataset_name, \
            "../../data/%s" % opt.dataset_name, \
            channel = opt.channels, datasize = opt.datasize, transforms_=None, unaligned=False, mask=True, opt = opt),
            batch_size=opt.batch_size, 
            shuffle=True,
            num_workers=0,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = FeatureOptions()
    parser = FeatureOptions.initialize(parser)
    opt = parser.parse_args()

    pass