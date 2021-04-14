import torch
from torch import nn
from models.basic_block import *

class Discriminator(nn.Module):
    '''
    Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake. 
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    '''

    def __init__(self, in_channels):
        super(Discriminator,self).__init__()
        self.upfeature = FeatureMap3DBlock(in_channels, in_channels//2)
        self.contract1 = ContractingBlock(in_channels//2)
        self.contract2 = ContractingBlock(in_channels)
        self.contract3 = ContractingBlock(in_channels*2)
        self.contract4 = ContractingBlock(in_channels*4)

        self.featuremap = FeatureMap3DBlock(in_channels*8, 1)
    
    def forward(self, x, condition): ##x为label或fake, y为condition,低光照的图像作为label
        input_data = torch.cat([x, condition], axis=1)
        x0 = self.upfeature(input_data)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        x5 = self.featuremap(x4)

        return x5

def test():
    condition = torch.randn(1, 3, 64, 64, 64) #16为depth,4是通道数
    fake = torch.randn(1, 3, 64, 64, 64)
    disc = Discriminator(6)
    patch_res = disc(fake, condition)
    print(patch_res.shape) #torch.Size([1, 1, 4, 4, 4])
    one_labels = torch.zeros_like(patch_res)
    print(one_labels.shape)

#test()

