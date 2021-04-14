import torch
from torch import nn
from models.basic_block import *

class UNetGenerator(nn.Module):
    '''
    UNet Class
    A series of 4 contracting blocks followed by 4 expanding blocks to 
    transform an input image into the corresponding paired image, with an upfeature
    layer at the start and a downfeature layer at the end.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels):
        super(UNetGenerator, self).__init__()
        self.contract1 = ContractingBlock(input_channels)
        self.contract2 = ContractingBlock(input_channels*2)
        self.contract3 = ContractingBlock(input_channels*4)
        self.contract4 = ContractingBlock(input_channels*8)

        self.expand1 = ExpandingBlock(input_channels*16)
        self.expand2 = ExpandingBlock(input_channels*8)
        self.expand3 = ExpandingBlock(input_channels*4)
        self.expand4 = ExpandingBlock(input_channels*2)

    def forward(self, x):
        x0 = self.contract1(x) #x0.shape torch.Size([1, 6, 32, 32, 32])
        x1 = self.contract2(x0) #x1.shape torch.Size([1, 12, 16, 16, 16])
        x2 = self.contract3(x1) #x2.shape torch.Size([1, 24, 8, 8, 8])
        x3 = self.contract4(x2) #x3.shape torch.Size([1, 48, 4, 4, 4])

        x4 = self.expand1(x3, x2) #x4.shape torch.Size([1, 24, 8, 8, 8])
        x5 = self.expand2(x4, x1) #x5.shape torch.Size([1, 12, 16, 16, 16])
        x6 = self.expand3(x5, x0) #x6.shape torch.Size([1, 6, 32, 32, 32])
        x7 = self.expand4(x6, x) #x7.shape torch.Size([1, 3, 64, 64, 64])

        return x7

def test():
    net_input = torch.randn(1, 3, 64, 64, 64) #16为depth,4是通道数
    gen = UNetGenerator(3) #注意这里设置为3的话，就要求net_input必须是3个通道的

    res = gen(net_input)
    print('generator output shape:', res.shape) #generator output shape: torch.Size([1, 3, 64, 64, 64])

#test()

