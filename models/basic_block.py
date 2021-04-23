import torch
from torch import nn
from models.attention import SELayer3D, SELayer3DNew, eca_layer, eca_layer3d

"""

功能描述：
double channel number, half depth,width and height.
"""
class ContractingBlock(nn.Module):
    def __init__(self, input_channels):
        super(ContractingBlock, self).__init__()

        self.conv3d_1 = nn.Conv3d(in_channels=input_channels, out_channels=input_channels*2, \
            kernel_size=(3, 3, 3),stride=(1, 1, 1), padding=(1, 1, 1))
        self.activation = nn.ReLU()
        self.conv3d_2 = nn.Conv3d(in_channels=input_channels*2, out_channels=input_channels*2, \
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.max_pooling3d = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv3d_half = nn.Conv3d(in_channels=input_channels*2, out_channels=input_channels*2, \
            kernel_size=4, stride=2, padding=1)
        #self.channel_atten = SELayer3DNew(input_channels*2)
        #self.eca_layer3d = eca_layer3d(input_channels*2)

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.activation(x)
        x = self.conv3d_2(x)
        x = self.activation(x)
        
        x = self.conv3d_half(x)
        #weighted_featuremap = self.channel_atten(x)

        #x = weighted_featuremap
        #featuremap = self.eca_layer3d(x)
        #x = featuremap
        return x

"""
function:
half input channel number, double width, height and depth
"""
class ExpandingBlock(nn.Module):
    def __init__(self, input_channels):
        super(ExpandingBlock, self).__init__()

        self.conv3d_1 = nn.Conv3d(in_channels=input_channels, out_channels=input_channels//2, \
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(in_channels=input_channels, out_channels=input_channels//2, \
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(in_channels=input_channels//2, out_channels=input_channels//2, \
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.activation = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True) #该函数可以处理3D tensor
        self.transpose = nn.ConvTranspose3d(in_channels=input_channels, out_channels=input_channels, \
            kernel_size=4, stride=2, padding=1)
        #self.channel_atten = SELayer3DNew(input_channels//2)
        #self.eca_layer3d = eca_layer3d(input_channels//2)

    def forward(self, x, skip_con_x):
        x = self.upsample(x)
        #x = self.transpose(x)
        x = self.conv3d_1(x)
        x = torch.cat([x, skip_con_x], axis=1)

        x = self.conv3d_2(x)
        x = self.activation(x)

        x = self.conv3d_3(x)
        x = self.activation(x)

        #weight_featuremap = self.channel_atten(x)

        #x = weight_featuremap
        #atten_weighted_featuremap = self.eca_layer3d(x)
        #x = atten_weighted_featuremap

        return x



"""
函数功能：
通过1x1卷积改变输入输出通道的数目

"""
"""
    FeatureMapBlock Class
    The final layer of a U-Net - 
    maps each pixel to a pixel with the correct number of output dimensions
    using a 1x1 convolution.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output

    总结起来feature map类的作用就是通过1x1的卷积变换通道的数目，
    当通道数不相同时，可以使用此函数，
    但是此函数实现的时候，使用的是2D卷积，因此只能处理类似二维的图像数据，对于3维数据
    的处理，最好使用三维卷积
"""
class FeatureMap3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureMap3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x



def test():
    net_input = torch.randn(1, 4, 16, 160, 160) #16为depth,4是通道数
    contract_block = ContractingBlock(4)

    res = contract_block(net_input)
    print(res.shape) #torch.Size([1, 8, 8, 80, 80])

    net_input = torch.randn(1, 4, 16, 64, 64) #16为depth,4是通道数
    skip_con_x_sam = torch.randn(1, 2, 32, 128, 128)
    expand_block = ExpandingBlock(4)

    res = expand_block(net_input, skip_con_x_sam)
    print(res.shape) #torch.Size([1, 2, 32, 128, 128])

    net_input = torch.randn(1, 4, 16, 64, 64) #16为depth,4是通道数
    featuremapblock = FeatureMap3DBlock(4, 6) #通过这个将4通道变成6通道
    res = featuremapblock(net_input)
    print('featuremap shape = ', res.shape) #featuremap shape =  torch.Size([1, 6, 16, 64, 64])

"""
#测试函数
test()
"""
test()