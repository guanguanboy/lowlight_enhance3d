import torch
from torch import nn
from torch.nn.parameter import Parameter

"""
3D channel attention
"""
class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel , bias=False),
            nn.ReLU(),
            nn.Linear(channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _,_ = x.size()
        #print(b, c)
        y = self.avg_pool(x).view(b, c) #squeeze produces a channel descriptor by aggregating feature maps across their spatial dimensions
        #print(y.shape)
        y = self.fc(y).view(b, c, 1, 1, 1)
        #print('attention map shape:', y.shape) #torch.Size([16, 64, 1, 1, 1])
        return x * y.expand_as(x) # *号表示哈达玛积，既element-wise乘积, 用输入x乘以attention map


class SELayer3DNew(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer3DNew, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//4 , bias=False),
            nn.ReLU(),
            nn.Linear(channel//4, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, band_num, _,_ = x.size()
        #print(b, c)
        y = self.avg_pool(x).view(b, c, band_num) #squeeze produces a channel descriptor by aggregating feature maps across their spatial dimensions
        #print(y.shape)
        y = self.fc(y).view(b, c, band_num, 1, 1)
        #print('attention map shape:', y.shape) #torch.Size([16, 64, 1, 1, 1])
        return x * y.expand_as(x) # *号表示哈达玛积，既element-wise乘积, 用输入x乘以attention map


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
class eca_layer3d(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer3d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.conv = nn.Conv1d(channel, channel, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, depth, h, w = x.size()
        #print(b, c, depth, h, w)
        # feature descriptor on the global spatial information
        y = self.avg_pool(x) #将最后两维压缩掉了

        #print('y.shape =', y.shape) #(b, c, 1, 1)
        #print('y.squeeze = ',y.squeeze(-1).shape)#y.squeeze(-1)表示压缩掉最后一维

        #print('y.squeeze.squeeze shape =', y.squeeze(-1).squeeze(-1).shape)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1) #unsqueeze(-1) 表示在最后一维增加一个维度
        #print('after y shape =', y.shape)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x) #最后使用获得的注意力对原始输入进行了加权

def test():
    input = torch.randn(1, 32, 64, 64)
    efficient_channel_attention = eca_layer(32)
    output = efficient_channel_attention(input)
    print(output.shape) #torch.Size([1, 32, 64, 64]),也就是输出的shape与输入的shape是相同的

def test_ecalayer_3d():
    features = torch.randn(16, 64, 64,48, 32)

    selayer = eca_layer3d(64)

    feature_out = selayer(features) #通过SElayer之后，features的shape还是保持不变torch.Size([16, 64, 64, 48, 32])
    print(feature_out.shape)    

#test()
#test_ecalayer_3d()