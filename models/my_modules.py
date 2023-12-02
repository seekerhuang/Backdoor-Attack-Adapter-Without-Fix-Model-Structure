import torch
from torch import nn
import torch.nn.functional as F



class ConvAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=3, padding=0, dilation=1, groups=1, bias=True):
        super(ConvAdapter, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.conv1_1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=self.bias)
        # #init for zero
        # self.conv1_1.weight.data.zero_()
        # if self.conv1_1.bias is not None:
        #     self.conv1_1.bias.data.zero_()


    def forward(self, x):
        # xinit=x
        x = self.conv(x)
        x = self.bn(x)
        # x = self.conv1_1(x)
        # x = self.conv2(x)
        return x
    