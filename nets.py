import torch
from torch import nn

class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride = 1, padding = 0):
        super(Conv1dBlock, self).__init__()
        self.blocks = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        return self.blocks(x)

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride = 1, padding = 0):
        super(Conv2dBlock, self).__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        return self.blocks(x)

class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride = 1, padding = 0):
        super(Conv3dBlock, self).__init__()
        self.blocks = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        return self.blocks(x)

class Resnet3dBlock(nn.Module):
    def __init__(self, channels, use_dropout = True, use_bias = True):
        super(Resnet3dBlock, self).__init__()
        layers = [
            nn.ReplicationPad3d(1),
            nn.Conv3d(channels, channels, kernel_size = 3, padding = 0, bias = use_bias),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace = True),
        ]
        if use_bias:
            layers.append(nn.Dropout(0.5))
        layers.extend([
            nn.ReplicationPad3d(1),
            nn.Conv3d(channels, channels, kernel_size = 3, padding = 0, bias = use_bias),
            nn.BatchNorm3d(channels),
        ])

        self.blocks = nn.Sequential(*layers)
    
    def forward(self, x):
        return x + self.blocks(x)

class Deconv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride = 1, padding = 0, dilation = 1, output_padding = 0):
        super(Deconv3dBlock, self).__init__()
        self.blocks = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel, stride, padding, dilation = dilation, output_padding = output_padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        return self.blocks(x)
