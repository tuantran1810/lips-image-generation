import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
from nets import Resnet3dBlock, Deconv3dBlock, Conv3dBlock

class ImagesDecoder(nn.Module):
    def __init__(self, in_channels = 512, out_channels = 3):
        super(ImagesDecoder, self).__init__()
        self.layers = nn.Sequential(
            Resnet3dBlock(in_channels),
            Deconv3dBlock(in_channels = in_channels, out_channels = 256, kernel = 3, stride = (1, 2, 2), padding = 1, output_padding = (0, 1, 1)),
            Deconv3dBlock(in_channels = 256, out_channels = 128, kernel = 3, stride = (1, 2, 2), padding = 1, output_padding = (0, 1, 1)),
            Deconv3dBlock(in_channels = 128 , out_channels = 3, kernel = 7, stride = (1, 2, 2), padding = 3, output_padding = (0, 1, 1)),
        )

    def forward(self, x):
        return self.layers(x)

def main():
    tmp = torch.ones(59, 512, 16, 8, 8)
    dec = ImagesDecoder()
    print(dec)
    result = dec(tmp)
    print(result.shape)

if __name__ == "__main__":
    main()
