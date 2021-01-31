import torch
from torch import nn
from nets import Conv3dBlock, Deconv3dBlock, Resnet3dBlock

class Video3dEncoder(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 64):
        super(Video3dEncoder, self).__init__()
        channels = 4
        layers = [
            Conv3dBlock(in_channels = in_channels, out_channels = channels, kernel = 3, stride = 2, padding = 1)
        ]
        while True:
            if channels * 2 > out_channels:
                break
            layers.append(
                Conv3dBlock(in_channels = channels, out_channels = channels * 2, kernel = 3, stride = 2, padding = 1)
            )
            channels *= 2
        self.__layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.__layers(x)

class Video3dDecoder(nn.Module):
    def __init__(self, in_channels = 64, out_channels = 3):
        super(Video3dDecoder, self).__init__()
        channels = in_channels
        layers = []
        while True:
            if channels // 2 < out_channels:
                break
            layers.append(
                Deconv3dBlock(
                    in_channels = channels, 
                    out_channels = channels // 2, 
                    kernel = 3, 
                    stride = 2, 
                    padding = 1, 
                    output_padding = 1
                )
            )
            channels //= 2
        layers.append(
            Deconv3dBlock(
                in_channels = channels, 
                out_channels = out_channels, 
                kernel = 3, 
                stride = 2, 
                padding = 1, 
                output_padding = 1
            )
        )
        self.__layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.__layers(x)

class ResidualChain(nn.Module):
    def __init__(self, n = 6, channels = 64):
        super(ResidualChain, self).__init__()
        layers = []
        for _ in range(n):
            layers.append(Resnet3dBlock(channels))
            layers.append(nn.ReLU(inplace = True))
        self.__layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.__layers(x)

class VideoAutoEncoder(nn.Module):
    def __init__(self, video_channels = 3, hidden_channels = 64):
        super(VideoAutoEncoder, self).__init__()
        self.__encoder = Video3dEncoder(in_channels = video_channels, out_channels = hidden_channels)
        self.__residual = ResidualChain(n = 6, channels = hidden_channels)
        self.__decoder = Video3dDecoder(in_channels = hidden_channels, out_channels = video_channels)
        self.__layers = nn.Sequential(
            self.__encoder, 
            self.__residual, 
            self.__decoder,
        )

    def forward(self, x):
        return self.__layers(x)

    def getEncoder(self):
        return self.__encoder
    
    def getDecoder(self):
        return self.__decoder

    def getResidualChain(self):
        return self.__residual

if __name__ == "__main__":
    auto_enc = VideoAutoEncoder()
    video = torch.ones(60, 3, 64, 64, 64)
    output = auto_enc(video)
    print(output.shape)
