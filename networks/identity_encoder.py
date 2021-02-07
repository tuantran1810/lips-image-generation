import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
import numpy as np
from nets import Conv2dBlock, Conv3dBlock

class IdentityEncoder(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 256, init_inner_channels = 64, nlayers = 3):
        super(IdentityEncoder, self).__init__()
        channels = init_inner_channels
        layers = [Conv2dBlock(in_channels, channels, kernel = 7, stride = 2, padding = 3)]
        for _ in range(nlayers - 1):
            conv_block = Conv2dBlock(
                channels, 
                out_channels,
                kernel = 3, 
                stride = 2, 
                padding = 1,
            )
            channels = out_channels
            layers.append(conv_block)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class StreamEncoder(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 256, init_inner_channels = 64, nlayers = 4):
        super(StreamEncoder, self).__init__()
        layers = [
            Conv3dBlock(in_channels, init_inner_channels, kernel = 4, stride = 2, padding = 1),
            Conv3dBlock(init_inner_channels, out_channels, kernel = 4, stride = 2, padding = 1),
            Conv3dBlock(out_channels, out_channels, kernel = 4, stride = 2, padding = 1),
            Conv3dBlock(out_channels, out_channels, kernel = 4, stride = 2, padding = 1),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).squeeze(2)

class FlowEncoder(nn.Module):
    def __init__(self, in_channels = 2, out_channels = 128, init_inner_channels = 32, nlayers = 3):
        super(FlowEncoder, self).__init__()
        channels = init_inner_channels
        layers = [Conv3dBlock(in_channels, channels, kernel = (3, 7, 7), stride = (2, 4, 4), padding = (1, 3, 3))]
        for _ in range(nlayers - 1):
            cout = channels * 2
            conv_block = Conv3dBlock(
                channels, 
                cout if cout < out_channels else out_channels,
                kernel = (3, 7, 7), 
                stride = (2, 4, 4), 
                padding = (1, 3, 3),
            )
            if cout < out_channels:
                channels = cout
            layers.append(conv_block)
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(out_channels * 2, out_channels)
        self.__out_channels = out_channels

    def forward(self, x):
        '''
        batch video shape is: (batch, frames, channels, h, w)
        '''
        batch_size = x.shape[0]
        x = x.transpose(1, 2)
        x = self.layers(x)
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    import pickle
    v_enc = IdentityEncoder()
    print(v_enc)
    s_enc = StreamEncoder()
    print(s_enc)
    with open("./../grid-dataset/sample/s1/bbaf2n.pkl", 'rb') as fd:
        data = pickle.load(fd)
        video = data['video']
        print(video.shape)
        video = torch.tensor(video)
        processed_video = v_enc(video.float())
        print(processed_video.shape)
        print("---------------------------------------------")
        sample = torch.ones(59, 3, 16, 64, 64)
        output = s_enc(sample.float())
        print(output.shape)
