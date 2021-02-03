import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
import numpy as np
from nets import Conv2dBlock

class AudioEncoderGen(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 256, init_inner_channels = 32, nlayers = 4):
        super(AudioEncoderGen, self).__init__()
        channels = init_inner_channels
        layers = [Conv2dBlock(in_channels, channels, kernel = 1)]
        for _ in range(nlayers - 2):
            cout = channels * 2
            conv_block = Conv2dBlock(
                channels, 
                cout if cout < out_channels else out_channels,
                kernel = 3, 
                stride = (2, 2), 
                padding = 1,
            )
            if cout < out_channels:
                channels = cout
            layers.append(conv_block)
        layers.append(
            Conv2dBlock(channels, out_channels, kernel = 3, stride = (1, 2), padding = 1)
        )
        layers.append(nn.MaxPool2d(kernel_size = 3, stride = (1, 2), padding = 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class AudioEncoderDis(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 256, init_inner_channels = 32, nlayers = 4):
        super(AudioEncoderDis, self).__init__()
        channels = init_inner_channels
        layers = [Conv2dBlock(in_channels, channels, kernel = 1)]
        for _ in range(nlayers - 2):
            cout = channels * 2
            conv_block = Conv2dBlock(
                channels, 
                cout if cout < out_channels else out_channels,
                kernel = 3, 
                stride = 2, 
                padding = 1,
            )
            if cout < out_channels:
                channels = cout
            layers.append(conv_block)
        layers.append(
            Conv2dBlock(channels, out_channels, kernel = 3, stride = 2, padding = 1)
        )
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(out_channels * 16 * 8, out_channels * 4)
        self.__out_channels = out_channels

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.layers(x)
        x = x.reshape(batch_size, -1)
        return self.fc(x).reshape(batch_size, self.__out_channels, -1)

if __name__ == "__main__":
    import pickle
    a_enc = AudioEncoderGen()
    print(a_enc)
    a_enc_dis = AudioEncoderDis()
    print(a_enc_dis)

    def __produce_audio_batch(audio, offset):
        tmp = audio[offset:offset+16, :, :]
        tmp = torch.flatten(tmp, start_dim = 0, end_dim = 1)
        return tmp

    with open("./../grid-dataset/sample/s1/bbaf2n.pkl", 'rb') as fd:
        data = pickle.load(fd)
        audio = data['audio']
        audio = torch.tensor(np.transpose(audio, (0, 2, 1)))
        audio_arr = [
            __produce_audio_batch(audio, i) for i in range(0, audio.shape[0]-16)
        ]
        audio = torch.stack(audio_arr)
        audio = torch.unsqueeze(audio, axis = 1)
        
        print(audio.shape)
        processed_audio = a_enc(audio.float())
        print(processed_audio.shape)
        print("-----------------------------------------------")
        processed_audio = a_enc_dis(audio.float())
        print(processed_audio.shape)
