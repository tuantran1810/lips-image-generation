import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
from audio_encoder import AudioEncoderDis
from identity_encoder import StreamEncoder
from nets import Conv2dBlock

class ThreeStreamDisciminator(nn.Module):
    def __init__(self):
        super(ThreeStreamDisciminator, self).__init__()
        self.__audio_enc = AudioEncoderDis()
        self.__video_stream_enc = StreamEncoder()
        self.__flow_enc = StreamEncoder(in_channels = 2)
        self.__layers = nn.Sequential(
            Conv2dBlock(in_channels = 256*3, out_channels = 512, kernel = 3, stride = 2, padding = 1),
            Conv2dBlock(in_channels = 512, out_channels = 1, kernel = 3, stride = 2, padding = 1),
        )

    def forward(self, audio, stream, flow):
        audio_feature = self.__audio_enc(audio)
        audio_feature = audio_feature.unsqueeze(2).repeat(1, 1, 4, 1)
        stream_feature = self.__video_stream_enc(stream)
        flow_feature = self.__flow_enc(flow)
        combine = torch.cat((audio_feature, stream_feature, flow_feature), axis = 1)
        final = self.__layers(combine).squeeze(3).squeeze(2).squeeze(1)
        return torch.sigmoid(final)

if __name__ == "__main__":
    dis = ThreeStreamDisciminator()
    audio = torch.ones(59, 1, 64, 128)
    stream = torch.ones(59, 3, 16, 64, 64)
    flow = torch.ones(59, 2, 16, 64, 64)
    result = dis(audio.float(), stream.float(), flow.float())
    print(result.shape)
