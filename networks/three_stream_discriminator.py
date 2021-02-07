import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
from audio_encoder import AudioEncoderDis
from identity_encoder import StreamEncoder
from nets import Conv2dBlock
from flownet2.models import FlowNet2
from video_flownet import FlownetArgs, VideoFlownet

class ThreeStreamDisciminator(nn.Module):
    def __init__(self):
        super(ThreeStreamDisciminator, self).__init__()
        self.__audio_enc = AudioEncoderDis()
        self.__video_stream_enc = StreamEncoder()

        args = FlownetArgs(
            rgb_max = 1.0,
            fp16 = False,
        )
        flownet_img = FlowNet2(args)
        self.__flownet = VideoFlownet(flownet_img, fast_mode=False)

        self.__flow_enc = StreamEncoder(in_channels = 2)
        self.__layers = nn.Sequential(
            Conv2dBlock(in_channels = 256*3, out_channels = 512, kernel = 3, stride = 2, padding = 1),
            Conv2dBlock(in_channels = 512, out_channels = 1, kernel = 3, stride = 2, padding = 1),
        )

    def forward(self, audio, stream, img):
        '''
        audio: (batch, channels, t, values)
        stream: (batch, channels, t, w, h)
        img: (batch, channels, w, h)
        '''
        audio_feature = self.__audio_enc(audio)
        audio_feature = audio_feature.unsqueeze(2).repeat(1, 1, 4, 1)
        stream_feature = self.__video_stream_enc(stream)

        img = img.unsqueeze(2)
        stream = torch.cat((img, stream), axis = 2)
        stream = stream.transpose(1, 2)
        print(stream.shape)
        flow = None
        with torch.no_grad():
            flow = self.__flownet(stream)
        flow = flow.transpose(1, 2)

        flow_feature = self.__flow_enc(flow)
        combine = torch.cat((audio_feature, stream_feature, flow_feature), axis = 1)
        final = self.__layers(combine).squeeze(3).squeeze(2).squeeze(1)
        return torch.sigmoid(final)

if __name__ == "__main__":
    dis = ThreeStreamDisciminator().cuda()
    audio = torch.ones(30, 1, 64, 128).cuda()
    stream = torch.ones(30, 3, 16, 64, 64).cuda()
    img = torch.ones(30, 3, 64, 64).cuda()
    result = dis(audio.float(), stream.float(), img.float())
    print("done")
    print(result.shape)
