import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
from audio_encoder import AudioEncoderCorr
from identity_encoder import FlowEncoder
from flownet2.models import FlowNet2
from video_flownet import FlownetArgs, VideoFlownet

class AudioDerivativeEncoder(nn.Module):
    def __init__(self):
        super(AudioDerivativeEncoder, self).__init__()
        self.__enc = AudioEncoderCorr()

    def forward(self, x):
        '''
        x: (batch, channels, t, values)
        output: (batch, vector)
        '''
        t_total = x.shape[2]
        x0 = x[:,:,:t_total-1,:]
        x1 = x[:,:,1:,:]
        x = x1 - x0
        return self.__enc(x)

class CorrelationNet(nn.Module):
    def __init__(self):
        super(CorrelationNet, self).__init__()
        self.__audio_derivative_enc = AudioDerivativeEncoder()
        self.__flow_enc = FlowEncoder()
        args = FlownetArgs(
            rgb_max = 1.0,
            fp16 = False,
        )
        flownet_img = FlowNet2(args)
        self.__flownet = VideoFlownet(flownet_img, fast_mode=False)
        self.__loss = nn.CosineEmbeddingLoss()

    def forward(self, audio_feature, gen_video):
        '''
        audio_feature: (batch, channels, t, values)
        gen_video: (batch, frames, channels, w, h)
        output: loss
        '''
        batch_size = audio_feature.shape[0]
        audio_vec = self.__audio_derivative_enc(audio_feature)
        flow = None
        with torch.no_grad():
            flow = self.__flownet(gen_video)
        flow_vec = self.__flow_enc(flow)
        one_corr = torch.ones(batch_size).float().cuda()
        return self.__loss(audio_vec, flow_vec, one_corr)

def main():
    corr = CorrelationNet().cuda()
    audio = torch.ones(59, 256, 16, 8).cuda()
    video = torch.ones(59, 16, 3, 64, 64).cuda()
    y = corr(audio.float(), video.float())
    print(y.shape)
    print(y)

if __name__ == "__main__":
    main()
