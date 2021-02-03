import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
from audio_encoder import AudioEncoderGen
from identity_encoder import IdentityEncoder
from decoder import ImagesDecoder

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.aenc = AudioEncoderGen()
        self.ienc = IdentityEncoder()
        self.dec = ImagesDecoder()

    def forward(self, audio, image):
        audio = self.aenc(audio)
        image = self.ienc(image)
        audio_ext = audio.unsqueeze(3).repeat(1, 1, 1, 8, 1)
        image_ext = image.unsqueeze(2).repeat(1, 1, 16, 1, 1)
        feature = torch.cat((audio_ext, image_ext), axis = 1)
        return self.dec(feature), audio

def main():
    import pickle

    def __produce_audio_batch(audio, offset):
        tmp = audio[offset:offset+16, :, :]
        tmp = torch.flatten(tmp, start_dim = 0, end_dim = 1)
        return tmp

    gen = Generator().cuda()
    print(gen)
    with open("./../grid-dataset/sample/s1/bbaf2n.pkl", 'rb') as fd:
        data = pickle.load(fd)
        video = torch.tensor(data['video'])
        print(video.shape)
        audio = torch.tensor(data['audio'].transpose(0,2,1))

        audio_arr = [
            __produce_audio_batch(audio, i) for i in range(0, audio.shape[0]-16)
        ]
        audio = torch.stack(audio_arr)
        audio = torch.unsqueeze(audio, axis = 1)
        
        audio_batch = audio.shape[0]
        images = video[0:audio_batch,:,:]
        gen_images, audio_feature = gen(audio.float().cuda(), images.float().cuda())
        print(gen_images.shape)
        print(audio_feature.shape)

if __name__ == "__main__":
    main()
