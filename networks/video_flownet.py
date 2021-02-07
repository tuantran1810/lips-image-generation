import sys, os
sys.path.append(os.path.dirname(__file__))
import random
from dataclasses import dataclass
import torch
from torch import nn
from flownet2.models import FlowNet2

@dataclass
class FlownetArgs:
    rgb_max: float = 255.0
    fp16: bool = True

class VideoFlownet(nn.Module):
    def __init__(self, model):
        super(VideoFlownet, self).__init__()
        self.__model = model

    def __infervideo(self, x):
        #video shape is: (frames, channels, h, w)
        n_frames = x.shape[0]
        batch_input = []
        for i in range(n_frames-1):
            first_frame = x[i,:,:,:]
            second_frame = x[i+1,:,:,:]
            inp = torch.stack([first_frame, second_frame])
            inp = inp.transpose(1, 0)
            batch_input.append(inp)
        batch_input = torch.stack(batch_input)
        return self.__model(batch_input)

    def forward(self, x):
        # batch video shape is: (batch, frames, channels, h, w)
        batch_size = x.shape[0]
        output = []
        for i in range(batch_size):
            output.append(self.__infervideo(x[i,:,:,:,:]))
        output = torch.stack(output)
        print(output.shape)
        return output

def main():
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    import flowiz as fz


    def writeFlow(name, flow):
        f = open(name, 'wb')
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)
        f.flush()
        f.close()

    def showFlow(rows, cols, video):
        videos: (frames, channels, w, h)
        fig, axis = plt.subplots(rows, cols)
        n_frames = video.shape[0]
        for i in range(rows):
            for j in range(cols):
                ith_frame = i*cols + j
                if ith_frame >= n_frames: break
                frame = video[i*cols + j,:,:,:].transpose(1, 2, 0)
                flow = fz.convert_from_flow(frame)
                axis[i][j].imshow(flow)
        plt.show()

    args = FlownetArgs(
        rgb_max = 1.0,
        fp16 = False
    )
    model = FlowNet2(args)
    tmp = torch.load("./flownet2/checkpoint/FlowNet2_checkpoint.pth.tar")
    model.load_state_dict(tmp["state_dict"])

    video_model = VideoFlownet(model).cuda()
    video_model.eval()

    videos = []
    paths = [
        "./../grid-dataset/sample/s1/bbaf2n.pkl",
        "./../grid-dataset/sample/s1/bbaf3s.pkl",
        "./../grid-dataset/sample/s2/bbaf1n.pkl",
        "./../grid-dataset/sample/s2/bbaf2s.pkl",
        "./../grid-dataset/sample/s3/bbaf1s.pkl",
        "./../grid-dataset/sample/s3/bbaf2p.pkl",
    ]

    for p in paths:
        with open(p, 'rb') as fd:
            data = pickle.load(fd)
            video = data['video']
            video = video/255.0
            video = torch.tensor(video)
            videos.append(video)

    videos = torch.stack(videos).cuda()
    video_flow = video_model(videos.float()).cpu().detach().numpy()
    for i in range(video_flow.shape[0]):
        showFlow(5, 15, video_flow[i,:,:,:,:])

if __name__ == "__main__":
    main()
