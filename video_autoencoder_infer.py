import os
import random
import torch
import cv2
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pickle
from networks.video_autoencoder import VideoAutoEncoder
from utils.path_dataset import PathDataset
from framework.inference import Inference
from loguru import logger as log
from matplotlib import pyplot as plt

def __data_processing(fd):
    data = pickle.load(fd)
    video = data['video']
    total_frames = video.shape[0]
    i = random.randint(0, total_frames - 16)
    video = video[i:i+16,:,:,:]
    video = video.transpose(1, 0, 2, 3)
    video = torch.tensor(video).float()
    video = video/255.0
    return video

def main():
    log.info("start inference")
    data_paths = list()
    for path, _ , files in os.walk("./grid-dataset/sample"):
        for name in files:
            data_paths.append(os.path.join(path, name))

    dataset = PathDataset(data_paths, __data_processing)
    params = {
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 1,
    }
    dataloader = DataLoader(dataset, **params)

    def __get_input():
        return dataloader

    def __produce_output(out):
        out = out.squeeze(0)
        out = out.detach().numpy()
        return out.transpose(1, 2, 3, 0)

    inference = Inference(
        VideoAutoEncoder(),
        "./model-video-autoenc/final.pt",
        __get_input,
        __produce_output,
    )

    for out in inference.get_output():
        fig, axis = plt.subplots(4, 4)
        for i in range(4):
            for j in range(4):
                axis[i][j].imshow(cv2.cvtColor(out[i*4+j,:,:,:], cv2.COLOR_BGR2RGB))
        plt.show()

if __name__ == "__main__":
    main()
