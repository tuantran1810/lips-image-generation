import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
import pickle
from networks.video_autoencoder import VideoAutoEncoder
from utils.path_dataset import PathDataset

class VideoAutoencoderTrainer:
    def __init__(self, dataroot, use_gpu = True):
        data_paths = list()
        for path, _ , files in os.walk(dataroot):
            for name in files:
                data_paths.append(os.path.join(path, name))
        dataset = PathDataset(data_paths, self.__data_postprocessing)
        self.__use_gpu = torch.cuda.is_available()

        params = {
            'batch_size': 32,
            'shuffle': True,
            'num_workers': 6,
        }
        self.__dataloader = DataLoader(dataset, **params)
        self.__model = VideoAutoEncoder()
        if self.__use_gpu:
            self.__model.cuda()
        self.__mse_loss = nn.MSELoss()
        self.__optim = torch.optim.Adam(self.__model.parameters(), lr = 0.001)

    def __data_postprocessing(self, fd):
        data = pickle.load(fd)
        video = data['video']
        video = video[:16,:,:,:]
        video = video.transpose(1, 0, 2, 3)
        video = torch.tensor(video).float()
        video = F.normalize(video, dim = 1)
        return video

    def train(self, epochs = 10):
        for epoch in range(epochs):
            self.__model.train()
            print(f"{'-'*30} epoch {epoch} {'-'*30}")
            for batch_data in self.__dataloader:
                if self.__use_gpu:
                    batch_data = batch_data.cuda()

                self.__optim.zero_grad()
                yhat = self.__model(batch_data)
                loss = self.__mse_loss(yhat, batch_data)
                loss.backward()
                self.__optim.step()
                print(f"end batch, loss = {loss}")

if __name__ == "__main__":
    v_autoenc = VideoAutoencoderTrainer("/media/tuantran/raid-data/dataset/GRID/mouth_images")
    v_autoenc.train()
