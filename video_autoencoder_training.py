import os
import random
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pickle
from networks.video_autoencoder import VideoAutoEncoder
from utils.path_dataset import PathDataset
from framework.trainer import Trainer

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
    data_paths = list()
    for path, _ , files in os.walk("/media/tuantran/rapid-data/dataset/GRID/mouth_images"):
        for name in files:
            data_paths.append(os.path.join(path, name))
    dataset = PathDataset(data_paths, __data_processing)
    params = {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 6,
    }
    dataloader = DataLoader(dataset, **params)
    model_backup = "./model-video-autoenc/final.pt"

    pre_model = VideoAutoEncoder()
    if model_backup != None:
        pre_model.load_state_dict(torch.load(model_backup))
    
    def __produce_data_train():
        for data in dataloader:
            yield data, data

    def __produce_data_test():
        for i, data in enumerate(dataloader):
            if i >= 10:
                return
            yield data, data

    def __inject_dataloader():
        return __produce_data_train, __produce_data_test

    def __inject_model():
        return pre_model

    def __inject_optim(model):
        return torch.optim.Adam(model.parameters(), lr = 0.001)

    def __inject_loss_fn():
        return nn.MSELoss()

    trainer = Trainer(
        "video-autoenc",
        __inject_dataloader,
        __inject_model,
        __inject_optim,
        __inject_loss_fn,
        100,
    )

    trainer.train(10)

if __name__ == "__main__":
    main()
