import sys, os
sys.path.append(os.path.dirname(__file__))
from torch.utils.data import Dataset

class PathDataset(Dataset):
    def __init__(self, pathlist, post_processing):
        self.__data_paths = pathlist
        self.__post_processing  = post_processing

    def __len__(self):
        return len(self.__data_paths)

    def __getitem__(self, index):
        path = self.__data_paths[index]
        with open(path, "rb") as fd:
            return self.__post_processing(fd)
