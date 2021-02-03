import sys, os
sys.path.append(os.path.dirname(__file__))
import torch

class Inference:
    def __init__(self, model, modelpath, get_input, produce_output):
        self.__model = model
        self.__model.load_state_dict(torch.load(modelpath))
        self.__get_input = get_input
        self.__produce_output = produce_output

    def get_output(self, include_input = False):
        self.__model.eval()
        for inp in self.__get_input():
            out = self.__model(inp)
            yield self.__produce_output(out)
