import numpy as np
import torch.nn as nn


def count_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

models = [
    nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(2, 2), stride=2, padding=0, bias=True),
    nn.Conv2d(in_channels=8, out_channels=22, kernel_size=(5, 3), stride=(2, 1), padding=0, bias=True),
    nn.Sequential(nn.Linear(100,100)),
    nn.Sequential(nn.Linear(100,20), nn.Linear(20, 100), nn.Linear(100,10)),
    nn.Sequential(nn.Linear(10, 20), nn.PReLU(), nn.Linear(20, 8), nn.PReLU()),
    nn.RNN(input_size=256, hidden_size=512, bias=False),
    nn.LSTM(input_size=256, hidden_size=512, bias=False),
    nn.GRU(input_size=256, hidden_size=512, bias=False)
]



for model in models:
    print(f"{model._get_name()}".ljust(15) + f": {count_num_params(model)}")