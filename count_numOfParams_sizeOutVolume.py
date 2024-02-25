import numpy as np
import torch.nn as nn
import torch

def count_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(2, 2), stride=2, padding=0, bias=True)
conv2d = nn.Conv2d(in_channels=8, out_channels=22, kernel_size=(5, 3), stride=(2,1), padding=0, bias=True)
conv3d = nn.Conv3d(in_channels=3, out_channels=8, kernel_size=(3,3,3), stride=1, padding=0, bias=False)
mlp_one_layer = nn.Sequential(nn.Linear(100,100))
mlp_3_layer = nn.Sequential(nn.Linear(100,20), nn.Linear(20, 100), nn.Linear(100,10))
mlp_2_layer_act = nn.Sequential(nn.Linear(5, 20), nn.Sigmoid(), nn.Linear(20, 10), nn.Sigmoid())
rnn = nn.RNN(input_size=256, hidden_size=512, bias=False)
lstm = nn.LSTM(input_size=256, hidden_size=512, bias=False)
gru = nn.GRU(input_size=256, hidden_size=512, bias=False)
multi_head_self_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, bias=False)



models = [
    conv1d, conv2d, conv3d, mlp_one_layer, mlp_3_layer, mlp_2_layer_act, rnn, lstm, gru,multi_head_self_attention,

    ]



print("Model".ljust(18) + "| Number of trainable parameters")
print("--------------------------------------------------")
for model in models:
    print(f"{model._get_name()}".ljust(18) + f"| {count_num_params(model)}")


# Put model from above here to eval output volume size
output_volume_check = conv2d
array_shape = (6, 8, 36, 36)

# output_volume_check = False
if(output_volume_check != False):
    # Create a numpy array of zeros with the specified dimensions
    zeros_array = torch.ones(array_shape)
    out = output_volume_check(zeros_array)
    print(f"\nOutput volume size of {model._get_name()}:")
    print(out.shape)
