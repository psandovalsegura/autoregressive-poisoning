import torch
import torch.nn as nn
import numpy as np
from autoregressive_params import ALL_2x2_AR_FILTERS

class PerfectARModel(nn.Module):
    """A simple CNN with 10 filters of size 3x3 followed by 
       one pooling layer, and one fully connected layer which 
       produces a 10 dimensional output tensor.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=30, padding=0)
        self.fc1 = nn.Linear(10, 10)

        # Load pre-defined AR filters by constructing a (10,3,3,3) tensor
        predefined_filters = []
        for _, f in ALL_2x2_AR_FILTERS.items():
            three_channel_filter = torch.tensor(np.stack([f]*3)) 
            predefined_filters.append(three_channel_filter)
        predefined_filters = torch.stack(predefined_filters).float()
        self.conv1.weight.data = predefined_filters

        # Set linear layer to compute softmax(1-x_i)
        fc_weight = -1 * torch.eye(10)
        fc_bias = torch.ones(10)
        self.fc1.weight.data = fc_weight
        self.fc1.bias.data = fc_bias
    
    def forward(self, x, post_pooling_output=False):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        if post_pooling_output:
            return x
        x = x.view(-1, 10)
        x = self.fc1(x)
        x = nn.functional.softmax(x, dim=1) #nn.functional.relu(x)
        return x

    