import torch
import torch.nn as nn
import torch.nn.functional as F

class MapUpsample(nn.Module):
    def __init__(self, d_model):
        super(MapUpsample, self).__init__()

        self.conv1=nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(6,6), padding=0, stride=6)

    def forward(self, x):
        B, _, C = x.shape
        y = x[:,-2:,:]
        x = x[:,:-2,:]
        x = x.permute(0, 2, 1).reshape(B*C,1,-1,360)
        x = self.conv1(x)
        x = x.reshape(B,C,-1).permute(0, 2, 1)
        return torch.cat([x, y], 1)
    
class MapDownsample(nn.Module):
    def __init__(self, d_model):
        super(MapDownsample, self).__init__()

        self.conv1=nn.Conv1d(in_channels=64800//(6*6), out_channels=64800, kernel_size=1, padding=0)

    def forward(self, x):
        B, _, C = x.shape
        y = x[:,-2:,:]
        x = x[:,:-2,:]
        x = self.conv1(x)
        return torch.cat([x, y], 1)