import numpy as np
import torch 
from torch import nn
from torch.utils import data
from torch.nn import functional as F

#默认strides=1高宽不变
class Residual(nn.Module):
    def __init__(self, input_channels, output_channels, use_1x1conv=False, strides=1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, stride=strides)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(output_channels) 
        self.bn2 = nn.BatchNorm2d(output_channels)
        
    def forward(self, X):
        Y = F.leaky_relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(X))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.leaky_relu(Y)

#高宽减半
def resnet_block(input_channels, output_channels, num_residuals, first_block=True):
    blk = []
    for i in range(num_residuals):
        if i == 0 and first_block:
            blk.append(Residual(input_channels, output_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(output_channels, output_channels, first_block=False)) 
    return blk        
        
l1 = nn.Sequential(nn.Conv2d(input_channels, 32, kernel_size=3, ))        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

