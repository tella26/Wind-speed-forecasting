"""

Below is a ResNet18 architecture:

There are 4 convolutional layers in each module (excluding the 1×1 convolutional layer). 
Together with the first 7×7 convolutional layer and the final fully-connected layer, there are 
18 layers in total. Therefore, this model is a ResNet-18.
"""
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
#from torch.utils.tensorboard import SummaryWriter

#writter = SummaryWriter("runs/reswind")

class Residual(nn.Module):
    def __init__(self, in_channel, out_channel, use_1x1Conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        if use_1x1Conv:
            self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, X):
        out = F.relu(self.bn1(self.conv1(X)))
        out = self.bn2(self.conv2(out))
        if self.conv3:
            X = self.conv3(X)
        out += X
        return F.relu(out)

def residualBlock(in_channel, out_channel, num_residuals, first_block=False):
    blks = []
    for i in range(num_residuals):
        if i==0 and not first_block:
            blks.append(Residual(in_channel, out_channel, use_1x1Conv=True,
            strides=2))
        else:
            blks.append(Residual(out_channel, out_channel))
    
    return blks



class ResNet_wind(nn.Module):
    def __init__(self, input_channel, n_classes):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*residualBlock(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*residualBlock(64, 128, 2))
        self.b4 = nn.Sequential(*residualBlock(128, 256, 2))
        self.b5 = nn.Sequential(*residualBlock(256, 512, 2))
        self.finalLayer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),nn.Linear(512, n_classes))

        self.b1.apply(self.init_weights)
        self.b2.apply(self.init_weights)
        self.b3.apply(self.init_weights)
        self.b4.apply(self.init_weights)
        self.b5.apply(self.init_weights)
        self.finalLayer.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Conv2d:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out')
        if type(layer) == nn.Linear:
            nn.init.normal_(layer.weight, std=1e-3)
        if type(layer) == nn.BatchNorm2d:
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)
        


    def forward(self, X):
        out = self.b1(X)
        out = self.b2(out)
        out = self.b3(out)
        out = self.b4(out)
        out = self.b5(out)
        out = self.finalLayer(out)

        return out
    
#model = ResNet(input_channel , n_classes ).to(device)
#writter.add_graph(model, download_data)
#writter.close
#sys.exit()