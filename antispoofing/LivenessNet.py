import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.bn1   = nn.BatchNorm2d(self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2   = nn.BatchNorm2d(self.out_channels)
        self.pool  = nn.MaxPool2d(kernel_size=2)
        self.drop  = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.pool(x)
        out = self.drop(x)

        return out
        
class LivenessNet(nn.Module):
    def __init__(self, in_channels=3):
        super(LivenessNet, self).__init__()
        
        self.in_channels = in_channels
        
        # first CONV => RELU => CONV => RELU => POOL layer set
        self.block1 = ConvBlock(self.in_channels, 16)
        
        # second CONV => RELU => CONV => RELU => POOL layer set
        self.block2 = ConvBlock(16, 32)
        
        # first (and only) set of FC => RELU layers
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(2048, 64)
        self.relu5   = nn.ReLU()
        self.bn5     = nn.BatchNorm1d(64)
        self.drop3   = nn.Dropout(0.5)
        
        self.linear2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu5(x)
        x = self.bn5(x)
        x = self.drop3(x)
        x = self.linear2(x)
        
        x = x.view(-1)
        out = self.sigmoid(x)
        
        return out