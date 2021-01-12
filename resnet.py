import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(in_channels, out_channels, stride = 1):
        super(self, ResidualBlock).__init__()

        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, filter_size = 3, stride = stride, Pad = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, filter_size = 3, stride = stride, Pad = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip_layer = nn.Sequential()

        if in_channels != out_channels or stride != 1:
            self.skip_layer = nn.Sequential(
                nn.Conv2d(in_channels = in_channels, out_channels = out_channels, stride = stride, Pad = None),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        x = nn.ReLu(self.conv1(x))
        x = nn.bn1(x)
        x = nn.ReLu(self.conv2(x))
        x = nn.bn2(x)
        x += self.skip_layer(x)
        x = nn.ReLu()(x)

        return x

class ResNet18(nn.Module):
    def __init__ (self, num_classes):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, filter_size = 7, stride = 2, Pad = None, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride = 2)

        self.layer1 = self._create_block_(64, 64, 2)
        self.layer2 = self._create_block_(64,128, 2)
        self.layer3 = self._create_block_(128, 256, 2)
        self.layer4 = self._create_block_(256, 512, 2)
        self.fc = nn.Linear(512, num_classes)

    def _create_block_(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels = in_channels, out_channels = out_channels, stride = 2),
            ResidualBlock(in_channels = out_channels, out_channels = out_channels, stride=1)
        )

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = nn.ReLu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = nn.AvgPool2d(4)(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x 