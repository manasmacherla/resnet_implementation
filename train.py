from resnet import ResNet18
import torch
import torch.nn as nn

myNet = ResNet18(1000)
myNet.load_state_dict(torch.load("resnet18-5c106cde.pth"))