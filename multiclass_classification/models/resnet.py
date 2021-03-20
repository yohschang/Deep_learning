# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 11:56:06 2020

@author: YX
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
        self.fcn = True
        
        
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # print(out.size())
        if self.fcn:
            return out
        else:
            out = F.avg_pool2d(out, 4)
            out = F.avg_pool2d(out, out.size()[3])
            
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out

class fcn(nn.Module):
    def __init__(self):
        super().__init__()
        self.backnone = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=2)
        self.conv1 = nn.Conv2d(512,64,3,1,1)
        self.conv2 = nn.Conv2d(64,2,1,1)
        self.convtrans1 = nn.ConvTranspose2d(2,2, 16, 8, 4)
        self.convtrans1.weight.data = bilinear_kernel(2,2, 16)
    
    def forward(self, x):
        x = self.backnone(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.convtrans1(x)
        
        return x
    
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)
    


def ResNet18(num_classes = 10,multi_class = False):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes = 10,multi_class = False):
    return ResNet(BasicBlock, [3, 4, 6, 3],num_classes=num_classes)


def ResNet50(num_classes = 10,multi_class = False):
    return ResNet(Bottleneck, [3, 4, 6, 3],num_classes=num_classes)


def ResNet101(num_classes = 10,multi_class = False):
    return ResNet(Bottleneck, [3, 4, 23, 3],num_classes=num_classes)


def ResNet152(num_classes = 10,multi_class = False):
    return ResNet(Bottleneck, [3, 8, 36, 3],num_classes=num_classes)

def resnet_fcn():
    return fcn()

def test():

    # net = ResNet18(num_classes = 2)
    fcn = resnet_fcn()
    y = fcn(torch.randn(1, 1, 768,128))
    print(y.size())
    # print(fcn(y).size())

if __name__ == "__main__":
    test()
