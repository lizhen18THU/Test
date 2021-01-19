"""
这是针对于imageNet数据集的resnet网络结构，没有予以使用，仅做参考和学习
"""

import torch.nn as nn
import torch


class BasicBlock(nn.Module):  # 18层或34层残差网络的 残差模块
    expansion = 1  # 记录各个层的卷积核个数是否有变化

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)  # 有无bias对bn没多大影响
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.downsample = downsample

    def forward(self, x):
        identity = x  # 记录上一个残差模块输出的结果
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):  # 50层，101层，152层的残差网络的 残差模块
    expansion = 4  # 第三层卷积核的个数（256，512，1024，2048）是第一层或第二层的卷积核个数（64，128，256，512）的4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels 降维
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels 升维
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):  # 网络框架

    # 参数：block         如果定义的是18层或34层的框架 就是BasicBlock， 如果定义的是50,101,152层的框架，就是Bottleneck
    #       blocks_num   残差层的个数，对应34层的残差网络就是 [3,4,6,3]
    #       include_top  方便以后在resnet的基础上搭建更复杂的网络

    def __init__(self, block, blocks_num, num_classes=100, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  # 上一层的输出channel数，及这一层的输入channel数

        #   part 1 卷积+池化  conv1+pooling
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)  # 利用in-place计算可以节省内（显）存，同时还可以省去反复申请和释放内存的时间。但是会对原变量覆盖，只要不带来错误就用。计算结果不会有影响
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #   part 2  残差网络的四部分残差块：conv2,3,4,5
        self.layer1 = self._make_layer(block, 64, blocks_num[0])  # 5中不同深度的残差网络的第一部分残差块个数：2，3，3，3，3
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)  # 5中不同深度的残差网络的第一部分残差块个数：2，4，4，4，8
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)  # 5中不同深度的残差网络的第一部分残差块个数：2，6，6，23，36
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)  # 5中不同深度的残差网络的第一部分残差块个数：2，3，3，3，3
        # 五种深度分别是18,34,50,101，152
        #   part 3  平均池化层+全连接层
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 卷积层的初始化操作
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            # 虚线部分
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))  # stride=1，downsample=None

        return nn.Sequential(*layers)  # 将list转换为非关键字参数传入

    def forward(self, x):

        # part 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # part 2
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # part 3
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet18(num_classes=20, include_top=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def resnet34(num_classes=20, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=100, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
