import torch
import torch.utils.data as Data
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
import WideResnet
from main import Fc_layer

# print(1//10)
# a=np.load("./data/train.npy")
# b=np.load("./data/train_handle100.npy")

# print(a[0])
# print(b[47500])
# print(a[1])
# print(b[27500])
# print(a[54])
# print(b[500*72+8])

# if not False and True:
#     print("test")
# a = torch.IntTensor([])
# b = torch.IntTensor([1,2,3])
# a = torch.cat((a, b))
# print(a.type())
# net = WideResnet.WideResnet28_10(28, 10, 20).cuda()
# # net = ResNest.resNest41(num_classes).to(device)
# fc = Fc_layer(net.feature_num, 20).cuda()
# # fc = None
# optimizer = optim.SGD([{'params': net.parameters()},
#                        {'params': fc.parameters()}], lr=0.1, momentum=0.9,
#                       weight_decay=0.9, nesterov=True)
# print(optimizer.param_groups[0].keys())
# for param_group in optimizer.param_groups:
#     print(param_group.keys())
# epoch = 180
# fig = plt.figure(1, (4, 3))
# # x轴数据
# xdata = [i for i in range(1, epoch + 1)]
# # 画出训练数据的错误率图像
# subFig1 = fig.add_subplot(2, 1, 1)
# subFig1.set_title("trainerror/loss-epoch figure")
# list_df = pd.read_csv("./halfTrained/list.csv").values
# trainErr_rate_list = list(list_df[:, 0])
# testErr_rate_list = list(list_df[:, 1])
# # 设定坐标轴范围
# subFig1.set_xlim([1, epoch + 1])
# ymax = max(trainErr_rate_list)
# ymin = min(trainErr_rate_list)
# ymin = min([ymin, 0])
# y_ticks = np.arange(ymin, ymax, 0.1)
# subFig1.set_ylim([ymin, ymax])
# plt.yticks(y_ticks, fontsize=6)
# subFig1.plot(xdata, trainErr_rate_list, color="tab:blue")
#
# # 画出测试数据的错误率图像
# subFig2 = fig.add_subplot(2, 1, 2)
# subFig2.set_title("testerror-epoch figure")
# # 设定坐标轴范围
# subFig2.set_xlim([1, epoch + 1])
# ymax = max(testErr_rate_list)
# ymin = min(testErr_rate_list)
# ymin = min([ymin, 0])
# y_ticks = np.arange(ymin, ymax, 0.1)
# subFig2.set_ylim([ymin, ymax])
# plt.yticks(y_ticks, fontsize=6)
# subFig2.plot(xdata, testErr_rate_list, color="tab:red")
# plt.show()
# a=np.array([0,2,3])
# b=np.array([1,2,3])
# c=np.array([a,b])
#
# print(sum(c))
# transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224)])
# data = np.load("./data/test.npy")
# data0 = np.reshape(data[2], (3, 32, 32))
# data0 = np.transpose(data0, (1, 2, 0))
# data1 = transform(data0)
# plt.figure(1)
# plt.imshow(data0)
# plt.show()
# plt.figure(2)
# plt.imshow(data1)
# plt.show()
a = torch.randn(3, 32, 32)
print(a.size)
b = torch.randn(3, 16, 16)
c = torch.add(a, b)
print(c.size())
#
# data1=data0
# transform=transforms.Compose(
#             [transforms.ToTensor(),
#              transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
#                                                (4, 4, 4, 4), mode='reflect').squeeze()),
#              transforms.ToPILImage(),
#              transforms.RandomCrop(32),
#              transforms.RandomHorizontalFlip(),
#              dataArgument.autoPolicy(),
#              ])
# data1=transform(data1)
# plt.figure(2)
# plt.imshow(data1)
# plt.show()
