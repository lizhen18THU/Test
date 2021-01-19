"""
用于预训练的文件
"""

import torch.utils.data as Data
import math
import matplotlib.pyplot as plt
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable
import os
import time
import dataAugment
from trainMethod import *

import WideResnet
import shake_shake_networks as shakeshake


# 我的数据集
class myDataset(Data.Dataset):

    def __init__(self, picFile, labelFile, task, transform=None):
        # 预训练进而判断网络准确性
        # 20/100分类：将50000张图片分为40000张训练，10000张测试
        if task == "pre_train":  # 40000张训练数据
            data = np.load(picFile)
            self.data = data[0:400]
            df = np.array(pd.read_csv(labelFile))
            self.labels = df[0:400, 1]
            for i in range(1, 100):
                self.data = np.vstack((self.data, data[500 * i:500 * i + 400]))
                self.labels = np.hstack((self.labels, df[500 * i:500 * i + 400, 1]))
        elif task == "pre_test":  # 10000张训练数据
            data = np.load(picFile)
            self.data = data[400:500]
            df = np.array(pd.read_csv(labelFile))
            self.labels = df[400:500, 1]
            for i in range(1, 100):
                self.data = np.vstack((self.data, data[500 * i + 400:500 * (i + 1)]))
                self.labels = np.hstack((self.labels, df[500 * i + 400:500 * (i + 1), 1]))

        # 最终训练与测试
        # 20/100分类训练：50000张训练数据
        elif task == "final_train":
            self.data = np.load(picFile)
            df = np.array(pd.read_csv(labelFile))
            self.labels = df[:, 1]
        # 20/100分类测试：10000张测试数据，实际标签未知
        elif task == "final_test":
            self.data = np.load(picFile)
        else:
            raise TypeError("task not defined!")
        self.transform = transform
        self.task = task

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # get a picture
        sample = self.data[index, :]
        sample = np.reshape(sample, (3, 32, 32))
        sample = np.transpose(sample, (1, 2, 0))
        sample = self.transform(sample)
        if self.task == "final_test":  # 最终的预测没有标签，只返回图片数据
            return sample
        else:
            # get a label
            label = self.labels[index]
            return sample, label


# 测试集验证
def test(testLoader, net, fc, device, pre_train_test=False, testErr_rate_list=None):
    # 在测试集上进行测试或预测,如果是预训练则计算测试集准确率
    if not pre_train_test:  # 预训练
        # 转为评估模式
        net.eval()
        fc.eval()
        with torch.no_grad():
            predict = torch.Tensor([]).to(device)
            for i, (samples) in enumerate(testLoader, 0):
                # 计算输出
                samples_val = Variable(samples.to(device))
                out = fc(net(samples_val))
                _, batch_predict = torch.max(out, 1)
                if predict.shape[0]:
                    predict = torch.cat((predict, batch_predict), 0)
                else:
                    predict = batch_predict
            # 返回预测结果
            predict = predict.cpu().numpy()
        return predict
    else:
        total_test_samples = 0
        err_test_samples = 0
        # 转为评估模式
        net.eval()
        fc.eval()
        with torch.no_grad():
            for i, (samples, labels) in enumerate(testLoader, 0):
                # 计算输出
                samples_val, labels_val = Variable(samples.to(device)), Variable(labels.to(device))
                out = fc(net(samples_val))
                # 计算错误样本数
                total_test_samples += labels_val.size(0)
                _, predict = torch.max(out, 1)
                err_test_samples += (predict != labels_val).sum().item()
        # 将测试集的错误率放入列表并输出
        cur_rate = err_test_samples / total_test_samples
        testErr_rate_list.append(cur_rate)
        print("Test Error: %.7f %%" % (100 * cur_rate))


# 每10个epoch展示一次错误率随训练过程的图像
def errPlotShow(epoch, trainErr_rate_list, testErr_rate_list):
    if (epoch) % 10 == 0:
        if len(testErr_rate_list):
            fig = plt.figure(1, (4, 3))
            # x轴数据
            xdata = [i for i in range(1, epoch + 1)]
            # 画出训练集的错误率图像
            subFig1 = fig.add_subplot(2, 1, 1)
            subFig1.set_title("trainerror/loss-epoch figure")
            # 设定坐标轴范围
            subFig1.set_xlim([1, epoch + 1])
            ymax = max(trainErr_rate_list)
            ymin = min(trainErr_rate_list)
            ymin = min([ymin, 0])
            y_ticks = np.arange(ymin, ymax, 0.1)
            subFig1.set_ylim([ymin, ymax])
            plt.yticks(y_ticks, fontsize=6)
            subFig1.plot(xdata, trainErr_rate_list, color="tab:blue")

            # 画出测试集的错误率图像
            subFig2 = fig.add_subplot(2, 1, 2)
            subFig2.set_title("testerror-epoch figure")
            # 设定坐标轴范围
            subFig2.set_xlim([1, epoch + 1])
            ymax = max(testErr_rate_list)
            ymin = min(testErr_rate_list)
            ymin = min([ymin, 0])
            y_ticks = np.arange(ymin, ymax, 0.1)
            subFig2.set_ylim([ymin, ymax])
            plt.yticks(y_ticks, fontsize=6)
            subFig2.plot(xdata, testErr_rate_list, color="tab:red")
            plt.show()
        else:
            plt.figure(1, (4, 3))
            # x轴数据
            xdata = [i for i in range(1, epoch + 1)]
            # 画出训练集的错误率图像
            plt.title("trainerror/loss-epoch figure")
            # 设定坐标轴范围
            plt.xlim([1, epoch + 1])
            ymax = max(trainErr_rate_list)
            ymin = min(trainErr_rate_list)
            ymin = min([ymin, 0])
            y_ticks = np.arange(ymin, ymax, 0.1)
            plt.ylim([ymin, ymax])
            plt.yticks(y_ticks, fontsize=6)
            plt.plot(xdata, trainErr_rate_list, color="tab:red")
            plt.show()


# 将两个列表写入CSV，可以是Error列表，也可以是后面为了将预测的分类结果写入csv
def writeto_csv(array1, array2, name, path):
    # 预训练
    if len(array2):
        array1 = array1[:, np.newaxis]
        array2 = array2[:, np.newaxis]
        combinedList = np.hstack((array1, array2))
        content = pd.DataFrame(columns=name, data=combinedList)
        content.to_csv(path_or_buf=path, index=False)
    # 最终训练
    else:
        array1 = array1[:, np.newaxis]
        content = pd.DataFrame(columns=name, data=array1)
        content.to_csv(path_or_buf=path, index=False)


# 显式定义全连接层,也可以直接在网络内部定义全连接层,这里显式的定义全连接层是为了后面直接调用损失函数ISDAloss
class Fc_layer(nn.Module):
    def __init__(self, feature_num, class_num):
        super(Fc_layer, self).__init__()
        self.class_num = class_num
        self.fc = nn.Linear(feature_num, class_num)

    def forward(self, x):
        x = self.fc(x)
        return x


def main(argv=None):
    # Train on GPU 并且使用 CUDNN 加速
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    # 任务种类与数据文件路径
    train_task = "pre_train"
    test_task = "pre_test"
    pre_train_test = True
    num_classes = 20  # 20分类/100分类
    print(num_classes, "classification " + train_task + " and " + test_task)

    # 网络和数据中间结果的读写路径
    statePath_read = "./halfState/state_shakeshakepre20_stat0.pth"
    statePath_write = "./halfState/state_shakeshakepre20_stat0.pth"
    listPath_read = "./halfState/list_shakeshakepre20_stat0.csv"
    listPath_write = "./halfState/list_shakeshakepre20_stat0.csv"

    # 使用的训练方法，ISDA和cutmix不可同时使用，最终选择ISDA
    # ISDA
    usingISDA = True
    if usingISDA:
        print("ISDA")
    # cutmix
    is_cutmix = False
    beta = 1
    cutmix_prob = 0.5
    if is_cutmix:
        print("cutmix")

    # ISDA和cutmix不同时使用
    if usingISDA and is_cutmix:
        raise NotImplementedError("you can't use ISDA and cutmix at the same time!")

    # 根据不同网络选择参数，便于参数的集中调试，经过训练和测试，选择效果最好的WideResnet
    netType = "Shakeshake"
    if netType == "ResNet":
        # 训练参数,ResNet
        epochs = 100
        initial_lr = 0.1
        batchSize = 64
        weight_decay = 1e-4
        momentum = 0.9
        cos_lr = False
        lrChanPoint = [50, 75]
        lrChanRate = 0.1
        nesterov = True
        print("ResNet")
    elif netType == "ResNest":
        # 训练参数 ,ResNest
        epochs = 550
        initial_lr = 0.1
        batchSize = 128
        weight_decay = 1e-4
        momentum = 0.9
        cos_lr = False
        lrChanPoint = []
        lrChanRate = 0.1
        nesterov = True
        print("ResNest")
    elif netType == "WideResnet":
        # 训练参数，WideResnet
        epochs = 460
        initial_lr = 0.1
        batchSize = 128
        weight_decay = 5e-4
        momentum = 0.9
        cos_lr = True
        lrChanPoint = []
        lrChanRate = 0.1
        nesterov = True
        print("WideResnet")
    elif netType == "Shakeshake":
        # 训练参数，shakeshake
        epochs = 460
        initial_lr = 0.2
        batchSize = 128
        weight_decay = 1e-4
        momentum = 0.9
        cos_lr = True
        lrChanPoint = []
        lrChanRate = 0.1
        nesterov = True
        print("Shakeshake")
    else:
        raise TypeError("do not has this netType:" + netType)

    # 数据处理以及数据增强方式,最终选择cutout+autoAugment
    augment = "cutout_autoAugment"
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    if augment == "noAugment":
        transform_train = transforms.Compose([transforms.ToTensor(), normalize])
        print("No Augment!")
    elif augment == "standardAugment":
        transform_train = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                               (4, 4, 4, 4),
                                               mode='reflect').squeeze()),
             transforms.ToPILImage(),
             transforms.RandomCrop(32),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             normalize])
        print("Standard Augment!")
    elif augment == "cutout":
        transform_train = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                               (4, 4, 4, 4), mode='reflect').squeeze()),
             transforms.ToPILImage(),
             transforms.RandomCrop(32),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             dataAugment.Cutout(1, 16),
             normalize])
        print("cutout!")
    elif augment == "autoAugment":
        transform_train = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                               (4, 4, 4, 4), mode='reflect').squeeze()),
             transforms.ToPILImage(),
             transforms.RandomCrop(32),
             transforms.RandomHorizontalFlip(),
             dataAugment.autoPolicy(),
             transforms.ToTensor(),
             normalize])
        print("AutoAugment!")
    elif augment == "cutout_autoAugment":
        transform_train = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                               (4, 4, 4, 4), mode='reflect').squeeze()),
             transforms.ToPILImage(),
             transforms.RandomCrop(32),
             transforms.RandomHorizontalFlip(),
             dataAugment.autoPolicy(),
             transforms.ToTensor(),
             dataAugment.Cutout(1, 16),
             normalize])
        print("cutout and AutoAugment!")
    else:
        raise TypeError("do not has this Augment:" + augment)

    # 测试数据不数据增强，只进行归一化
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    # 训练数据集和迭代器
    trainDataset = myDataset(picFile="./data/train.npy", labelFile="./data/train1.csv", task=train_task,
                             transform=transform_train)
    trainLoader = Data.DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    # 测试数据集和迭代器
    testDataset = myDataset(picFile="./data/train.npy", labelFile="./data/train1.csv", task=test_task,
                            transform=transform_test)
    testLoader = Data.DataLoader(testDataset, batch_size=batchSize, shuffle=False)

    # 定义网络
    net = shakeshake.shake_resnet26_2x112d(num_classes).to(device)
    # net = WideResnet.WideResnet28_10(28, 10, num_classes).to(device)
    fc = Fc_layer(net.feature_num, num_classes).to(device)

    # 定义损失函数和优化器
    if usingISDA:
        Criterion = ISDALoss(net.feature_num, num_classes).to(device)
    else:
        Criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD([{'params': net.parameters()},
                           {'params': fc.parameters()}], lr=initial_lr, momentum=momentum,
                          weight_decay=weight_decay, nesterov=nesterov)

    # 如果是继续未训练完的网络，将之前的参数读取进来
    if os.path.exists(statePath_read):
        state = torch.load(statePath_read)
        net.load_state_dict(state['net'])
        fc.load_state_dict(state['fc'])
        optimizer.load_state_dict(state['optimizer'])
        preEpochs = state['epoch']
        print("networkState:pre-trained network")
    else:
        preEpochs = 0
        print("networkState:Initial network")

    # 如果前面有未训练完的模型，读取之前保存的错误率列表
    if os.path.exists(listPath_read):
        list_df = pd.read_csv(listPath_read).values
        loss_list = list(list_df[:, 0])
        trainErr_rate_list = loss_list
        testErr_rate_list = list(list_df[:, 1]) if pre_train_test else []
    else:
        loss_list = []
        trainErr_rate_list = []
        testErr_rate_list = []

    start_time = time.time()
    for epoch in range(preEpochs + 1, epochs + 1):

        last_time = time.time()
        # 调整学习率
        if epoch <= 10:
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr * epoch / 10
        elif cos_lr:  # 是否使用cos_lr调整学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.5 * initial_lr * (1 + math.cos(math.pi * (epoch - 11) / (epochs - 10)))
        else:
            if epoch in lrChanPoint:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lrChanRate

        # 进行一个epoch的训练，并计算准确率
        total_train_samples = 0
        err_train_samples = 0
        # 转换为训练模式
        net.train()
        fc.train()
        for i, curData in enumerate(trainLoader, 0):
            samples, labels = curData[0].to(device), curData[1].to(device)
            samples_val = Variable(samples)
            labels_val = Variable(labels)

            # 使用cutmix训练方法
            if is_cutmix and beta > 0:
                temper_loss = CutMix(beta, cutmix_prob, samples_val, labels_val, net, fc, Criterion,
                                     optimizer)
            # 使用ISDA训练方法
            elif usingISDA:
                ratio = 0.5 * (epoch / epochs)
                loss, out = Criterion(net, fc, samples_val, labels_val, ratio)
                with torch.no_grad():
                    total_train_samples += labels.size(0)
                    _, predict = torch.max(out, 1)
                    err_train_samples += (predict != labels).sum().item()
                # 反向传播并优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # 使用正常的训练方法
            else:
                out = fc(net(samples_val))
                # 计算分类错误样本数
                with torch.no_grad():
                    total_train_samples += labels.size(0)
                    _, predict = torch.max(out, 1)
                    err_train_samples += (predict != labels).sum().item()
                # 计算损失并反向传播
                loss = Criterion(out, labels_val)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if is_cutmix and beta > 0:
            loss_list.append(temper_loss)
            print("----------------------------------------------------------")
            print("epoch: ", epoch)
            print("loss: ", temper_loss)
        else:
            cur_rate = err_train_samples / total_train_samples
            trainErr_rate_list.append(cur_rate)
            print("----------------------------------------------------------")
            print("epoch: ", epoch)
            print("loss:", loss.item())
            print("Train Error: %.7f %%" % (100 * cur_rate))

        # 在测试集上测试
        if pre_train_test:
            test(testLoader=testLoader, net=net, fc=fc, device=device, pre_train_test=pre_train_test,
                 testErr_rate_list=testErr_rate_list)
        # 记录当前epoch用时
        cur_time = time.time()
        print("time using for this epoch: %.7f s" % (cur_time - last_time))
        if epoch % 10 == 0:
            # 保存训练的网络状态中间结果
            if torch.__version__ < "1.6":
                state = {"net": net.state_dict(), "fc": fc.state_dict(), "optimizer": optimizer.state_dict(),
                         "epoch": epoch}
                torch.save(state, statePath_write)
            else:
                state = {"net": net.state_dict(), "fc": fc.state_dict(), "optimizer": optimizer.state_dict(),
                         "epoch": epoch}
                torch.save(state, statePath_write, _use_new_zipfile_serialization=False)
            # 记录下error列表
            if is_cutmix and beta > 0:
                errPlotShow(epoch, loss_list, testErr_rate_list)
                name = ["loss", "testErr"] if len(testErr_rate_list) else ["loss"]
                writeto_csv(array1=np.array(loss_list), array2=np.array(testErr_rate_list), name=name,
                            path=listPath_write)
            else:
                errPlotShow(epoch, trainErr_rate_list, testErr_rate_list)
                name = ["trainErr", "testErr"] if len(testErr_rate_list) else ["trainErr"]
                writeto_csv(array1=np.array(trainErr_rate_list), array2=np.array(testErr_rate_list), name=name,
                            path=listPath_write)

    cur_time = time.time()
    print("------------------------------------------------------")
    print("------------------------------------------------------")
    if pre_train_test:
        print("Finish training")
    else:
        # 网络训练完毕，对最终的测试机进行预测并保存csv文件
        predict = test(testLoader, net, fc, device, pre_train_test)
        image_id = np.arange(0, 10000, 1)
        if num_classes == 20:
            name = ["image_id", "coarse_label"]
            writeto_csv(array1=image_id, array2=predict, name=name,
                        path="./results/1.csv")
        else:
            name = ["image_id", "fine_label"]
            writeto_csv(array1=image_id, array2=predict, name=name,
                        path="./results/2.csv")
        print("Finish training and predicting")
    print("time using for all epochs: %.7f s" % (cur_time - start_time))


if __name__ == '__main__':
    main()
