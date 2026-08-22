import copy
import os
import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import AlexNet
import torch.nn as nn
import time
import datetime
import pandas as pd


def train_val_data_process():
    # 加载 FashionMNIST 数据集，用train_data去接收这个对象
    train_data = FashionMNIST(root="./data",
                              train=True,  # 训练集,6w张图片
                              transform=transforms.Compose([transforms.Resize(size=227), transforms.ToTensor()]),
                              download=True)

    # 将训练集拆分为训练集与验证集（8:2比例）
    train_data, val_data = Data.random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])

    # 创建训练集 DataLoader，迭代器对象
    train_data_loader = Data.DataLoader(dataset=train_data,
                                        batch_size=64,  # 所以一次epoch训练大概需要48000/128=375批
                                        shuffle=True,  # 打乱顺序
                                        num_workers=0)  # 指定加载数据时使用的CPU子进程数量
    # pytorch中加载数据时，主进程在GPU上训练，子进程在CPU上提前获取下一批数据，也叫异步加载
    # 创建验证集 DataLoader
    val_data_loader = Data.DataLoader(dataset=val_data,
                                      batch_size=64,
                                      shuffle=False,
                                      num_workers=0)

    return train_data_loader, val_data_loader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    # 设定训练所用到的设备，有GPU用GPU没有GPU用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用Adam优化器，学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 损失函数为交叉熵函数
    criterion = nn.CrossEntropyLoss()
    # 将模型放入到训练设备中
    model = model.to(device)
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高标准度
    best_acc = 0.0
    # 训练损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []
    # 当前时间
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        # 初始化参数
        # 训练集损失
        train_loss = 0.0
        # 训练集中预测正确的样本数的计数
        train_corrects = 0.0
        # 验证集损失
        val_loss = 0.0
        # 验证集中预测正确的样本数的计数
        val_corrects = 0.0
        # 训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0

        # 对每一个mini—batch进行训练和计算
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将图像和标签放入到训练设备中，根据设备类型决定是cpu还是gpu
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 设置模型为训练模式，父类nn.Module中实现，实际上是执行model.training = True
            model.train()

            # 前向传播过程，输出为一个batch中对应的预测，实际传入对象b_x为四维张量，包含batch值，channel，height，width四个维度信息
            output = model(b_x)  # 输出为tensor[128,10]，128*10的二维张量矩阵，

            # 查找每一行中最大值(得分)对应的行标，输入对象为张量和哪个维度dimension，返回一维张量，取行标dim=0，取列标dim=1
            pre_lab = torch.argmax(output, dim=1)

            # 计算每一个batch的损失函数，输入对象为张量和实际标签
            loss = criterion(output, b_y)

            # 将梯度初始化为0，如果每次反向传播前不清零，梯度会累加，
            # 假设第一次反向传播梯度计算为0.2，第二次如果计算出来是0.3，实际上梯度是0.5，梯度累加导致参数更新错误
            optimizer.zero_grad()

            # 反向传播计算，张量的方法
            loss.backward()

            # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数的值，优化器对象
            optimizer.step()

            # 对损失函数进行累加 .item()是提取张量里的标量值，b_x.size(0)是当前batch中的样本数量，
            # 此处也就是batch值128，因为前面说了b_x实际上是一个四维张量
            # 相乘是得到该batch中的总损失，因为loss值的计算一般是该batch中的平均值
            train_loss += loss.item() * b_x.size(0)

            # 如果预测正确，则准确度train_corrects加1
            train_corrects += torch.sum(pre_lab == b_y)
            # 当前用于训练的样本数量
            train_num += b_x.size(0)

        # 计算并保存每一次迭代的loss值和准确率
        train_loss_all.append(train_loss / train_num)
        # tensor(x)->tensor(x.0)->x.0
        train_acc_all.append(train_corrects.double().item() / train_num)

        for step, (b_x, b_y) in enumerate(val_dataloader):
            # 将特征放入到验证设备中
            b_x = b_x.to(device)
            # 将标签放入到验证设备中
            b_y = b_y.to(device)
            # 设置模型为评估模式
            model.eval()
            # 前向传播过程，输入一个batch，输出为一个batch中对应的预测
            output = model(b_x)

            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)

            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)

            # 对损失函数进行累加
            val_loss += loss.item() * b_x.size(0)

            # 如果预测正确，则准确度train_corrects加1
            val_corrects += torch.sum(pre_lab == b_y)

            # 当前用于训练的样本数量
            val_num += b_x.size(0)

        # 计算并保存验证集的loss值和准确率
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # 寻找最高准确度的权重
        if val_acc_all[-1] > best_acc:
            # 保存当前最高准确度
            best_acc = val_acc_all[-1]
            # 保存当前的最高准确度,返回一个字典，包含tensor数据格式
            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算该轮次耗时
        time_use = time.time() - since
        print("Time taken: {:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))

    # 选择最优参数
    # 加载最高准确率下的模型参数
    torch.save(best_model_wts, "best_model.pth")

    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all})

    return train_process


def matplot_acc_loss(train_process):
    # 创建保存目录（若不存在则自动创建）
    save_dir = "./result_figures"
    os.makedirs(save_dir, exist_ok=True)

    # 绘图
    plt.figure(figsize=(12, 4))

    # -------- loss 曲线 --------
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, "r-", label="train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, "b-", label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    # -------- acc 曲线 --------
    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, "r-", label="train acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, "b-", label="val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")

    # -------- 保存图像 --------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"train_result_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 训练曲线已保存至：{save_path}")

    # 显示图像
    plt.show()


if __name__ == "__main__":
    # 将模型实例化
    model = AlexNet()
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(model, train_dataloader, val_dataloader, 20)
    matplot_acc_loss(train_process)
