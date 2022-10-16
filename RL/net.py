# -*- coding: utf-8 -*-
# @File    : net.py
# @Date    : 2021-12-06
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0.0)


class MyMulticastNet(nn.Module):
    def __init__(self, states_channel, action_num):
        super(MyMulticastNet, self).__init__()
        self.conv1 = nn.Conv2d(states_channel, 32, kernel_size=(1, 1))
        self.relu1 = nn.ReLU()

        self.fc1 = nn.Linear(32 * 14 * 14, 512)
        self.fc1_relu = nn.ReLU()

        # self.fc2 = nn.Linear(512, 256)
        # self.fc2_relu = nn.ReLU()

        self.adv1 = nn.Linear(512, 256)
        self.adv_relu = nn.ReLU()
        self.adv2 = nn.Linear(256, action_num)

        self.apply(weight_init)

    def forward(self, x):
        x = self.relu1(self.conv1(x))

        x = x.view(x.shape[0], -1)
        x = self.fc1_relu(self.fc1(x))
        # x = self.fc2_relu(self.fc2(x))

        adv = self.adv_relu(self.adv1(x))
        q_value = self.adv2(adv)

        return q_value


class MyMulticastNet2(nn.Module):
    def __init__(self, states_channel, action_num):
        super(MyMulticastNet2, self).__init__()
        self.conv1 = nn.Conv2d(states_channel, 32, kernel_size=(1, 1))
        self.relu1 = nn.ReLU()

        self.fc1 = nn.Linear(6272, 512)
        self.fc1_relu = nn.ReLU()

        self.fc2 = nn.Linear(512, 256)
        self.fc2_relu = nn.ReLU()

        self.adv = nn.Linear(256, action_num)
        self.val = nn.Linear(256, 1)

        self.apply(weight_init)

    def forward(self, x):
        x = self.relu1(self.conv1(x))

        x = x.view(x.shape[0], -1)
        x = self.fc1_relu(self.fc1(x))
        x = self.fc2_relu(self.fc2(x))

        sate_value = self.val(x)
        advantage_function = self.adv(x)

        return sate_value + (advantage_function - advantage_function.mean())


class MyMulticastNet3(nn.Module):
    def __init__(self, states_channel, action_num):
        super(MyMulticastNet3, self).__init__()
        self.conv1_1 = nn.Conv2d(states_channel, 32, kernel_size=(5, 1))
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=(5, 1))

        self.conv2_1 = nn.Conv2d(states_channel, 32, kernel_size=(1, 5))
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=(1, 5))

        self.fc1 = nn.Linear(5376, 512)
        self.fc2 = nn.Linear(512, 256)

        self.adv = nn.Linear(256, action_num)
        self.val = nn.Linear(256, 1)

        self.apply(weight_init)

    def forward(self, x):
        x1_1 = F.leaky_relu(self.conv1_1(x))
        x1_2 = F.leaky_relu(self.conv1_2(x1_1))

        x2_1 = F.leaky_relu(self.conv2_1(x))
        x2_2 = F.leaky_relu(self.conv2_2(x2_1))

        x1_3 = x1_2.view(x.shape[0], -1)
        x2_3 = x2_2.view(x.shape[0], -1)
        x = torch.cat([x1_3, x2_3], dim=1)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        sate_value = self.val(x)
        advantage_function = self.adv(x)

        return sate_value + (advantage_function - advantage_function.mean())


class MyMulticastNet4(nn.Module):
    def __init__(self, states_channel, action_num):
        super(MyMulticastNet4, self).__init__()
        self.conv1_1 = nn.Conv2d(states_channel, 32, kernel_size=(3, 1))
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=(3, 1))
        self.conv1_3 = nn.Conv2d(32, 32, kernel_size=(3, 1))

        self.conv2_1 = nn.Conv2d(states_channel, 32, kernel_size=(1, 3))
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=(1, 3))
        self.conv2_3 = nn.Conv2d(32, 32, kernel_size=(1, 3))

        self.lstm = nn.LSTMCell(3584, 512)
        self.fc1 = nn.Linear(512, 256)

        self.adv = nn.Linear(256, action_num)
        self.val = nn.Linear(256, 1)

        self.apply(weight_init)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self, inputs):
        x, (hx, cx) = inputs
        x1_1 = F.leaky_relu(self.conv1_1(x))
        x1_2 = F.leaky_relu(self.conv1_2(x1_1))
        x1_3 = F.leaky_relu(self.conv1_3(x1_2))

        x2_1 = F.leaky_relu(self.conv2_1(x))
        x2_2 = F.leaky_relu(self.conv2_2(x2_1))
        x2_3 = F.leaky_relu(self.conv2_3(x2_2))

        x1_3 = x1_3.view(x.shape[0], -1)
        x2_3 = x2_3.view(x.shape[0], -1)
        x = x1_3 + x2_3

        hx, cx = self.lstm(x, (hx, cx))

        x = hx
        x = F.leaky_relu(self.fc1(x))

        sate_value = self.val(x)
        advantage_function = self.adv(x)

        return sate_value + (advantage_function - advantage_function.mean()), (hx, cx)
