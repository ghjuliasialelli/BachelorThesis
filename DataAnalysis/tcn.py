# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:08:15 2020

@author: Administrateur
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np



class ResidualTCN(nn.Module):
    def __init__(self, time_len, out_dim, in_chan, out_chan,
                 kernel_pool=2, stride_pool=2, kernel_conv=5, stride_conv=1):
        super(ResidualTCN, self).__init__()
        # dimension init:
        self.time_len = time_len  # T
        self.out_dim = out_dim  # H
        self.in_chan = in_chan  # C
        self.out_chan = out_chan  # C'

        self.kernel_pool = kernel_pool
        self.stride_pool = stride_pool

        self.kernel_conv = kernel_conv
        self.stride_conv = stride_conv

        self.out_pool_dim = int((self.time_len - self.kernel_pool) / self.kernel_pool + 1)  # Tp
        self.out_conv_dim = int((self.out_pool_dim - self.kernel_conv) / self.stride_conv + 1)  # T'

        self.in_fc_dim = self.out_conv_dim * self.out_chan  # T'C'=C'T'

        # layers
        self.max_pool = nn.MaxPool2d(kernel_size=[self.kernel_pool, 1], stride=[self.stride_pool, 1])
        self.conv = nn.Conv2d(in_channels=self.in_chan, out_channels=self.out_chan, kernel_size=[self.kernel_conv, 1],
                              stride=[self.stride_conv, 1])
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=self.in_fc_dim, out_features=self.out_dim)

        # init weight
        self.reset_weight()

    def forward(self, x):  # [B,C,T,D]
        batch_size = x.shape[0]
        x_dim = x.shape[3]
        out = self.max_pool(x)  # [B, C, Tp, D]
        out = self.conv(out)  # [B, C', T', D]
        out = self.relu(out)
        out = out.permute(0, 3, 1, 2)  # [B, D, C', T']Oui faisons
        out = out.view(batch_size, x_dim, self.in_fc_dim)  # [B, D, C'T']
        out = self.fc(out)  # [B, C, H]
        return out

    def reset_weight(self):
        self.conv.weight.data.normal_(std=0.1)
        self.fc.weight.data.normal_(std=0.1)


class TemporalConvNet(nn.Module):
    def __init__(self, in_dim, time_len, hid_dim, out_dim, in_chan=1, model_hyperparams=None):
        super(TemporalConvNet, self).__init__()
        default_hyperparams = {
            'hid_chan1': 6,  # C1
            'hid_chan2': 8,  # C2
            'hid_chan3': 10,  # C3

            'kernel_conv': 5,
            'stride_conv': 1,

            'kernel_pool': 2,
            'stride_pool': 2
        }

        # Store the hyperparameters
        self._update_params(default_hyperparams, model_hyperparams)

        # dimension layers
        self.in_dim = in_dim  # D
        self.time_len = time_len  # T
        self.hid_dim = hid_dim  # H
        self.out_dim = out_dim  # Y

        self.in_chan = in_chan  # C

        self.in_conv1 = self.time_len - self.kernel_conv + 1  # T1
        self.in_conv2 = self.in_conv1 - self.kernel_conv + 1  # T2
        self.in_conv3 = self.in_conv2 - self.kernel_conv + 1  # T2

        self.in_fc_dim = self.in_conv3 * self.hid_chan3  # T3*C3

        # Layers
        self.conv1 = nn.Conv1d(in_channels=self.in_chan, out_channels=self.hid_chan1,
                               kernel_size=[self.kernel_conv, 1], stride=[self.stride_conv, 1])

        self.conv2 = nn.Conv1d(in_channels=self.hid_chan1, out_channels=self.hid_chan2,
                               kernel_size=[self.kernel_conv, 1], stride=[self.stride_conv, 1])

        self.conv3 = nn.Conv1d(in_channels=self.hid_chan2, out_channels=self.hid_chan3,
                               kernel_size=[self.kernel_conv, 1], stride=[self.stride_conv, 1])

        self.resnet1 = ResidualTCN(time_len=self.in_conv1, out_dim=self.hid_dim, in_chan=self.hid_chan1,
                                   out_chan=self.hid_chan2,
                                   kernel_pool=self.kernel_pool, stride_pool=self.stride_pool,
                                   kernel_conv=self.kernel_conv, stride_conv=self.stride_conv)

        self.resnet2 = ResidualTCN(time_len=self.in_conv2, out_dim=self.hid_dim, in_chan=self.hid_chan2,
                                   out_chan=self.hid_chan3,
                                   kernel_pool=self.kernel_pool, stride_pool=self.stride_pool,
                                   kernel_conv=self.kernel_conv, stride_conv=self.stride_conv)

        self.relu = nn.ReLU()

        self.fc = nn.Linear(in_features=self.in_fc_dim, out_features=self.hid_dim)

        self.fcb = nn.Linear(in_features=3 * self.hid_dim, out_features=self.time_len)

        self.fcd = nn.Linear(in_features=self.in_dim, out_features=self.out_dim)

        # init weight
        self.reset_weights()

    def forward(self, x):  # [B, C, T, D]
        batch_size = x.shape[0]
        x_dim = x.shape[3]
        out1_1 = self.conv1(x)  # [B, C1, T1, D]
        out1_1 = self.relu(out1_1)

        out1_2 = self.resnet1(out1_1)  # [B, D, H]

        out2_1 = self.conv2(out1_1)  # [B, C2, T2, D]

        out2_2 = self.resnet2(out2_1)  # [B, D, H]

        out3_1 = self.conv3(out2_1)  # [B, C3, T3, D]
        out3_1 = self.relu(out3_1)
        out3_1 = out3_1.permute(0, 3, 1, 2)  # [B, D, C3, T3]
        out3_1 = out3_1.view(batch_size, x_dim, self.in_fc_dim)  # [B, D, C3*T3]
        out3_2 = self.fc(out3_1)  # [B, D, H]

        out_cat = torch.cat((out1_2, out2_2, out3_2), dim=2)  # [B, D, 3H]
        out4 = self.fcb(out_cat)  # [B, D, T]
        out4 = out4.permute(0, 2, 1)  # [B, T, D]
        out = self.fcd(out4)  # [B, T, Y]
        return out

    def reset_weights(self):
        self.resnet1.reset_weight()
        self.resnet2.reset_weight()
        self.conv1.weight.data.normal_(std=0.1)
        self.conv2.weight.data.normal_(std=0.1)
        self.conv3.weight.data.normal_(std=0.1)
        self.fc.weight.data.normal_(std=0.1)
        self.fcb.weight.data.normal_(std=0.1)
        self.fcd.weight.data.normal_(std=0.1)

    def _set_params(self, params):
        for k in params.keys():
            self.__setattr__(k, params[k])

    def _update_params(self, prev_params, new_params):
        if new_params:
            params = update_param_dict(prev_params, new_params)
        else:
            params = prev_params
        self._set_params(params)



def train(model, train_input, train_target, loss_fn, optimizer, batch_size=1):
    model.train()
    train_loss = 0

    for b in range(0, train_input.size(0), batch_size):
        model.zero_grad()
        #print(train_input.narrow(0, b, batch_size).shape)
        pred = model(train_input.narrow(0, b, batch_size))
        loss = loss_fn(pred, train_target.narrow(0, b, batch_size))  # +ridge ?
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    return train_loss / train_input.shape[0]


def test(model, test_input, test_target, loss_fn, batch_size=1):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for b in range(0, test_input.size(0), batch_size):
            pred = model(test_input.narrow(0, b, batch_size))
            loss = loss_fn(pred, test_target.narrow(0, b, batch_size))
            test_loss += loss.item()

    return test_loss / test_input.shape[0]

def update_param_dict(prev_params, new_params):
    params = prev_params.copy()
    for k in prev_params.keys():
        if k in new_params.keys():
            params[k] = new_params[k]
    return params

