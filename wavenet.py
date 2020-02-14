# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:18:27 2020

@author: Administrateur
"""

import torch.nn as nn
import numpy as np


class Wavenet(nn.Module):
    def __init__(self, conv_size, nb_of_measurements, nb_of_outputs):
        super(Wavenet, self).__init__()
        self.in_channels = nb_of_measurements
        self.out_channels = nb_of_outputs
        self.nb_timesteps = conv_size
        self.nb_layers = int(np.log2(self.nb_timesteps))
        self.conv = nn.ModuleList()
        self.layer_change = int(np.floor((self.in_channels - self.out_channels) / int(np.log2(self.nb_timesteps))))
        in_c = self.in_channels
        out_c = self.in_channels - self.layer_change
        for i in range(self.nb_layers):
            self.conv.append(nn.Conv1d(in_channels=in_c,
                                       out_channels=out_c,
                                       kernel_size=2,
                                       stride=2))

            if (((out_c >= self.out_channels - self.layer_change)
                 & (self.in_channels >= self.out_channels))
                    | ((out_c <= self.out_channels + self.layer_change)
                       & (self.in_channels < self.out_channels))):
                if i == self.nb_layers - 2:
                    in_c = out_c
                    out_c = self.out_channels
                else:
                    in_c = in_c - self.layer_change
                    out_c = out_c - self.layer_change
            else:
                in_c = self.out_channels
                out_c = self.out_channels
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(self.nb_layers):
            x = self.conv[i](x)
        x = x.squeeze()
        return x

    def reset_parameters(self):
        for i in range(self.nb_layers):
            self.conv[i].reset_parameters()


# function to create a list containing mini-batches 
def create_mini_batches(X, y, batch_size): 
    mini_batches = [] 
    data = np.hstack((X, y)) 
    n_minibatches = data.shape[0] // batch_size 
    i = 0
  
    for i in range(n_minibatches + 1): 
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
        X_mini = mini_batch[:, :-1] 
        Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches.append((X_mini, Y_mini)) 
    if data.shape[0] % batch_size != 0: 
        mini_batch = data[i * batch_size:data.shape[0]] 
        X_mini = mini_batch[:, :-1] 
        Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches.append((X_mini, Y_mini)) 
    return mini_batches 

