# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:55:40 2020

@author: Ghjulia Sialelli

Helper functions.
"""

import numpy as np

np.seterr(divide='raise')

def standardize(x):
    std_data = np.zeros(x.shape)
    if len(x.shape) > 2:
        mean = np.mean(x.reshape((x.shape[0]*x.shape[1], x.shape[2])), axis=0)
        std = np.std(x.reshape((x.shape[0]*x.shape[1], x.shape[2])), axis=0)
        for i in range(x.shape[0]):
            centered_data = x[i] - mean
            std_data[i] = centered_data / std
    else:
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        centered_data = x - mean
        std_data = centered_data / std
    return std_data

