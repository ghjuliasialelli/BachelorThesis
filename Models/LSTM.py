# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:26:42 2020

@author: Ghjulia Sialelli

LSTM Model Implementation

To do :
x Investigate the performance depending on the depth of the network
x Investigate on the optimal number of epochs for the training
x Node 818 is dropped because is NaN for some timesteps for some dps
x Investigate validation_set 
"""

from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense
from keras.models import Sequential
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import os.path

#############################################################################################
NNODES = 818                                # number of nodes for one dp
NPOINTS = 100                               # number of points in a minute 
LENFRAME = 12                               # length of a time frame to study 
NROWS = NNODES * (NPOINTS * LENFRAME + 1)   # number of rows to read for one dp
RATIO = 0.2                                 # Ratio of testing to training set 

inputs = ['pressure', 'velocity-magnitude','acoustic-source-power-db']
outputs = ['inlet_velocity [m/s]','coef_drag [-]','coef_lift [-]','drag_force [N]',	
           'lift_force [N]','angle_attack [°]']


PATHS = ['../DataAnalysis/testset/naca_study2_training_data1.csv']

#############################################################################################
'''
Post-processing the data to feed it to LSTM
Input expected : (batch_size, NNODES, len(inputs))
Output format: (batch_size, 1, len(outputs))
'''
def transform_dataset_LSTM(PATHS):
    Xs = []
    Ys = []
    for PATH in PATHS :
        for df in pd.read_csv(PATH, chunksize = NROWS, nrows = 2*NROWS): # remove nrows when whole dataset
            # Process dataframe
            df = df.dropna()
            df = df[df['nodenumber'] != 818]
            df[['time [s]']] = df[['time [s]']].astype('float64')
            df = df.drop(df[2.0 > df['time [s]']].index) 
            
            for time in df['time [s]'].unique() :
                dftime = df[df['time [s]'] == time]
                # Append to Xs a feature of the NN : shape (NNODES,len(inputs))
                Xs.append(np.asarray(dftime[inputs].values))
                # Append to Ys the labels corresponding to the feature, shape (1,len(outputs))
                Ys.append(np.asarray(dftime[outputs].iloc[0]))
    return np.asarray(Xs),np.asarray(Ys)

'''
Helper function to compute the appropriate batch size 
'''
def computeHCF(x, y):
    if x > y:
        smaller = y
    else:
        smaller = x
    for i in range(1, smaller+1):
        if((x % i == 0) and (y % i == 0)):
            hcf = i

    return hcf

#############################################################################################
print('Transforming / Loading dataset........')
if os.path.isfile('X_lstm.npy') and os.path.isfile('Y_lstm.npy') : 
    X = np.load('X_lstm.npy',allow_pickle='TRUE').item()
    Y = np.load('Y_lstm.npy',allow_pickle='TRUE').item()
else:    
    X, Y = transform_dataset_LSTM(PATHS)
    X = np.stack(X)
    np.save('X_lstm.npy', X)
    np.save('Y_lstm.npy', Y)
print('Done!')


X_train, X_test,Y_train, Y_test = train_test_split(X,Y, test_size=RATIO, shuffle=False)

# Defining the parameters of the network
batch_size = computeHCF(X_train.shape[0], X_test.shape[0])
print('BATCH SIZE : ',batch_size)
data_dim = len(inputs)
BS = batch_size
EPOCHS = 500 

# Create the stacked LSTM neural network 
model = Sequential()
model.add(LSTM(data_dim, return_sequences = True, stateful = True,
    batch_input_shape = (BS, NNODES-1, data_dim)))
model.add(LSTM(data_dim, return_sequences = True, stateful = True))
model.add(LSTM(data_dim, return_sequences = True, stateful = True))
model.add(LSTM(data_dim))
model.add(Dense(Y.shape[1]))
model.summary()

# Configure the model for training
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model for a fixed number of epochs
losses = []
for epoch in range(EPOCHS):
    print('Epoch {}/{}'.format(epoch, EPOCHS))
    model.fit(X_train, Y_train, batch_size=BS, shuffle=False)
    losses.append(model.evaluate(X_test, Y_test, batch_size=BS, verbose=1))

# Plot the loss as a function of the number of epochs
plt.plot(losses, range(1,EPOCHS+1))
