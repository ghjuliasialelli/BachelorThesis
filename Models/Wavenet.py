# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 09:13:55 2020

@author: Ghjulia Sialelli

Wavenet architecture implementation

To change when running on the cluster :
x remove 'nrows' attribute in pd.read_csv
x import PATHS from file ../DataAnalysis/testset/paths.py

To do :
x Investigate the performance depending on the depth of the network
x Investigate on the optimal number of epochs for the training
x Node 818 is dropped because is NaN for some timesteps for some dps
x Investigate validation_set 
x Regularization term ?
"""

from keras.layers import Conv1D, Multiply, Add, Input, Activation, TimeDistributed
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras import optimizers
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
Post-processing the data to feed it to Wavenet
'''
def transform_dataset_Wavenet(PATHS):
    Xs = []
    Ys = []
    
    for PATH in PATHS :
        for df in pd.read_csv(PATH, chunksize = NROWS, nrows = 2*NROWS): # remove nrows when whole dataset
            # Process dataframe
            df = df.dropna() 
            df = df[df['nodenumber'] != 818]
            df[['time [s]']] = df[['time [s]']].astype('float64')
            df = df.drop(df[2.0 > df['time [s]']].index) 
            
            X = df[inputs].to_numpy()
            X = X.reshape((X.shape[0],1,3))
            Xs.append(X)
            
            Y = df[outputs].to_numpy()
            Y = Y.reshape((Y.shape[0],1,6))
            Ys.append(Y)
            
    return np.concatenate(Xs,axis=0),np.concatenate(Ys,axis=0)


#############################################################################################
KERNEL_SIZE = 2
DILATION_DEPTH = 6
N_FILTERS = 32
ACTIVATION_FUN = 'linear'
EPOCHS = 100
BS = 100
#############################################################################################

class WaveNetRegressor():
    '''
    Parameters :
        input_shape : tuple of input shape
        output_shape : tuple of output shape
        kernel_size : size of convolutional operations in residual blocks
        dilation_depth : depth of residual blocks
        activation : activation function used 
    '''
    def __init__(self, input_shape, output_shape, kernel_size = KERNEL_SIZE, 
                 dilation_depth = DILATION_DEPTH, n_filters = N_FILTERS,
                 activation = ACTIVATION_FUN):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kernel_size = kernel_size
        self.dilation_depth = dilation_depth
        self.n_filters = n_filters
        self.activation = activation
        self.model = self.model()
        self.history = None
    
    '''
    Implementation of the Residual and Skip connections.
    '''
    def skip_connections(self, x, i):
        tanh = Conv1D(self.n_filters, self.kernel_size, dilation_rate = self.kernel_size ** i,
                      padding = 'causal', activation = 'tanh')(x)
        sigmoid = Conv1D(self.n_filters, self.kernel_size, dilation_rate = self.kernel_size ** i,
                         padding = 'causal', activation = 'sigmoid')(x)
        z = Multiply()([tanh, sigmoid])
        skip = Conv1D(self.n_filters, kernel_size = 1)(z)
        out = Add()([skip,x])
        return out, skip

    '''
    Construction of the model.
    '''
    def model(self):
        x = Input(shape = self.input_shape)
        out = Conv1D(self.n_filters, kernel_size = 2, padding = 'causal')(x)
   
        skip_connections = []
        for i in range(self.dilation_depth):
            out, skip = self.skip_connections(out,i)
            skip_connections.append(skip)
        out = Add()(skip_connections)
   
        out = Activation('relu')(out)
        out = Conv1D(self.n_filters, kernel_size = 2, padding = 'same', activation = 'relu')(out)

        out = Conv1D(self.n_filters, self.output_shape[0], activation='relu')(out)
        out = Conv1D(self.output_shape[1], self.output_shape[0])(out)

        out = TimeDistributed(Activation(self.activation))(out)
        
        model = Model(input=x, output=out)
        model.summary()

        return model
    
    '''
    Fitting of the model on the data.
    '''
    def fit_model(self, X, Y, optimizer, validation_data = None, epochs = EPOCHS, batch_size = BS):
        self.model.compile(optimizer, loss = 'mean_squared_error', metrics = None)
        self.history = self.model.fit(X,Y,shuffle=False,batch_size=BS, epochs=EPOCHS, validation_data = validation_data)
        return self.history


#############################################################################################
print('Transforming / Loading dataset........')
if os.path.isfile('X_wn.npy') and os.path.isfile('Y_wn.npy') : 
    X = np.load('X_wn.npy',allow_pickle='TRUE')
    Y = np.load('Y_wn.npy',allow_pickle='TRUE')
else:    
    X, Y = transform_dataset_Wavenet(PATHS)
    np.save('X_wn.npy', X)
    np.save('Y_wn.npy', Y)
    
print('Done!')

print('X shape : ', X.shape)
print('Y shape : ', Y.shape)

Xtr,Xte,Ytr,Yte = train_test_split(X,Y, test_size=RATIO, shuffle=False)

print('Xtr shape : ', Xtr.shape)
print('Xte shape : ', Xte.shape)
print('Ytr shape : ', Ytr.shape)
print('Yte shape : ', Yte.shape)

wn = WaveNetRegressor((1,3),(1,6))
adam = optimizers.Adam(lr=0.00075, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
wn.fit_model(Xte,Yte,adam)