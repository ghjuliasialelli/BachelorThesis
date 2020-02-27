# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:02:26 2020

@author: Ghjulia Sialelli

TCN architecture implementation
"""

from keras.layers import Conv1D, Input, ReLU, Concatenate, Dense, MaxPooling1D
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
import matplotlib.pyplot as plt
from keras.models import Model
from keras import optimizers
from math import inf
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
'''
REL_NODES = [] 
'''
#############################################################################################
'''
Post-processing the data to feed it to Wavenet.
When new dataset comes available, need to modify this function.
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

'''
def transform_dataset_LSTM(PATHS):
    Xs = []
    Ys = []
    for PATH in PATHS :
        for df in pd.read_csv(PATH, chunksize = NROWS, nrows = 15*NROWS): # remove nrows when whole dataset
            # Process dataframe
            df = df.dropna()
            df = df[df['nodenumber'] != 818]
            df[['time [s]']] = df[['time [s]']].astype('float64')
            df = df.drop(df[2.0 > df['time [s]']].index) 
            
            # For each timestep, do the following :
            # for the input : select the ~3 nodes which were deemed relevant for
            #       the analysis, and get the pressure data from them
            # for the output : get all pressure and velocity data from all nodes
            for time in df['time [s]'].unique() :
                dftime = df[df['time [s]'] == time]
                
                Y = dftime[['pressure','inlet_velocity [m/s]']].to_numpy()
                Y = Y.reshape((1,Y.shape[0], Y.shape[1]))
                Ys.append(Y)
                
                Xdf = dftime[(dftime['nodenumber'] == REL_NODES[0]) | (dftime['nodenumber'] == REL_NODES[1]) | (dftime['nodenumber'] == REL_NODES[2])]
                X = Xdf[['pressure']].to_numpy()
                X = X.reshape((1,X.shape[0], X.shape[1]))
                Xs.append(X)
               
    return np.asarray(Xs),np.asarray(Ys)
'''

#############################################################################################
KERNEL_SIZE = 5
DILATION_DEPTH = 6
N_FILTERS = 10
EPOCHS = 100
BS = 800
#############################################################################################

class TCN():
    '''
    Parameters :
        input_shape : tuple of input shape
        output_shape : tuple of output shape
        kernel_size : size of convolutional operations in residual blocks
    '''
    def __init__(self, input_shape, output_shape, kernel_size = KERNEL_SIZE, 
                 n_filters = N_FILTERS):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.regularizer = l2(0.0001)
        self.model = self.model()
        self.history = None
        self.loss = inf
    
    
    '''
    Construction of the model.
    '''
    def model(self):
        x = Input(shape = self.input_shape)
        
        conv11 = Conv1D(self.n_filters, self.kernel_size, kernel_regularizer = self.regularizer, padding='same')(x)
        relu11 = ReLU()(conv11)
        
        pooling11 = MaxPooling1D(pool_size = 2, strides = 2, padding='same')(relu11)
        conv12 = Conv1D(self.n_filters, self.kernel_size, kernel_regularizer = self.regularizer, padding='same')(pooling11)
        relu12 = ReLU()(conv12)
        fc12 = Dense(self.n_filters, kernel_regularizer = self.regularizer)(relu12)
        
        conv21 = Conv1D(self.n_filters, self.kernel_size, kernel_regularizer = self.regularizer, padding='same')(relu11) 
        relu21 = ReLU()(conv21)
        
        pooling21 = MaxPooling1D(pool_size = 2, strides = 2, padding='same')(relu21)
        conv22 = Conv1D(self.n_filters, self.kernel_size, kernel_regularizer = self.regularizer, padding='same')(pooling21)
        relu22 = ReLU()(conv22)
        fc22 = Dense(self.n_filters, kernel_regularizer = self.regularizer)(relu22)
        
        conv31 = Conv1D(self.n_filters, self.kernel_size, kernel_regularizer = self.regularizer, padding='same')(relu21)
        relu31 = ReLU()(conv31)
        fc31 = Dense(self.n_filters, kernel_regularizer = self.regularizer)(relu31)
    
        concat = Concatenate()([fc12, fc22, fc31])
        relu4 = ReLU()(concat)
        fc4 = Dense(self.output_shape[1], kernel_regularizer = self.regularizer)(relu4)
        
        model = Model(inputs=x, outputs=fc4)
        model.summary()
        
        return model
        
    
    '''
    Fitting of the model on the data.
    '''
    def fit_model(self, X, Y, optimizer, validation_data = None, epochs = EPOCHS, batch_size = BS):
        self.model.compile(optimizer, loss = 'mean_squared_error', metrics = None)
        self.history = self.model.fit(X,Y,shuffle=False,batch_size=BS, epochs=EPOCHS, validation_data = validation_data)
        return self.history
    
    '''
    Evaluating the model after training on the testing set.
    Returns : scalar test loss
    '''
    def eval_model(self, X, Y, batch_size = BS):
        self.loss = self.model.evaluate(X,Y,batch_size)
        return self.loss
    
#############################################################################################
print('Transforming / Loading dataset........')
if os.path.isfile('X_tcn.npy') and os.path.isfile('Y_tcn.npy') : 
    X = np.load('X_tcn.npy',allow_pickle='TRUE')
    Y = np.load('Y_tcn.npy',allow_pickle='TRUE')
else:    
    X, Y = transform_dataset_Wavenet(PATHS)
    np.save('X_tcn.npy', X)
    np.save('Y_tcn.npy', Y)
    
print('Done!')

Xtr,Xte,Ytr,Yte = train_test_split(X,Y, test_size=RATIO, shuffle=False)

print('Xtr shape : ', Xtr.shape)
print('Xte shape : ', Xte.shape)
print('Ytr shape : ', Ytr.shape)
print('Yte shape : ', Yte.shape)

tcn = TCN(Xtr.shape[1:3],Ytr.shape[1:3])

adam = optimizers.Adam(lr=0.00005, beta_1 = 0.99, beta_2 = 0.999, amsgrad = False)

history = tcn.fit_model(Xte,Yte,adam)
    
# Summarize history for loss
plt.plot(history.history['loss']) 
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

# Get loss on testing set
print('Evaluating the model')
loss = tcn.eval_model(Xte,Yte)
print('Testing loss : ', loss)
    
    