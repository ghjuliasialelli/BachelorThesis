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
from math import gcd
import pandas as pd
import numpy as np
import os.path
import sys

import helper

# 0 or 1
index = int(sys.argv[1])

corres = ['drag_force [N]','lift_force [N]', 'angle_attack [°]', 'inlet_velocity [m/s]']

outputs = corres[index]

#############################################################################################
bij_type = {'inlet_velocity [m/s]' : 'velocity-magnitude',
             'coef_drag [-]' : 'velocity-magnitude',
             'coef_lift [-]' : 'pressure',
             'drag_force [N]' : 'velocity-magnitude',	
             'lift_force [N]' : 'pressure',
             'angle_attack [°]' : 'acoustic-source-power-db'}

bij_loc = {'inlet_velocity [m/s]' : [],
             'coef_drag [-]' : [],
             'coef_lift [-]' : [],
             'drag_force [N]' : [],	
             'lift_force [N]' : [],
             'angle_attack [°]' : []}

inputs = bij_type[outputs]
#rel_ids = bij_loc[outputs]

### 4 chosen sensors for optimized predictions ###
rel_ids = [7347.5, 7301.49452, 6527.5, 7327.49997]
########## press   press        vel     vel ######


NNODES = len(rel_ids)                       # number of sensors we consider
NPOINTS = 100                               # number of points in a second 
LENFRAME = 12                               # length of a time frame to study 
NROWS = NNODES * (NPOINTS * LENFRAME + 1)   # equivalent of one experiment (in rows)
RATIO = 0.25                                # Ratio of testing to training set

TOTROWS = 818 * (NPOINTS * LENFRAME + 1)

PATHS = ['naca_study3_training_data1.csv',
         'naca_study3_training_data2.csv',
         'naca_study3_training_data3.csv',
         'naca_study3_training_data4.csv',
         'naca_study3_training_data5.csv',
         'naca_study3_training_data6.csv',
         'naca_study2_training_data1.csv',
         'naca_study2_training_data2.csv',
         'naca_study2_training_data3.csv',
         'naca_study2_training_data4.csv',
         'naca_study2_training_data5.csv',
         'naca_study2_training_data6.csv',
         'naca_study4_training_data1.csv',
         'naca_study4_training_data2.csv',
         'naca_study4_training_data3.csv',
         'naca_study4_training_data4.csv',
         'naca_study4_training_data5.csv',
         'naca_study4_training_data6.csv']

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
        for df in pd.read_csv(PATH, chunksize = TOTROWS): 
            # Process dataframe
            df = df.dropna() 
            df[['time [s]']] = df[['time [s]']].astype('float64')
            df = df.drop(df[2.0 > df['time [s]']].index) 
            
            ######## modify this depending on the # of relevant nodes #############
            df = df[(df['cell-id'] == rel_ids[0]) | (df['cell-id'] == rel_ids[1]) | 
                    (df['cell-id'] == rel_ids[2]) | (df['cell-id'] == rel_ids[3]) ]
            #######################################################################
            
            for time in df['time [s]'].unique() :
                dftime = df[df['time [s]'] == time]
                df1 = dftime[(dftime['cell-id'] == rel_ids[0]) | (dftime['cell-id'] == rel_ids[1])][['pressure']].T.values
                df2 = dftime[(dftime['cell-id'] == rel_ids[2]) | (dftime['cell-id'] == rel_ids[3])][['velocity-magnitude']].T.values
                
                X = np.concatenate((df1,df2), axis = 1)[0]
                Xs.append(X
                          )
                Y = dftime[[outputs]].T.values[0][:1]
                Ys.append(Y)
                    
    # Standardize the data 
    Xs = helper.standardize(np.array(Xs))
    Ys = helper.standardize(np.array(Ys))
    
    return Xs.reshape((Xs.shape[0], 1, 4)), Ys.reshape((Ys.shape[0], 1, 1))

#############################################################################################
print('Transforming / Loading dataset........')
if os.path.isfile('X_wn_{}.npy'.format(outputs[:-6])) and os.path.isfile('Y_std_wn_{}.npy'.format(outputs[:-6])) : 
    X = np.load('X_wn_{}.npy'.format(outputs[:-6]),allow_pickle='TRUE')
    Y = np.load('Y_std_wn_{}.npy'.format(outputs[:-6]),allow_pickle='TRUE')
else:    
    X, Y = transform_dataset_LSTM(PATHS)
    X = np.stack(X)
    np.save('X_lstm.npy', X)
    np.save('Y_lstm.npy', Y)
print('Done!')

Y = Y.reshape((Y.shape[0], 1))
np.save('Y_lstm_{}.npy'.format(outputs[:-6]), Y)

X_train, X_test,Y_train, Y_test = train_test_split(X,Y, test_size = RATIO, shuffle = False)

##############################################################
# Defining the parameters of the network
data_dim = 4
BS = gcd(X_train.shape[0], X_test.shape[0])
EPOCHS = 25
##############################################################

# Create the stacked LSTM neural network 
model = Sequential()
model.add(LSTM(data_dim, return_sequences = True, stateful = True,
    batch_input_shape = (BS, 1, 4)))
model.add(LSTM(data_dim, return_sequences = True, stateful = True))
model.add(LSTM(data_dim, return_sequences = True, stateful = True))
model.add(LSTM(data_dim))
model.add(Dense(Y.shape[1]))
model.summary()

##############################################################
# Configure the model for training
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, Y_train, batch_size = BS, epochs = EPOCHS, validation_split = 0.2, shuffle=False)

MSE_train = history.history['loss']
MSE_validation = history.history['val_loss']

# Saving the losses to files
np.savetxt('MSE_train_lstm.out', np.asarray(MSE_train), delimiter=',')
np.savetxt('MSE_test_lstm.out', np.asarray(MSE_validation), delimiter=',')

print('Writing the model to file')
model.save('model_lstm.h5')


