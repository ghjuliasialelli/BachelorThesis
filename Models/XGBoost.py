# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 09:24:21 2020

@author: Ghjulia Sialelli

XGBoost Model implementation

To do :
x Investigate the performance depending on the depth of the network
x Investigate on the optimal number of epochs for the training
x Node 818 is dropped because is NaN for some timesteps for some dps
x Investigate validation_set 
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
import xgboost as xgb
import pandas as pd
import numpy as np
import os.path

#############################################################################################
NNODES = 818                                # number of nodes for one dp
NPOINTS = 100                               # number of points in a minute 
LENFRAME = 12                               # length of a time frame to study 
NROWS = NNODES * (NPOINTS * LENFRAME + 1)   # number of rows to read for one dp
RATIO = 0.2                                 # ratio of testing to training set 

PATHS = ['../DataAnalysis/testset/naca_study2_training_data1.csv']

inputs = ['pressure', 'velocity-magnitude','acoustic-source-power-db']
outputs = ['inlet_velocity [m/s]','coef_drag [-]','coef_lift [-]','drag_force [N]',	
           'lift_force [N]','angle_attack [°]']

#############################################################################################

def transform_dataset_xgboost(PATHS):
    Xs = []
    Y1,Y2,Y3,Y4,Y5,Y6 = [],[],[],[],[],[]
    Ys = [Y1,Y2,Y3,Y4,Y5,Y6]
    
    for PATH in PATHS :
        for df in pd.read_csv(PATH, chunksize = NROWS, nrows = 15*NROWS): # remove nrows when whole dataset
            # Process dataframe
            df = df.dropna()
            # not all dps are NaN for nodenumber = 818
            df = df[df['nodenumber'] != 818]
            df[['time [s]']] = df[['time [s]']].astype('float64')
            df = df.drop(df[2.0 > df['time [s]']].index) 
            
            Xs.append(np.asarray(df[inputs].values))
            
            # Append to Ys the labels corresponding to the feature
            for idx, output in enumerate(outputs):    
                Ys[idx].append(np.asarray(df[output].values))
                    
    return np.concatenate(Xs,axis=0),np.concatenate(Y1,axis=0),np.concatenate(Y2,axis=0),np.concatenate(Y3,axis=0),np.concatenate(Y4,axis=0),np.concatenate(Y5,axis=0),np.concatenate(Y6,axis=0)


def train_model(model,X_train,X_test,Y_train,Y_test):
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    return mse 


#############################################################################################
print('Transforming / Loading dataset........')


if os.path.isfile('X_xgb.npy') and os.path.isfile('Y1.npy') and os.path.isfile('Y2.npy') and os.path.isfile('Y3.npy') and os.path.isfile('Y4.npy') and os.path.isfile('Y5.npy') and os.path.isfile('Y6.npy'): 
    X  = np.load('X_xgb.npy',allow_pickle='TRUE')
    Y1 = np.load('Y1.npy',allow_pickle='TRUE')
    Y2 = np.load('Y2.npy',allow_pickle='TRUE')
    Y3 = np.load('Y3.npy',allow_pickle='TRUE')
    Y4 = np.load('Y4.npy',allow_pickle='TRUE')
    Y5 = np.load('Y5.npy',allow_pickle='TRUE')
    Y6 = np.load('Y6.npy',allow_pickle='TRUE')
else:    
    X,Y1,Y2,Y3,Y4,Y5,Y6 = transform_dataset_xgboost(PATHS)
    np.save('X_xgb.npy', X)
    np.save('Y1.npy', Y1)
    np.save('Y2.npy', Y2)
    np.save('Y3.npy', Y3)
    np.save('Y4.npy', Y4)
    np.save('Y5.npy', Y5)
    np.save('Y6.npy', Y6)
     
print('Done!')

X_train,  X_test,   Y1_train, Y1_test = train_test_split(X, Y1, test_size=RATIO, shuffle=False)
Y2_train, Y2_test,  Y3_train, Y3_test = train_test_split(Y2,Y3, test_size=RATIO, shuffle=False)
Y4_train, Y4_test,  Y5_train, Y5_test = train_test_split(Y4,Y5, test_size=RATIO, shuffle=False)
_,              _,  Y6_train, Y6_test = train_test_split(X, Y6, test_size=RATIO, shuffle=False)


# Setting the models parameters
EPOCHS = 100

# Creating the models (one for each output feature)
models = []
for i in range(len(outputs)):
    models.append(xgb.XGBRegressor(objective='reg:squarederror'))

# Train and evaluate the models for a fixed number of epochs 
MSE = [[],[],[],[],[],[]]
for epoch in range(EPOCHS):
    print('Epoch {}/{}'.format(epoch, EPOCHS))
    
    MSE[0].append(train_model(models[0],X_train,X_test,Y1_train, Y1_test))
    MSE[1].append(train_model(models[1],X_train,X_test,Y2_train, Y2_test))
    print('33% done........')
    MSE[2].append(train_model(models[2],X_train,X_test,Y3_train, Y3_test))
    MSE[3].append(train_model(models[3],X_train,X_test,Y4_train, Y4_test))
    print('66% done........')
    MSE[4].append(train_model(models[4],X_train,X_test,Y5_train, Y5_test))
    MSE[5].append(train_model(models[5],X_train,X_test,Y6_train, Y6_test))
    
fig, plt = plt.subplots(len(models))
for i,MSE_model in enumerate(MSE) :
    plt[i].plot(range(1,EPOCHS+1),MSE_model)

            