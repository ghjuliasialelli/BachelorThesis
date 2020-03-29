#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:35:56 2020

@author: ghjuliasialelli

TO DO :
    - select some Dps from study 3 (validation set bc unseen by model)
    - estimate min max and mean bias error for a waveform
    - for all the models and study cases
    - visualize 
"""
from keras.models import load_model
import pandas as pd
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import xgboost as xgb

NPOINTS = 100                               # number of points in a second 
LENFRAME = 12                               # length of a time frame to study 
TOTROWS = 818 * (NPOINTS * LENFRAME + 1)

PATH = '../data/naca_20dp_VALIDATION.csv'

corres = ['drag_force [N]','lift_force [N]', 'angle_attack [°]', 'inlet_velocity [m/s]']
models = ['XGB',  'Wavenet', 'TCN', 'LSTM']
sensors = ['4P_SHAP', '4P_RANDOM', '2P2V', '2P1V1N', '100P']
mapping = {'4P_SHAP' : ([7347.5, 7301.49452, 7295.5, 7349.4958],['pressure','pressure','pressure','pressure']),
           '2P2V' : ([7347.5, 7301.49452, 6527.5, 7327.49997],['pressure','pressure','velocity-magnitude','velocity-magnitude']),
           '4P_RANDOM' : ([7347.5, 7301.49452, 7295.5, 7349.4958],['pressure','pressure','pressure','pressure'])
           }

########################
# index = int(sys.argv[1])
index = 0
########################

outputs = corres[index]
model = 'Wavenet'
sensor = '4P_SHAP'
rel_ids = mapping[sensor][0]
rel_types = mapping[sensor][1]

# we randomly select 3 dps from the study 3 ##################################
listofdf = []
listofdps = []

'''
for _ in range(3):
    df = pd.read_csv(PATH, nrows = TOTROWS, skiprows = [i for i in range(1,1+random.randrange(0,19) * TOTROWS)])
    listofdps.append(np.unique(np.array(df[['dp_number']]))[0])
    listofdf.append(df)
'''

df_13 = pd.read_csv(PATH, nrows = TOTROWS, skiprows = [i for i in range(1,1 + 13* TOTROWS)])
df_18 = pd.read_csv(PATH, nrows = TOTROWS, skiprows = [i for i in range(1,1 + 18* TOTROWS)])

listofdf = [df_13,df_18]
listofdps = [13,18]
##############################################################################

def standardize(X, std = None, mean = None):
    Xresh = X.reshape((X.shape[0]*X.shape[1],))
    if mean == None : mean = np.mean(Xresh)
    if std == None : std = np.std(Xresh)
    return (X-mean)/std, std, mean

def destandardize(Xstd, std, mean):
    return Xstd*std + mean


def transform_dataset(df):
    Xs = []
    Ys = []
    
    df = df.dropna() 
    df[['time [s]']] = df[['time [s]']].astype('float64')
    df = df.drop(df[2.0 > df['time [s]']].index) 
    
    ######## modify this depending on the # of relevant nodes #############
    df = df[(df['cell-id'] == rel_ids[0]) | (df['cell-id'] == rel_ids[1]) | 
            (df['cell-id'] == rel_ids[2]) | (df['cell-id'] == rel_ids[3]) ]
    #######################################################################
    
    for time in df['time [s]'].unique() :
        dftime = df[df['time [s]'] == time]
        
        df1 = dftime[dftime['cell-id'] == rel_ids[0]][[rel_types[0]]].T.values
        df2 = dftime[dftime['cell-id'] == rel_ids[1]][[rel_types[1]]].T.values
        df3 = dftime[dftime['cell-id'] == rel_ids[2]][[rel_types[2]]].T.values
        df4 = dftime[dftime['cell-id'] == rel_ids[3]][[rel_types[3]]].T.values
        
        X = np.concatenate((df1,df2,df3,df4), axis = 1)[0]
        Xs.append(X)
        Y = dftime[[outputs]].T.values[0][:1]
        Ys.append(Y)
                    
    return np.array(Xs), np.array(Ys)


# now, for each df in listofdf, we predict using one of the models ###########

if model == 'Wavenet' : 
    temp = '_wn'
    model_name = 'Wavenet'
    
    #model1 = load_model('../WAVENET/23.03/{}/model{}_{}.h5'.format('100P', temp, outputs[:-6]))
    model2 = load_model('../WAVENET/23.03/{}/model{}_{}.h5'.format('4P_SHAP', temp, outputs[:-6]))
    #model3 = load_model('../WAVENET/23.03/{}/model{}_{}.h5'.format('2P1V1N', temp, outputs[:-6]))
    
    #info = np.loadtxt('../WAVENET/23.03/{}/dataset_info_{}.out'.format('100P', outputs[:-6]), delimiter = ',')
    #Xtr_std, Xtr_mean, Ytr_std, Ytr_mean = info[0], info[1], info[2], info[3]
    
    info2 = np.loadtxt('../WAVENET/23.03/{}/dataset_info_{}.out'.format('4P_SHAP', outputs[:-6]), delimiter = ',')
    Xtr_std2, Xtr_mean2, Ytr_std2, Ytr_mean2 = info2[0], info2[1], info2[2], info2[3]
    
    #info3 = np.loadtxt('../WAVENET/23.03/{}/dataset_info_{}.out'.format('2P1V1N', outputs[:-6]), delimiter = ',')
    #Xtr_std3, Xtr_mean3, Ytr_std3, Ytr_mean3 = info3[0], info3[1], info3[2], info3[3]

if model == 'XGB' : 
    temp = 'xgb'
    model_name = 'XGBRegressor'
    
    model = xgb.Booster()
    model.load_model('../XGB/{}/model{}_{}.json'.format(sensor, outputs[:-6], temp))
    
# to load again, with X_predict = xgb.DMatrix(X_predict)
# bst = xgb.Booster()
# bst.load_model("model{}_xgb.json".format(outputs[:-6]))
# bst.predict(X_predict)

Ys = []
Ypreds1 = []
Ypreds2 = []
Ypreds3 = []

for df,dp_number in zip(listofdf, listofdps) : 
    print(df)
    
    X, Y = transform_dataset(df)
    
    #Xstd, _, _ = standardize(X, Xtr_std, Xtr_mean)
    #Ystd, _, _ = standardize(Y, Ytr_std, Ytr_mean)
    
    Xstd2, _, _ = standardize(X, Xtr_std2, Xtr_mean2)
    Ystd2, _, _ = standardize(Y, Ytr_std2, Ytr_mean2)
    
    #Xstd3, _, _ = standardize(X, Xtr_std3, Xtr_mean3)
    #Ystd3, _, _ = standardize(Y, Ytr_std3, Ytr_mean3)
    
    if model == 'Wavenet' :
        #Xstd = Xstd.reshape((Xstd.shape[0], 1, 4))
        #Ystd = Ystd.reshape((Ystd.shape[0], 1, 1))
        Xstd2 = Xstd2.reshape((Xstd2.shape[0], 1, 4))
        Ystd2 = Ystd2.reshape((Ystd2.shape[0], 1, 1))
        #Xstd3 = Xstd3.reshape((Xstd3.shape[0], 1, 4))
        #Ystd3 = Ystd3.reshape((Ystd3.shape[0], 1, 1))
    
    #Ypred1 = model1.predict(Xstd) 
    #Ypred_nonstd = destandardize(Ypred1, Ytr_std, Ytr_mean)
    #Ypred_nonstd = Ypred_nonstd.reshape((Ypred_nonstd.shape[0], 1))
    #Ypred1 = Ypred1.reshape((Ypred1.shape[0], 1))
    #Ypreds1.append(Ypred1)
 
    Ypred2 = model2.predict(Xstd2)    
    Ypred_nonstd2 = destandardize(Ypred2, Ytr_std2, Ytr_mean2)
    Ypred_nonstd2 = Ypred_nonstd2.reshape((Ypred_nonstd2.shape[0], 1))
    Ypred2 = Ypred2.reshape((Ypred2.shape[0], 1))
    Ypreds2.append(Ypred2)
 
    #Ypred3 = model3.predict(Xstd3)    
    #Ypred_nonstd3 = destandardize(Ypred3, Ytr_std3, Ytr_mean3)
    #Ypred_nonstd3 = Ypred_nonstd3.reshape((Ypred_nonstd3.shape[0], 1))
    #Ypred3 = Ypred3.reshape((Ypred3.shape[0], 1))
    #Ypreds3.append(Ypred3)

    #Ystd = Ystd.reshape((Ystd.shape[0],1))
    Ystd2 = Ystd2.reshape((Ystd2.shape[0],1))
    #Ystd3 = Ystd3.reshape((Ystd3.shape[0],1))
    
    
    xrange = np.arange(2,12.01, 0.01)
    plt.figure(figsize = (15,10))
    #plt.plot(xrange, Ystd, label = 'true value')
    #plt.plot(xrange, Ypred1, label = 'predicted value', linestyle = 'dotted')
    plt.plot(xrange, Y, label = 'true value')
    #plt.plot(xrange, Ypred_nonstd, label = '100P predicted value', linestyle = 'dotted')
    plt.plot(xrange, Ypred_nonstd2, label = '4P_SHAP predicted value', linestyle = 'dotted')
    #plt.plot(xrange, Ypred_nonstd3, label = '2P1V1N predicted value', linestyle = 'dotted')
    #plt.plot(xrange, Ypred2, label = 'predicted value (old)')
    plt.legend()
    plt.title('{} prediction'.format(model_name))
    plt.xlabel('time [s]')
    plt.ylabel(outputs)
    plt.tight_layout(pad=3.0)
    #plt.savefig('{}_{}_vs_4P_SHAP_{}_{}.png'.format(model_name, sensor, outputs[:-6], dp_number))
    plt.show()  
       