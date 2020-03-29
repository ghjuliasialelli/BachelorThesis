#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:38:09 2020

@author: ghjuliasialelli

Proper Analysis
"""

from keras.models import load_model
from keras import backend 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

NPOINTS = 100                               # number of points in a second 
LENFRAME = 12                               # length of a time frame to study 
TOTROWS = 818 * (NPOINTS * LENFRAME + 1)

#PATH = '../data/naca_20dp_VALIDATION.csv'
#PATH = '../data/naca_Nonharmonic_validation.csv'
PATH = '../data/naca_2Harmonic_2Nonharmonic_validation.csv'

corres = ['drag_force [N]','lift_force [N]', 'angle_attack [°]', 'inlet_velocity [m/s]']
#models = ['XGB',  'Wavenet', 'TCN', 'LSTM']
sensors = ['4P_SHAP', '2P1V1N']
mapping = { '2P1V1N' : ([7347.5, 7301.49452, 6527.5, 7315.49998], ['pressure','pressure','velocity-magnitude','acoustic-source-power-db']),
            '4P_SHAP' : ([7347.5, 7301.49452, 7295.5, 7349.4958],['pressure','pressure','pressure','pressure']),
            '4P_RANDOM' : ([7347.5, 7301.49452, 7295.5, 7349.4958],['pressure','pressure','pressure','pressure']),
            '100P' : ([7783.5    , 7229.50231, 8057.501690000001, 6961.497170000001, 6933.502579999999,
                   7467.5    , 7645.4986 , 6643.5    , 7291.5    , 7013.49951,
                   6759.5    , 6747.5    , 7559.5    , 7217.496279999999, 6509.48116,
                   6815.5    , 8101.50508, 6887.5    , 6581.50339, 8061.505090000001,
                   6875.5    , 6773.496940000001, 7391.5    , 6805.49917, 6913.4985400000005,
                   7099.5    , 7347.5    , 6711.5    , 7409.49568, 7359.5    ,
                   6883.5    , 7033.4993 , 7261.5018900000005, 8037.50267, 7067.5    ,
                   6583.5    , 7043.5    , 6747.5    , 7187.5    , 6903.5    ,
                   7645.4986 , 6963.5    , 7553.499379999999, 6553.50361, 7195.5    ,
                   7281.5021799999995, 6643.5    , 7531.5    , 7737.50252, 6633.49952,
                   7915.5    , 8041.50162, 6973.49915, 6845.5005200000005, 7553.499379999999,
                   7109.501740000001, 6949.5    , 6529.49715, 7529.5018 , 6869.499629999999,
                   7455.5    , 6671.5    , 6537.50187, 6963.5    , 6519.49998,
                   7627.5    , 8015.5    , 7717.49777, 6973.49915, 6903.5    ,
                   7103.5    , 6675.5    , 7317.501509999999, 7705.499640000001, 7327.49997,
                   7957.49608, 6989.5005599999995, 7057.4999099999995, 7629.5016 , 6703.5    ,
                   8111.5    , 7843.5    , 7483.5    , 8135.4992 , 7747.5    ,
                   7317.501509999999, 7099.5    , 6723.5    , 8021.501029999999, 7489.4994799999995,
                   7945.50285, 7007.5    , 7411.5    , 7151.5    , 7897.50247,
                   7623.5    , 7679.5    , 7315.49998, 7411.5    , 7035.5    ], ['pressure' for _ in range(100)])}


model = 'Wavenet'

print('Loading the 2 dps')
#df_3 = pd.read_csv(PATH, nrows = TOTROWS, skiprows = [i for i in range(1,1 + 3* TOTROWS)])
#df_8 = pd.read_csv(PATH, nrows = TOTROWS, skiprows = [i for i in range(1,1 + 8* TOTROWS)])

#df_13 = pd.read_csv(PATH, nrows = TOTROWS, skiprows = [i for i in range(1,1 + 13* TOTROWS)])
#df_18 = pd.read_csv(PATH, nrows = TOTROWS, skiprows = [i for i in range(1,1 + 18* TOTROWS)])

df_1 = pd.read_csv(PATH, nrows = TOTROWS, skiprows = [i for i in range(1,1 + 1* TOTROWS)])
df_2 = pd.read_csv(PATH, nrows = TOTROWS, skiprows = [i for i in range(1,1 + 2* TOTROWS)])

print('Done!')

dfs = [df_1,df_2]
dps = [1,2]

wrong = []

##############################################################################

def standardize(X, std = None, mean = None):
    Xresh = X.reshape((X.shape[0]*X.shape[1],))
    if mean == None : mean = np.mean(Xresh)
    if std == None : std = np.std(Xresh)
    return (X-mean)/std, std, mean

def destandardize(Xstd, std, mean):
    return Xstd*std + mean

def transform_dataset(df, rel_ids, rel_types, outputs):
    Xs = []
    Ys = []
    
    #df = df.dropna() 
    df[['time [s]']] = df[['time [s]']].astype('float64')
    df = df.drop(df[2.0 > df['time [s]']].index) 
    
    ######## modify this depending on the # of relevant nodes #############
    #df = df[(df['cell-id'] == rel_ids[0]) | (df['cell-id'] == rel_ids[1]) | 
    #        (df['cell-id'] == rel_ids[2]) | (df['cell-id'] == rel_ids[3]) ]
    #######################################################################
    
    for time in df['time [s]'].unique() :
        dftime = df[df['time [s]'] == time]
        
        df1 = dftime[dftime['cell-id'] == rel_ids[0]][[rel_types[0]]].T.values
        df2 = dftime[dftime['cell-id'] == rel_ids[1]][[rel_types[1]]].T.values
        df3 = dftime[dftime['cell-id'] == rel_ids[2]][[rel_types[2]]].T.values
        df4 = dftime[dftime['cell-id'] == rel_ids[3]][[rel_types[3]]].T.values
        
        X = np.concatenate((df1,df2,df3,df4), axis = 1)[0]
        
        if len(X) != 4 : 
            wrong.append(dftime)
            print('----------------------')
            print('time : ',time)
            print('cell-ids : ', rel_ids)
            print('X : ', X)
            print('dftime : ', dftime)
            print('df1,2,3,4: ', df1,df2,df3,df4)

        
        Xs.append(X)
        Y = dftime[[outputs]].T.values[0][:1]
        Ys.append(Y)
                    
    return np.array(Xs), np.array(Ys)


maxB = dict()
minB = dict()
stdB = dict()

    
# We make predictions for each dp
for df, dp in zip(dfs,dps):
    backend.clear_session()
    print('Processing dp_number {}'.format(dp))
    
    # and for each output, we make a plot for the 4 models
    for outputs in corres : 
        print('...Processing output {}'.format(outputs))
        
        models = []
        Xtr_stds = []
        Xtr_means = []
        Ytr_stds = []
        Ytr_means = []
        
        Ypreds = []     # predictions for each model
        Xstds = []      # standardized data for each model, input of predictions
        Xs = []         # Xs unstandardized input data
    
        # We load the four models (one for each sensor type selection)
        
        for sensor in sensors :
            print('......Processing sensor {}'.format(sensor))
            models.append(load_model('../WAVENET/23.03/{}/model_wn_{}.h5'.format(sensor, outputs[:-6])))
            info = np.loadtxt('../WAVENET/23.03/{}/dataset_info_{}.out'.format(sensor, outputs[:-6]), delimiter = ',')
            Xtr_stds.append(info[0])
            Xtr_means.append(info[1])
            Ytr_stds.append(info[2])
            Ytr_means.append(info[3])
            
            # and we prepare the input datasets
            print('.........Preparing the data')
            rel_ids = mapping[sensor][0]
            rel_types = mapping[sensor][1]
            X, Y = transform_dataset(df, rel_ids, rel_types, outputs)
            
            print('X shape : ', X.shape)
            print('Y shape : ', Y.shape)
            '''
            if len(X.shape) == 1 :
                
                flag = True
                
                indices = []
                for i,x in enumerate(X):
                    if len(x) != 4 : indices.append(i)
                X = np.delete(X, indices)
                Y = np.delete(Y, indices)
                X = np.stack(X)
            '''
            
            Xs.append(X)
            print('.........Done')

        print('......Standardizing')
        for i in range(len(sensors)):  
            Xstds.append(standardize(Xs[i], Xtr_stds[i], Xtr_means[i])[0])
        print('......Done')
    
        if model == 'Wavenet' :
            for i in range(len(Xstds)):
                Xstds[i] = Xstds[i].reshape((Xstds[i].shape[0],1,4))  
            
        print('......Predicting')
        for i in range(len(sensors)):
            Ypred_std = models[i].predict(Xstds[i])
            Ypred_nonstd = destandardize(Ypred_std, Ytr_stds[i], Ytr_means[i])
            Ypred_nonstd = Ypred_nonstd.reshape((Ypred_nonstd.shape[0], 1))
            Ypreds.append(Ypred_nonstd)
        print('......Done')
    
        xrange = np.arange(2,12.01, 0.01)
    
            
        plt.figure(figsize = (15,10))
        plt.plot(xrange, Y, label = 'true value')
        
        for i,sensor in enumerate(sensors) : 
            diff = abs(Y - Ypreds[i])
            
            minB['{}_{}_{}'.format(sensor, outputs[:-6], dp)] = min(diff)[0]
            maxB['{}_{}_{}'.format(sensor, outputs[:-6], dp)] = max(diff)[0]
            stdB['{}_{}_{}'.format(sensor, outputs[:-6], dp)] = np.std(diff)
            
            
            plt.plot(xrange, Ypreds[i], label = '{} predicted value'.format(sensor), linestyle = 'dotted')
            
        plt.legend()
        plt.title('WaveNet models prediction')
        plt.xlabel('time [s]')
        plt.ylabel(outputs)
        plt.tight_layout(pad=3.0)
        plt.savefig('WaveNet_generalization_{}_dp{}.png'.format(outputs[:-6], dp))
        plt.show()  
        
print('maxB : ', maxB) 
print('minB : ', minB) 
print('stdB : ', stdB)

#np.save('minB.npy', minB) 
#np.save('maxB.npy', maxB)
#np.save('stdB.npy', stdB)