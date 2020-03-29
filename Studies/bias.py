#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 22:06:50 2020

@author: ghjuliasialelli

bias
"""


from keras.models import load_model
import pandas as pd
import numpy as np

NPOINTS = 100                               # number of points in a second 
LENFRAME = 12                               # length of a time frame to study 
TOTROWS = 818 * (NPOINTS * LENFRAME + 1)

PATH = 'naca_20dp_VALIDATION.csv'

corres = ['drag_force [N]','lift_force [N]', 'angle_attack [°]', 'inlet_velocity [m/s]']
#models = ['XGB',  'Wavenet', 'TCN', 'LSTM']
sensors = ['4P_SHAP', '4P_RANDOM', '2P1V1N']
#sensors = ['4P_SHAP']
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
df_13 = pd.read_csv(PATH, nrows = TOTROWS, skiprows = [i for i in range(1,1 + 13* TOTROWS)])
df_18 = pd.read_csv(PATH, nrows = TOTROWS, skiprows = [i for i in range(1,1 + 18* TOTROWS)])
print('Done!')

dfs = [df_13,df_18]
dps = [13,18]

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


maxB = dict()
minB = dict()
stdB = dict()

    
# We make predictions for each dp
for df, dp in zip(dfs[-1:],dps[-1:]):
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
            models.append(load_model('{}/model_wn_{}.h5'.format(sensor, outputs[:-6])))
            info = np.loadtxt('{}/dataset_info_{}.out'.format(sensor, outputs[:-6]), delimiter = ',')
            Xtr_stds.append(info[0])
            Xtr_means.append(info[1])
            Ytr_stds.append(info[2])
            Ytr_means.append(info[3])
            
            # and we prepare the input datasets
            print('.........Preparing the data')
            rel_ids = mapping[sensor][0]
            rel_types = mapping[sensor][1]
            X, Y = transform_dataset(df, rel_ids, rel_types, outputs)
            
            if len(X.shape) == 1 :
                
                flag = True
                
                indices = []
                for i,x in enumerate(X):
                    if len(x) != 4 : indices.append(i)
                X = np.delete(X, indices)
                Y = np.delete(Y, indices)
                X = np.stack(X)
            
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
        
        # Y is the true value
        # Ypred is the predicted value for each model
        for i,sensor in enumerate(sensors) : 
            diff = abs(Y - Ypreds[i])
            
            minB['{}_{}_{}'.format(sensor, outputs[:-6], dp)] = min(diff)
            maxB['{}_{}_{}'.format(sensor, outputs[:-6], dp)] = max(diff)
            stdB['{}_{}_{}'.format(sensor, outputs[:-6], dp)] = np.std(diff)
        
print('maxB : ', maxB) 
print('minB : ', minB) 
print('stdB : ', stdB)

np.save('minB.npy', minB) 
np.save('maxB.npy', maxB)
np.save('stdB.npy', stdB)
#read_dictionary = np.load('my_file.npy',allow_pickle='TRUE').item()

maxB :  {'4P_SHAP_drag_for_13': 13.93192869, '4P_RANDOM_drag_for_13': 14.72835356, '2P1V1N_drag_for_13': 13.64769369, 
         '4P_SHAP_lift_for_13': 51.19787022, '4P_RANDOM_lift_for_13': 97.69698797, '2P1V1N_lift_for_13': 56.01069066, 
         '4P_SHAP_angle_atta_13': 3.86944572, '4P_RANDOM_angle_atta_13': 22.66610184, '2P1V1N_angle_atta_13': 8.52205936, 
         '4P_SHAP_inlet_velocity_13': 1.87569427, '4P_RANDOM_inlet_velocity_13': 14.31966305, '2P1V1N_inlet_velocity_13': 10.34070873, 
         
         '4P_SHAP_drag_for_18': 0.30429658, '4P_RANDOM_drag_for_18': 9.60165108, '2P1V1N_drag_for_18': 0.40189794, 
         '4P_SHAP_lift_for_18': 1.72947337, '4P_RANDOM_lift_for_18': 15.95911939, '2P1V1N_lift_for_18': 1.7523604, 
         '4P_SHAP_angle_atta_18': 0.35328197, '4P_RANDOM_angle_atta_18': 20.78710556, '2P1V1N_angle_atta_18': 0.38760793, 
         '4P_SHAP_inlet_velocity_18': 0.47456129, '4P_RANDOM_inlet_velocity_18': 14.34511645, '2P1V1N_inlet_velocity_18': 1.26163373}

minB :  {'4P_SHAP_drag_for_13': array([0.00131315]), '4P_RANDOM_drag_for_13': array([0.00387253]), '2P1V1N_drag_for_13': array([0.00010053]), 
         '4P_SHAP_lift_for_13': array([0.01930521]), '4P_RANDOM_lift_for_13': array([0.25233573]), '2P1V1N_lift_for_13': array([0.00086761]), 
         '4P_SHAP_angle_atta_13': array([0.00020233]), '4P_RANDOM_angle_atta_13': array([0.03700718]), '2P1V1N_angle_atta_13': array([0.00066574]), 
         '4P_SHAP_inlet_velocity_13': array([0.00028038]), '4P_RANDOM_inlet_velocity_13': array([0.08044052]), '2P1V1N_inlet_velocity_13': array([0.00117683]), 
         
         '4P_SHAP_drag_for_18': array([0.02502691]), '4P_RANDOM_drag_for_18': array([1.20393687]), '2P1V1N_drag_for_18': array([0.03268114]), 
         '4P_SHAP_lift_for_18': array([0.00344026]), '4P_RANDOM_lift_for_18': array([0.00610144]), '2P1V1N_lift_for_18': array([0.00070734]), 
         '4P_SHAP_angle_atta_18': array([0.02142002]), '4P_RANDOM_angle_atta_18': array([0.04614379]), '2P1V1N_angle_atta_18': array([0.00049198]), 
         '4P_SHAP_inlet_velocity_18': array([0.00018111]), '4P_RANDOM_inlet_velocity_18': array([5.79842739]), '2P1V1N_inlet_velocity_18': array([0.00066876])}

stdB :  {'4P_SHAP_drag_for_13': 1.3848004294901213, '4P_RANDOM_drag_for_13': 3.6528083291869446, '2P1V1N_drag_for_13': 1.3622853695111592, 
         '4P_SHAP_lift_for_13': 5.551821192672257, '4P_RANDOM_lift_for_13': 17.66451624051024, '2P1V1N_lift_for_13': 5.9637191783676835, 
         '4P_SHAP_angle_atta_13': 0.5902564960410247, '4P_RANDOM_angle_atta_13': 4.894457371856675, '2P1V1N_angle_atta_13': 0.5771624761483166, 
         '4P_SHAP_inlet_velocity_13': 0.22942297248569832, '4P_RANDOM_inlet_velocity_13': 3.0192517567328814, '2P1V1N_inlet_velocity_13': 0.7637670609266666, 
         
         '4P_SHAP_drag_for_18': 0.026973484067107194, '4P_RANDOM_drag_for_18': 0.9844397626513972, '2P1V1N_drag_for_18': 0.04664508779640156, 
         '4P_SHAP_lift_for_18': 0.21891245239757232, '4P_RANDOM_lift_for_18': 2.3245300589205913, '2P1V1N_lift_for_18': 0.2514431540478133, 
         '4P_SHAP_angle_atta_18': 0.02972898236006728, '4P_RANDOM_angle_atta_18': 3.8287731814864814, '2P1V1N_angle_atta_18': 0.021848330397201392, 
         '4P_SHAP_inlet_velocity_18': 0.05696268907576783, '4P_RANDOM_inlet_velocity_18': 2.8211027153514547, '2P1V1N_inlet_velocity_18': 0.13991280739192996}

'''
rsync -a ../../scratch/sialelli/results/WAVENET/4P_SHAP/model_wn_lift_for.h5 $TMPDIR/4P_SHAP
rsync -a ../../scratch/sialelli/results/WAVENET/4P_SHAP/model_wn_inlet_velocity.h5 $TMPDIR/4P_SHAP
rsync -a ../../scratch/sialelli/results/WAVENET/4P_SHAP/model_wn_drag_for.h5 $TMPDIR/4P_SHAP
rsync -a ../../scratch/sialelli/results/WAVENET/4P_SHAP/model_wn_angle_atta.h5 $TMPDIR/4P_SHAP
rsync -a ../../scratch/sialelli/results/WAVENET/4P_SHAP/dataset_info_lift_for.out $TMPDIR/4P_SHAP
rsync -a ../../scratch/sialelli/results/WAVENET/4P_SHAP/dataset_info_inlet_velocity.out $TMPDIR/4P_SHAP
rsync -a ../../scratch/sialelli/results/WAVENET/4P_SHAP/dataset_info_drag_for.out $TMPDIR/4P_SHAP
rsync -a ../../scratch/sialelli/results/WAVENET/4P_SHAP/dataset_info_angle_atta.out $TMPDIR/4P_SHAP

rsync -a ../../scratch/sialelli/results/WAVENET/4P_RANDOM/model_wn_lift_for.h5 $TMPDIR/4P_RANDOM
rsync -a ../../scratch/sialelli/results/WAVENET/4P_RANDOM/model_wn_inlet_velocity.h5 $TMPDIR/4P_RANDOM
rsync -a ../../scratch/sialelli/results/WAVENET/4P_RANDOM/model_wn_drag_for.h5 $TMPDIR/4P_RANDOM
rsync -a ../../scratch/sialelli/results/WAVENET/4P_RANDOM/model_wn_angle_atta.h5 $TMPDIR/4P_RANDOM
rsync -a ../../scratch/sialelli/results/WAVENET/4P_RANDOM/dataset_info_lift_for.out $TMPDIR/4P_RANDOM
rsync -a ../../scratch/sialelli/results/WAVENET/4P_RANDOM/dataset_info_inlet_velocity.out $TMPDIR/4P_RANDOM
rsync -a ../../scratch/sialelli/results/WAVENET/4P_RANDOM/dataset_info_drag_for.out $TMPDIR/4P_RANDOM
rsync -a ../../scratch/sialelli/results/WAVENET/4P_RANDOM/dataset_info_angle_atta.out $TMPDIR/4P_RANDOM

rsync -a ../../scratch/sialelli/results/WAVENET/2P1V1N/model_wn_lift_for.h5 $TMPDIR/2P1V1N
rsync -a ../../scratch/sialelli/results/WAVENET/2P1V1N/model_wn_inlet_velocity.h5 $TMPDIR/2P1V1N
rsync -a ../../scratch/sialelli/results/WAVENET/2P1V1N/model_wn_drag_for.h5 $TMPDIR/2P1V1N
rsync -a ../../scratch/sialelli/results/WAVENET/2P1V1N/model_wn_angle_atta.h5 $TMPDIR/2P1V1N
rsync -a ../../scratch/sialelli/results/WAVENET/2P1V1N/dataset_info_lift_for.out $TMPDIR/2P1V1N
rsync -a ../../scratch/sialelli/results/WAVENET/2P1V1N/dataset_info_inlet_velocity.out $TMPDIR/2P1V1N
rsync -a ../../scratch/sialelli/results/WAVENET/2P1V1N/dataset_info_drag_for.out $TMPDIR/2P1V1N
rsync -a ../../scratch/sialelli/results/WAVENET/2P1V1N/dataset_info_angle_atta.out $TMPDIR/2P1V1N
'''