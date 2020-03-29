# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 09:24:21 2020

@author: Ghjulia Sialelli

XGBoost Model implementation
"""

from sklearn.model_selection import train_test_split 
import xgboost as xgb
import pandas as pd
import numpy as np
import os.path
import sys


# 0 (for drag force) or 1 (for lift force) or 2 or 3
arg = int(sys.argv[1])
corres = ['drag_force [N]','lift_force [N]', 'angle_attack [°]', 'inlet_velocity [m/s]']
outputs = corres[arg]

EPOCHS = 25
#############################################################################################
bij_type = {'inlet_velocity [m/s]' : 'velocity-magnitude',
             'drag_force [N]' : 'velocity-magnitude',	
             'lift_force [N]' : 'pressure',
             'angle_attack [°]' : 'acoustic-source-power-db'}

inputs = bij_type[outputs]

#####  4P_SHAP  #########################################
rel_ids = [7347.5, 7301.49452, 7295.5, 7349.4958]
rel_types = ['pressure','pressure','pressure','pressure']
# all pressure ##########################################

##### 4P_RANDOM ##########################################
#rel_ids = [7435.5, 7243.5, 6613.49682, 7889.50247]      
#rel_types = ['pressure','pressure','pressure','pressure']
##########################################################

######## 2P1V1N ####################################################################
#rel_ids = [7347.5, 7301.49452, 6527.5, 7315.49998]    
#rel_types = ['pressure','pressure','velocity-magnitude','acoustic-source-power-db']
####################################################################################

######## 100P ##############################################################################
#rel_ids = [7783.5    , 7229.50231, 8057.501690000001, 6961.497170000001, 6933.502579999999,
#           7467.5    , 7645.4986 , 6643.5    , 7291.5    , 7013.49951,
#           6759.5    , 6747.5    , 7559.5    , 7217.496279999999, 6509.48116,
#           6815.5    , 8101.50508, 6887.5    , 6581.50339, 8061.505090000001,
#           6875.5    , 6773.496940000001, 7391.5    , 6805.49917, 6913.4985400000005,
#           7099.5    , 7347.5    , 6711.5    , 7409.49568, 7359.5    ,
#           6883.5    , 7033.4993 , 7261.5018900000005, 8037.50267, 7067.5    ,
#           6583.5    , 7043.5    , 6747.5    , 7187.5    , 6903.5    ,
#           7645.4986 , 6963.5    , 7553.499379999999, 6553.50361, 7195.5    ,
#           7281.5021799999995, 6643.5    , 7531.5    , 7737.50252, 6633.49952,
#           7915.5    , 8041.50162, 6973.49915, 6845.5005200000005, 7553.499379999999,
#           7109.501740000001, 6949.5    , 6529.49715, 7529.5018 , 6869.499629999999,
#           7455.5    , 6671.5    , 6537.50187, 6963.5    , 6519.49998,
#           7627.5    , 8015.5    , 7717.49777, 6973.49915, 6903.5    ,
#           7103.5    , 6675.5    , 7317.501509999999, 7705.499640000001, 7327.49997,
#           7957.49608, 6989.5005599999995, 7057.4999099999995, 7629.5016 , 6703.5    ,
#           8111.5    , 7843.5    , 7483.5    , 8135.4992 , 7747.5    ,
#           7317.501509999999, 7099.5    , 6723.5    , 8021.501029999999, 7489.4994799999995,
#           7945.50285, 7007.5    , 7411.5    , 7151.5    , 7897.50247,
#           7623.5    , 7679.5    , 7315.49998, 7411.5    , 7035.5    ]
#rel_types = ['pressure' for _ in range(len(rel_ids))]
#############################################################################################


NNODES = len(rel_ids)                       # number of sensors we consider
NPOINTS = 100                               # number of points in a second 
LENFRAME = 12                               # length of a time frame to study 
NROWS = NNODES * (NPOINTS * LENFRAME + 1)   # equivalent of one experiment (in rows)
RATIO = 0.25                                # Ratio of testing to training set

TOTROWS = 818 * (NPOINTS * LENFRAME + 1)

PATHS = ['naca_study1_training_data1.csv',
         'naca_study1_training_data2.csv',
         'naca_study1_training_data3.csv',
         'naca_study3_training_data1.csv',
         'naca_study3_training_data2.csv',
         'naca_study3_training_data3.csv',
         'naca_study2_training_data1.csv',
         'naca_study2_training_data2.csv',
         'naca_study2_training_data3.csv',
         'naca_study4_training_data1.csv',
         'naca_study4_training_data2.csv',
         'naca_study4_training_data3.csv'] 

#############################################################################################
'''
Standardizing the dataset
'''
def standardize(X, std = None, mean = None):
    Xresh = X.reshape((X.shape[0]*X.shape[1],))
    if mean == None : mean = np.mean(Xresh)
    if std == None : std = np.std(Xresh)
    return (X-mean)/std, std, mean

'''
Post-processing the data to feed it to Wavenet
'''
def transform_dataset(PATHS):
    Xs = []
    Ys = []
    
    for PATH in PATHS :
        for df in pd.read_csv(PATH, chunksize = TOTROWS):             
            # Process dataframe
            #df = df.dropna() 
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



'''
Helper function to train and evaluate a given XGBRegressor model.
input :
    model : XGBRegressor model to train and evaluate
    X_train,X_test,Y_train,Y_test : testing and training sets 
output :
    results['validation_0']['rmse'] : array, training root-mean-squared-error of the model
    results['validation_1']['rmse'] : array, testing root-mean-squared-error of the model
'''
def train_model(model,X_train,X_test,Y_train,Y_test):
    eval_set = [(X_train, Y_train), (X_test, Y_test)]
    model.fit(X_train, Y_train, eval_metric = 'rmse', eval_set = eval_set, verbose = True)
    results = model.evals_result()
    return results['validation_0']['rmse'],results['validation_1']['rmse']


#############################################################################################

print('Transforming / Loading dataset........')
if os.path.isfile('X_{}.npy'.format(outputs[:-6])) and os.path.isfile('Y_{}.npy'.format(outputs[:-6])) : 
    X = np.load('X_{}.npy'.format(outputs[:-6]),allow_pickle='TRUE')
    Y = np.load('Y_{}.npy'.format(outputs[:-6]),allow_pickle='TRUE')
else:    
    X, Y = transform_dataset(PATHS)
    np.save('X_{}.npy'.format(outputs[:-6]), X)
    np.save('Y_{}.npy'.format(outputs[:-6]), Y)
print('Done!')

print('X shape : ', X.shape)
print('Y shape : ', Y.shape)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = RATIO, shuffle = True)

# Now, we standardize the data. We obtain the std and mean of the training data, and apply it to the testing data
X_train, Xtr_std, Xtr_mean = standardize(X_train)
Y_train, Ytr_std, Ytr_mean = standardize(Y_train)
X_test, _, _              = standardize(X_test, Xtr_std, Xtr_mean)
Y_test, _, _              = standardize(Y_test, Ytr_std, Ytr_mean)

np.savetxt('dataset_info_{}.out'.format(outputs[:-6]), np.asarray([Xtr_std, Xtr_mean, Ytr_std, Ytr_mean]), delimiter=',')

model = xgb.XGBRegressor(n_estimators = EPOCHS, objective = 'reg:squarederror', booster = 'gbtree')
print('Training and evaluating the model')
model.fit(X_train, Y_train, eval_metric = 'rmse', eval_set = [(X_train, Y_train), (X_test, Y_test)], verbose = True)
results = model.evals_result()
print('Done')
RMSE_train, RMSE_test = results['validation_0']['rmse'],results['validation_1']['rmse']

# Saving the losses to files
np.savetxt('MSE_train_xgb{}.out'.format(outputs[:-6]),np.square(np.asarray(RMSE_train)), delimiter=',')
np.savetxt('MSE_test_xgb{}.out'.format(outputs[:-6]), np.square(np.asarray(RMSE_test)), delimiter=',')


print('Saving model to files')
bst = model.get_booster()
bst.save_model("model{}_xgb.json".format(outputs[:-6]))

# to load again, with X_predict = xgb.DMatrix(X_predict)
# bst = xgb.Booster()
# bst.load_model("model{}_xgb.json".format(outputs[:-6]))
# bst.predict(X_predict)


