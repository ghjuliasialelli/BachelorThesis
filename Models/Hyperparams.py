# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:32:50 2020

@author: Ghjulia Sialelli

Hyperparameters tuning of the models :
    XGBoost
    LSTM
    Wavenet
    TCN

Based on the data for one cylinder study, using scikit-learn GridSearchCV 
"""
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
import XGBoost, LSTM, Wavenet, TCN
K = 3           # k used for performing k-fold cross validation



'''
Wrapper function to tune the hyperparameters of any of the three models.
Input :
    fn : the building function of the model
    X : the input data of the desired model
    Y : the output data of the desired model
'''
def tune(fn, X, Y):
    model = KerasRegressor(build_fn = fn, verbose = 1)
    batch_size = [10,20,40,60,80,100,120,150]
    epochs = [10,50,100]
    param_grid = dict(batch_size = batch_size, epochs = epochs)
    grid = GridSearchCV(estimator = model, param_grid = param_grid, n_jobs = 1, cv = K)
    grid_result = grid.fit(X,Y)
    print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))


##############################################################################
# For LSTM
##############################################################################
def create_LSTM():
    return LSTM.model 

def tune_LSTM():
    return tune(create_LSTM, LSTM.X, LSTM.Y)

##############################################################################
# For Wavenet
##############################################################################
def create_WN():
    return Wavenet.wn.model 

def tune_WN():
    return tune(create_WN, Wavenet.X, Wavenet.Y)
    
##############################################################################
# For TCN
##############################################################################
def create_TCN():
    return TCN.tcn.model 

def tune_TCN():
    return tune(create_TCN, TCN.X, TCN.Y)

##############################################################################
# For XGBoost
##############################################################################



