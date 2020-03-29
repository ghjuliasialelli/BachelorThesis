#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:41:55 2020

@author: ghjuliasialelli

Get SHAP Values for XGBRegressor and validation data
"""

"""5
Which sensors (where) are best for predicting ?
"""
import pandas as pd
import xgboost
import shap
import numpy as np
import matplotlib.pyplot as plt

import cells818

outputs = ['inlet_velocity [m/s]','coef_drag [-]','coef_lift [-]','drag_force [N]',	
       'lift_force [N]','angle_attack [°]']

inputs = ['pressure', 'velocity-magnitude','acoustic-source-power-db']

bijection = {'inlet_velocity [m/s]' : 'velocity-magnitude',
             'drag_force [N]' : 'velocity-magnitude',	
             'lift_force [N]' : 'pressure',
             'angle_attack [°]' : 'acoustic-source-power-db',
             'pressure' : 'pressure'}

# Load cell_ids from file
cell_ids = cells818.cells

def create_dataframe(out_feat,df):
    in_feat = bijection[out_feat]
    res = pd.DataFrame()
    
    ########### cell_ids ###
    for cell in cell_ids : 
        
        ############# 'cell-id' #############################
        temp = df[df['cell-id'] == cell][[in_feat]].values
        temp = temp.reshape(temp.shape[0])
        
        
        res['Cell_{}'.format(cell)] = pd.Series(temp)

    output = df[::818][[out_feat]].values
    output = output.reshape(output.shape[0])
    res[out_feat] = pd.Series(output)
    

    return res.dropna(axis=1)

def run_create(df, out_feat):
    #for out_feat in outputs : 
    print('Processing output {}'.format(out_feat))
    res = create_dataframe(out_feat,df)
    print('...........writing to file')
    res.to_csv('shap_loc_df_{}.csv'.format(out_feat[:-6]), index = False)


def SHAP(out_feat, file):
    print('Initiating SHAP for {}'.format(out_feat))
    
    cols = pd.read_csv(file, nrows=1).columns
    X = pd.read_csv(file, usecols = cols[:-1])
    Y = pd.read_csv(file, usecols = [len(cols)-1])
    
    print('....training')
    model = xgboost.train({"learning_rate": 0.01},xgboost.DMatrix(X, label=Y))
    exp = shap.TreeExplainer(model)
    print('....calculating shap values')
    shap_values = exp.shap_values(X)
    
    print('....saving to file')
    np.save("SV_loc_{}.npy".format(out_feat[:-6]), shap_values)
    
    #print('....summary plot')
    #shap.summary_plot(shap_values, X, plot_type="bar", show = False)
    #matplotlib.pyplot.title('Influence of sensor location for predicting {}'.format(out_feat))
    #matplotlib.pyplot.savefig('shap_loc_plot_{}.png'.format(out_feat[:-6]), bbox_inches = 'tight')

#df = pd.read_csv('random_SHAP.csv')
#df = run_create(df, 'pressure')
#SHAP('pressure','shap_loc_df_pr.csv')

# Uncomment to obtain the datasets
#run_create(df, 'pressure')

# Uncomment to obtain the SHAP values and plots
#SHAP('lift_force [N]', 'shap_loc_df_lift_for.csv')

cols = pd.read_csv('shap_loc_df_pr.csv', nrows=1).columns
X = pd.read_csv('shap_loc_df_pr.csv', usecols = cols[:-1])

shap_values = np.load("SV_loc_pr.npy")
shap.summary_plot(shap_values, X, plot_type="bar", show = False)
plt.title('Influence of sensor location for predicting Pressure')
#plt.savefig('shap_loc_plot_pr.png', bbox_inches = 'tight')