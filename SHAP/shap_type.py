#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:48:54 2020

@author: ghjuliasialelli

Which sensors (what type) are best for predicting ?
"""
import pandas as pd
import xgboost
import shap
import sys
import matplotlib.pyplot as plt
import os.path
import numpy as np


NNODES = 409                                # number of nodes for one dp
NPOINTS = 100                               # number of points in a second 
LENFRAME = 12                               # length of a time frame to study 
NROWS = NNODES * (NPOINTS * LENFRAME + 1)   # number of rows to read for one dp  


output = ['inlet_velocity [m/s]','coef_drag [-]','coef_lift [-]','drag_force [N]',	
       'lift_force [N]','angle_attack [°]']

inputs = ['pressure', 'velocity-magnitude','acoustic-source-power-db']

#arg = int(sys.argv[1])
arg = 1

print('SHAP Values script for predicting {}'.format(output[arg-1]))

##############################################################################

if os.path.isfile('X_shap.csv') and os.path.isfile('Y_shap_{}.csv'.format(arg)): 
    X = pd.read_csv('X_shap.csv')
    Y = pd.read_csv('Y_shap_{}.csv'.format(arg))

else :
    X = pd.read_csv('../data/naca_study1-4_validation_data.csv', usecols = inputs)
    if arg == 1 : X.to_csv('X_shap.csv', index = False)

    Y = pd.read_csv('../data/naca_study1-4_validation_data.csv', usecols = [output[arg-1]])
    Y.to_csv('Y_shap_{}.csv'.format(arg), index = False)

#print('Initiating training of model')
#model = xgboost.train({"learning_rate": 0.01},xgboost.DMatrix(X, label=Y))

#exp = shap.TreeExplainer(model)

print('Initiating calculation of SHAP values')
#shap_values = exp.shap_values(X)
#np.save('SHAP_xgb_{}.npy'.format(output[arg-1]), shap_values)

shap_values = np.load("SHAP_xgb_inlet_velocity.npy")

shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title('SHAP values for predicting inlet_velocity [m/s]')
plt.savefig('SHAP_xgb_inlet_velocity.png', bbox_inches = 'tight')
plt.show()