#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:44:18 2020

@author: ghjuliasialelli
"""

import pandas as pd
import xgboost
import shap
import numpy as np
#import matplotlib.pyplot as plt
import itertools

outputs = ['inlet_velocity [m/s]','coef_drag [-]','coef_lift [-]','drag_force [N]',	
       'lift_force [N]','angle_attack [°]']

inputs = ['pressure', 'velocity-magnitude','acoustic-source-power-db']

nodenumbers = [i for i in range(1,818,2)]

print('listing all combinations')
combinations = list(itertools.combinations(nodenumbers, 3))

in_feat = 'pressure'
out_feat = 'pressure'

print('loading dataset')
df = pd.read_csv('naca_study1-4_validation_data.csv')

res = pd.DataFrame()
print('starting to construct dataframe')
for trio in combinations :
    a,b,c = trio 

    tempa = df[df['nodenumber'] == a][[in_feat]].values
    tempa = tempa.reshape(tempa.shape[0])
    
    tempb = df[df['nodenumber'] == b][[in_feat]].values
    tempb = tempb.reshape(tempb.shape[0])
    
    tempc = df[df['nodenumber'] == c][[in_feat]].values
    tempc = tempc.reshape(tempc.shape[0])
    
    combined = np.concatenate((tempa,tempb,tempc))
    
    res['nodes_{}_{}_{}'.format(a,b,c)] = pd.Series(combined)

output = df[::409][[out_feat]].values
output = output.reshape(output.shape[0])
#combined = np.concatenate((output,output,output))
#res[out_feat] = pd.Series(combined)
res[out_feat] = pd.Series(output)
res = res.dropna()

print('done')

#res.to_csv('shap_pressure_comb.csv', index = False)

X = res.drop('pressure', axis=1)
X.to_csv('X_comb.csv', index=False)

Y = res.filter(['pressure'], axis=1)

print('initiating model training')
model = xgboost.train({"learning_rate": 0.01},xgboost.DMatrix(X, label=Y))
exp = shap.TreeExplainer(model)
print('initiating shap values calculation')
shap_values = exp.shap_values(X)
print('done')

np.save("shap_pressure_combin.npy", shap_values)

#shap.summary_plot(shap_values, X, plot_type="bar", show=False)
#plt.title('SHAP values for surface pressure'.format(out_feat))
#plt.savefig('figs/plot_shap_pressure_comb.png'.format(out_feat[:-6]), bbox_inches = 'tight')
#plt.show()
