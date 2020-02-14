# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:33:57 2020

@author: Administrateur
"""

import pandas as pd
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt


df = pd.read_csv('naca_study2_training_data1.csv', skiprows= lambda x: x in [1, 163602], nrows=1000000) # Load the data
df = df.dropna()

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

output = ['inlet_velocity [m/s]','coef_drag [-]','coef_lift [-]','drag_force [N]',	
       'lift_force [N]','angle_attack [°]']

# output parameters
Y = df[output]

# input parameters
X = df[['pressure', 'velocity-magnitude','acoustic-source-power-db']]
#X = df.drop(output, axis=1)

# Split the data into train and test data:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)


# Build the model with the random forest regression algorithm:
model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
model.fit(X_train, Y_train)

import shap
shap_values = shap.TreeExplainer(model).shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")