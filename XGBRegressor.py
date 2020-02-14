# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:53:21 2020

@author: Administrateur
"""

NROWS = 1000000


import pandas as pd
import xgboost
import shap

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('naca_study2_training_data1.csv', skiprows= lambda x: x in [1, 163602], nrows=NROWS) # Load the data
df = df.dropna()

output = ['inlet_velocity [m/s]','coef_drag [-]','coef_lift [-]','drag_force [N]',	
       'lift_force [N]','angle_attack [°]']

def create_Ys(output):
    output_labels = {}
    for label in output:
        output_labels[label]=df[label]
    return output_labels        

X = df[['pressure', 'velocity-magnitude','acoustic-source-power-db']]

for label,Y in create_Ys(output).items():
    #X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2)
    model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=Y), 100)
    #model.fit(X,Y)
    #Y_pred = model.predict(X_test)

    #rmse_score = mse(Y_test, Y_pred) #squared=False)
    #print('RMSE for output {} : %.5f'.format(label) % rmse_score)
    
    #data = X.head(100)
    print('Shap values for output {}'.format(label))
    exp = shap.TreeExplainer(model)
    shap_values = exp.shap_values(X)
    shap.summary_plot(shap_values, X, plot_type="bar")
    # visualize the training set predictions
    shap.summary_plot(shap_values, X)


