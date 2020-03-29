#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:26:33 2020

@author: ghjuliasialelli

Plot
"""
import matplotlib.pyplot as plt
import numpy as np

tests = [np.loadtxt('RMSE_test_xgb1.out', delimiter=','),
         np.loadtxt('RMSE_test_xgb2.out', delimiter=','),
         np.loadtxt('RMSE_test_xgb3.out', delimiter=','),
         np.loadtxt('RMSE_test_xgb4.out', delimiter=','),
         np.loadtxt('RMSE_test_xgb5.out', delimiter=','),
         np.loadtxt('RMSE_test_xgb6.out', delimiter=',')]


trains = [np.loadtxt('RMSE_train_xgb1.out', delimiter=','),
         np.loadtxt('RMSE_train_xgb2.out', delimiter=','),
         np.loadtxt('RMSE_train_xgb3.out', delimiter=','),
         np.loadtxt('RMSE_train_xgb4.out', delimiter=','),
         np.loadtxt('RMSE_train_xgb5.out', delimiter=','),
         np.loadtxt('RMSE_train_xgb6.out', delimiter=',')]

outputs = ['inlet_velocity [m/s]','coef_drag [-]','coef_lift [-]','drag_force [N]',
           'lift_force [N]','angle_attack [°]']

cropped_out = ['velocity','coef_drag','coef_lift', 'drag_force', 'lift_force', 'angle_attack']


epochs=10
xrange = np.arange(1,epochs+1)

fig, axes = plt.subplots(2,3,figsize=(20,7))

def plot_xgb(subplot,i):
    subplot.plot(xrange, np.square(tests[i])[:epochs], label='testing loss')
    subplot.plot(xrange, np.square(trains[i])[:epochs], label='training loss')
    subplot.legend(loc='upper right')
    subplot.set_title('XGBRegressor performance for predicting {}'.format(outputs[i]))
    subplot.set_ylabel('MSE')
    subplot.set_xlabel('epoch')

plot_xgb(axes[0,0],0)
plot_xgb(axes[0,1],1)
plot_xgb(axes[0,2],2)
plot_xgb(axes[1,0],3)
plot_xgb(axes[1,1],4)
plot_xgb(axes[1,2],5)

fig.tight_layout(pad=3.0)
fig.savefig('XGB_{}.png'.format(epochs))
fig.show()