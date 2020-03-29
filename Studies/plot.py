#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:31:56 2020

@author: ghjuliasialelli

Comparison
"""

import matplotlib.pyplot as plt
import numpy as np

epochs = 25 
xrange = np.arange(1,epochs+1)

models = ['XGB',  'WAVENET', 'TCN', 'LSTM']

index = 3
corres = ['drag_force [N]','lift_force [N]', 'angle_attack [°]', 'inlet_velocity [m/s]']
outputs = corres[index]

'''
tests_wn = []
tests_xgb = []
tests_lstm = []

for feat in corres : 
    tests_wn.append(np.loadtxt('../WAVENET/mixed/MSE_train_wn_{}.out'.format(feat[:-6]), delimiter=','))
    tests_xgb.append(np.loadtxt('../XGB/mixed/MSE_train_xgb{}.out'.format(feat[:-6]), delimiter=','))
    
for i,feat in enumerate(corres) : 
    plt.plot(xrange, tests_wn[i], label = 'Wavenet model')
    plt.plot(xrange, tests_xgb[i], label = 'XGBRegressor model')
    plt.legend(loc='upper right')
    plt.title('Models performance comparison for predicting {}'.format(feat))
    plt.ylabel('MSE')
    plt.xlabel('epoch')
    plt.tight_layout(pad=3.0)
    plt.savefig('comparison_{}.png'.format(feat[:-6]))
    plt.show()
'''

feat = outputs

print(feat)

plt.plot(xrange, np.loadtxt('../WAVENET/23.03/2P1V1N/MSE_train_wn_{}.out'.format(feat[:-6]), delimiter=','), label = 'WaveNet 2P1V1N')
plt.plot(xrange, np.loadtxt('../XGB/23.03/2P1V1N/MSE_train_xgb{}.out'.format(feat[:-6]), delimiter=','), label = 'XGBRegressor 2P1V1N')
plt.plot(xrange, np.loadtxt('../WAVENET/23.03/4P_SHAP/MSE_train_wn_{}.out'.format(feat[:-6]), delimiter=','), label = 'WaveNet 4P_SHAP')
plt.plot(xrange, np.loadtxt('../XGB/23.03/4P_SHAP/MSE_train_xgb{}.out'.format(feat[:-6]), delimiter=','), label = 'XGBRegressor 4P_SHAP')

plt.legend()
plt.title('Models performance comparison for predicting {}'.format(feat))
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.tight_layout(pad=3.0)
plt.savefig('cmp_wn_xgb_2P1V1N_{}.png'.format(feat[:-6]))
plt.show()


plt.plot(xrange[15:], np.loadtxt('../WAVENET/23.03/2P1V1N/MSE_train_wn_{}.out'.format(feat[:-6]), delimiter=',')[15:], label = 'Wavenet 2P1V1N')
plt.plot(xrange[15:], np.loadtxt('../XGB/23.03/2P1V1N/MSE_train_xgb{}.out'.format(feat[:-6]), delimiter=',')[15:], label = 'XGBRegressor 2P1V1N')
plt.plot(xrange[15:], np.loadtxt('../WAVENET/23.03/4P_SHAP/MSE_train_wn_{}.out'.format(feat[:-6]), delimiter=',')[15:], label = 'Wavenet 4P_SHAP')
plt.plot(xrange[15:], np.loadtxt('../XGB/23.03/4P_SHAP/MSE_train_xgb{}.out'.format(feat[:-6]), delimiter=',')[15:], label = 'XGBRegressor 4P_SHAP')


plt.legend()
plt.title('Models performance comparison for predicting {}'.format(feat))
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.tight_layout(pad=3.0)
plt.savefig('cmp_closeup_wn_xgb_2P1V1N_{}.png'.format(feat[:-6]))
plt.show()



def plot_individual(outputs) :
    test = np.loadtxt('std/MSE_val_wn_{}.out'.format(outputs[:-6]), delimiter=',')
    val = np.loadtxt('std/MSE_train_wn_{}.out'.format(outputs[:-6]), delimiter=',')
    
    plt.plot(xrange, test, label='testing loss')
    plt.plot(xrange, val, label='training loss')
    plt.legend(loc='upper right')
    plt.title('Wavenet performance for predicting {}'.format(outputs))
    plt.ylabel('MSE')
    plt.xlabel('epoch')
    
    plt.tight_layout(pad=3.0)
    plt.savefig('WN_{}_{}.png'.format(outputs[:-6], epochs))
    plt.show()
    
