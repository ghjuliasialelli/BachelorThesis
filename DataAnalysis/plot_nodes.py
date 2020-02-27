# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:08:32 2020

@author: Ghjulia Sialelli
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#df = pd.read_csv('naca_study2_training_data1.csv', nrows = 1000000)
#df = pd.read_csv('transform_data/file1_chunk0.csv', nrows = 1000000)
df = pd.read_csv('naca_study2_training_data1.csv', nrows = 100000)



#df[['time [s]']] = df[['time [s]']].astype('float64')
#df = df.drop(df[2.0 > df['time [s]']].index) 

# retrieve all nodes for one timestep 

#times = [2.0,2.01,2.02,2.03,2.04,2.05]

times = [0.0,0.1]

nodenumbers = [i for i in range(817)]
fig, axs = plt.subplots(len(times), figsize=(150,150))

for i,time in enumerate(times) :
    dff = df.drop(df[df['time [s]']!=time].index) 
    x = np.asarray(dff['x-coordinate'])
    y =  np.asarray(dff['y-coordinate'])
    nn = np.asarray(dff['nodenumber'])
    
    
    axs[i].scatter(x,y)
    
    for j, text in enumerate(nn):
        axs[i].annotate(text, (x[j], y[j]))

fig.savefig('node_location_try0.png')