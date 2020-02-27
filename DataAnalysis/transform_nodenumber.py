# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:59:36 2020

@author: Ghjulia Sialelli

Script to transform the nodenumbers in the dataset
"""

import pandas as pd
from math import sqrt, atan2, pi
import numpy as np
import paths

NNODES = 818        # number of nodes for one dp
NPOINTS = 100       # number of points in a minute 
LENFRAME = 12       # length of a time frame to study (drop first 2 seconds)
NROWS = NNODES * (NPOINTS * LENFRAME + 1)   # number of rows to read for one dp


def create_base():
    # We load the data for the first 818 nodes, that we consider to be our base
    # data, since the angle of attack is 0
    base_df = pd.read_csv('naca_study2_training_data1.csv', nrows = 819)
    
    # Isolate x and y coordinates, and remove columns names
    Xs = np.asarray(base_df.iloc[:]['x-coordinate'])
    Ys = np.asarray(base_df.iloc[:]['y-coordinate'])
    NN = np.asarray([i for i in range(1,819)])

    mapping = dict()
    
    for x,y,nn in zip(Xs,Ys,NN):
        r = sqrt(pow(x,2)+pow(y,2))
        alpha = 180/pi * atan2(y,x)
        
        mapping[(r,alpha)] = (x,y,nn) 
                
    np.save('transform_data/base_data.npy', mapping)

'''
Helper function
'''
def dist(a,b):
    x1,y1=a[0],a[1]
    x2,y2=b[0],b[1]
    return pow(x1-x2,2)+pow(y1-y2,2)


'''
Function to retrieve real xnew and ynew coordinates of a node from its own x
and y coordinate, along with its angle of attack (alpha).

input : alpha, angle of attack in degrees
output : (xnew,ynew) : coordinates of the node
'''
def retrieve(df, angle_attack):
    Xs = np.asarray(df['x-coordinate'])
    Ys = np.asarray(df['y-coordinate'])
    
    if angle_attack == 0 :
        return Xs,Ys,np.asarray([i for i in range(1,819)])

    else :
        Xnew = []
        Ynew = []
        NN = []
        
        keys_list = list(mapping.keys())
        val_list = list(mapping.values())

        for x,y in zip(Xs,Ys):
            r = sqrt(pow(x,2)+pow(y,2))
            beta = 180/pi * atan2(y,x)
            alpha = beta - angle_attack 
            
            # If we obtain a perfect mapping
            if (r,alpha) in keys_list :
                xnew,ynew,nn = mapping[(r,alpha)]
                keys_list.remove((r,alpha))
            else : 
                # Find point closest to one obtained mathematically
                xnew,ynew,nn = min(val_list, key = lambda t : dist((t[0],t[1]),(x,y)))
                # Remove it from the list so that future points cannot be assigned to same node
                val_list.remove((xnew,ynew,nn))
                
            Xnew.append(xnew)
            Ynew.append(ynew)
            NN.append(nn)
    
        return Xnew,Ynew,NN


cols = ['nodenumber','angle_attack [°]','time [s]','x-coordinate','y-coordinate']
    
def replace(PATH,file_num):
    for chunk_num,df in enumerate(pd.read_csv(PATH, chunksize = NROWS)):
        print('Processing chunk {}'.format(chunk_num))
        df = df[cols]
        
        # Split df into many dataframes, on for each timestep
        timesteps = df['time [s]'].unique()
                
        Xs = []
        Ys = []
        NNs = [] 
        
        print('Iterating over timesteps......')
        for time in timesteps : 
            # print('Processing time {}'.format(time))
            
            df_time = df.drop(df[df['time [s]'] != time].index)
            angle = df_time['angle_attack [°]'].unique()[0]
            
            Xnew,Ynew,NN = retrieve(df_time, angle)
            Xs.append(Xnew)
            Ys.append(Ynew)
            NNs.append(NN)

        df['newx-coordinate'] = pd.Series(np.concatenate(Xs))
        df['neyx-coordinate'] =  pd.Series(np.concatenate(Xs))
        df['new_nodenumber'] = pd.Series(np.concatenate(NNs))
        
        print('Writing dataframe for chunk to file......')
        df.to_csv('transform_data/file{}_chunk{}.csv'.format(file_num,chunk_num))
        print('Done')

print('Creating base........')
create_base()  
print('Done') 
mapping = np.load('transform_data/base_data.npy',allow_pickle='TRUE').item()     
for i,PATH in enumerate(paths.PATHS2) : 
    print('Treating file {}'.format(i+1))
    replace(PATH,i+1)