# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:32:20 2020

@author: Ghjulia Sialelli
"""

LASTROW = 163601   # last row at which time < 2 s
NNODES = 818        # number of nodes for one dp
NPOINTS = 100       # number of points in a minute 
LENFRAME = 13       # length of a time frame to study (drop first 2 seconds)
NROWS = NNODES * NPOINTS * LENFRAME   # number of rows to read for one dp
MAX = 1000000

PATH = 'naca_study2_training_data1.csv'

import pandas as pd
import matplotlib.pyplot as plt

"""
Method to extract the time series for dp n, nodenumber m, 
from s1 to s2 seconds, from dataset with path PATH, for 
the input feature feat (passed as a string)
"""
def time_series(n, m, PATH, feat, s1=2.0, s2=12.0, plot=True):
    # Read the csv with the appropriate columns and rows
    cols = ['time [s]','nodenumber','dp_number']
    cols.append(feat)
    df = pd.read_csv(PATH, skiprows = lambda x: x in [1, 1 + n*NROWS], 
                     nrows = MAX, header=0, usecols = cols, parse_dates=['time [s]'])
    
    # Remove data from the node numbers we dont care about
    df=df.drop(df[df.nodenumber != m].index)
    
    # After having done the necessary cuts, can drop the nodenumber 
    # and dp_number columns
    df = df.drop('nodenumber', axis='columns')
    df = df.drop('dp_number', axis='columns')
    
    
    # Prepare for time series format 
    df[['time [s]']] = df[['time [s]']].astype('float64')
    df = df.drop(df[s1 > df['time [s]']].index) 
    df = df.drop(df[df['time [s]'] > s2].index) 
    df['datetime'] = pd.to_datetime(df['time [s]'],unit='s')
    df = df.set_index('datetime')
    df.drop(['time [s]'], axis=1, inplace=True)
    
    if plot == True:
        df.plot()
    
    return df

"""
Returns the frequency-string of the given time serie.
S : one second
L : one millisecond

Not relevant.
"""
def get_period(df):
    return pd.infer_freq(df.index)




