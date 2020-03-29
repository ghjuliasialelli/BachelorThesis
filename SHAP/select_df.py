#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 21:24:30 2020

@author: ghjuliasialelli
"""


import pandas as pd
import random

NPOINTS = 100                               # number of points in a second 
LENFRAME = 12                               # length of a time frame to study 
TOTROWS = 818 * (NPOINTS * LENFRAME + 1)

'''
PATHS = ['naca_study1_training_data1.csv',
         'naca_study1_training_data2.csv',
         'naca_study1_training_data3.csv',
         'naca_study1_training_data4.csv',
         'naca_study1_training_data5.csv',
         'naca_study1_training_data6.csv',
         'naca_study3_training_data1.csv',
         'naca_study3_training_data2.csv',
         'naca_study3_training_data3.csv',
         'naca_study3_training_data4.csv',
         'naca_study3_training_data5.csv',
         'naca_study3_training_data6.csv',
         'naca_study2_training_data1.csv',
         'naca_study2_training_data2.csv',
         'naca_study2_training_data3.csv',
         'naca_study2_training_data4.csv',
         'naca_study2_training_data5.csv',
         'naca_study2_training_data6.csv',
         'naca_study4_training_data1.csv',
         'naca_study4_training_data2.csv',
         'naca_study4_training_data3.csv',
         'naca_study4_training_data4.csv',
         'naca_study4_training_data5.csv',
         'naca_study4_training_data6.csv']
'''

PATHS = ['naca_study1_training_data1.csv',
         'naca_study1_training_data2.csv',
         'naca_study1_training_data3.csv',
         'naca_study3_training_data1.csv',
         'naca_study3_training_data2.csv',
         'naca_study3_training_data3.csv',
         'naca_study2_training_data1.csv',
         'naca_study2_training_data2.csv',
         'naca_study2_training_data3.csv',
         'naca_study4_training_data1.csv',
         'naca_study4_training_data2.csv',
         'naca_study4_training_data3.csv'] 

listofdf = []

for PATH in PATHS :
    print('Processing file {}'.format(PATH))
    rand = []
    
    for i in range(5):
        print('{}-th iteration'.format(i))
        
        surprise = random.randrange(0,48)
        while surprise in rand : 
            surprise = random.randrange(0,48)
        rand.append(surprise)
        
        print('...isolating dp')
        df = pd.read_csv(PATH, nrows = TOTROWS, skiprows = [i for i in range(1, 1 + surprise * TOTROWS)])
        print('...done')
        listofdf.append(df)
        
print('Done!')

bdf = pd.concat(listofdf)

print('Writing to file.')
bdf.to_csv('random_SHAP.csv')