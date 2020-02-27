# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 17:06:33 2020

@author: Ghjulia Sialelli
"""

'''
Where are the input nodes which are most correlated with the output ?
Based on correlation results from correlation.py, which tells us which
type of node is best correlated with a given output, we only focus on 
this kind of node in the present script.
'''

import pandas as pd
import numpy as np
import paths
import os.path
import math

NNODES = 818        # number of nodes for one dp
NPOINTS = 100       # number of points in a minute 
LENFRAME = 12       # length of a time frame to study (drop first 2 seconds)
NROWS = NNODES * (NPOINTS * LENFRAME + 1)   # number of rows to read for one dp
NDP2 = 296          # number of dp points for study2
NDP3 = 200          # number of dp points for study3



relevant_sensors_correlation = {'inlet_velocity [m/s]':['pressure'],
                    'coef_drag [-]':['velocity-magnitude'],
                    'coef_lift [-]':['velocity-magnitude'],
                    'drag_force [N]':['velocity-magnitude'],
                    'lift_force [N]':['velocity-magnitude'],
                    'angle_attack [°]':['velocity-magnitude']
                    } 


output = ['inlet_velocity [m/s]','coef_drag [-]','coef_lift [-]','drag_force [N]',	
       'lift_force [N]','angle_attack [°]']

inputs = ['pressure', 'velocity-magnitude','acoustic-source-power-db']

cols = output + inputs + ['nodenumber','dp_number','time [s]']


##############################################################################

"""
Returns a pandas DataFrame. Each column is composed of the values of the 
input_feature (e.g. 'pressure') for one nodenumber, from 2 to 12 seconds.

Because for now not sure how to read a lot of different dp_numbers, so 
instead of taking a few seconds (~ period of the time serie) for each 
experiment and average the importance of the location of the nodes over 
them, we look at the whole 10 seconds for one dp_number.

X1 : 'velocity-magnitude'
X2 = X4 : 'acoustic-source-power-db'
X3 = X5 = X6 : 'pressure'
"""
def create_X(df,dp_num, study_number,file_number):
    dp_num = dp_num + 50*file_number
    if os.path.isfile('Xs/X1_{}_{}.csv'.format(dp_num,study_number)) and os.path.isfile('Xs/X2_{}_{}.csv'.format(dp_num,study_number)):
        print('Xs already calculated.....reading from files...')
        X1 = pd.read_csv('Xs/X1_{}_{}.csv'.format(dp_num,study_number))
        X2 = pd.read_csv('Xs/X2_{}_{}.csv'.format(dp_num,study_number))
    else :
        X1,X2 = pd.DataFrame(),pd.DataFrame()
        for m in range(1,NNODES):
            tmpdf = df.drop(df[df.nodenumber != m].index)
            #tmpdf = tmpdf.drop('nodenumber', axis='columns')
            X1['node{}'.format(m)] = pd.Series(list(tmpdf['velocity-magnitude']))
            X2['node{}'.format(m)] = pd.Series(list(tmpdf['pressure']))
        X1.to_csv('Xs/X1_{}_{}.csv'.format(dp_num,study_number))
        X2.to_csv('Xs/X2_{}_{}.csv'.format(dp_num,study_number))

    return [X1,X2]


"""
Returns a pandas DataFrame. Since the truth values are the same for all 
nodenumbers, we just take one from the dataframe and filter out every column
but that of the output_feature we're interested in (e.g. 'drag force').
"""
def create_Y(df, output_feature):
    Y = pd.DataFrame()
    tmpdf = df.drop(df[df.nodenumber != 1].index)
    Y[output_feature] = tmpdf[output_feature]
    return pd.Series(np.asarray(Y).reshape((Y.shape[0],)))
        


def correlation(df, dp_num, study_number,file_number):
    study_i = study[study_number]
    inv_i = inv[study_number]
    
    # Processing
    
 
    df = df.drop(df[df.dp_number != dp_num].index)
    df = df.drop(columns=['dp_number','time [s]'])
    
    #print('\n DF : \n {} \n'.format(df))
    
    print('Creating Xs...')
    Xs = create_X(df,dp_num, study_number,file_number)
    print('Xs done!')
    
    
    for i, output_feature in enumerate(output[1:]) :
        print('Processing for output feature {}'.format(output_feature))
        
        print('Creating Y...')
        Y = create_Y(df, output_feature)
        print('Y done!')
        
        #print('Y : ', Y)
        
        if i == 0 : X = Xs[1]
        else : X = Xs[0]
        
        #print('X : ', X)
        
        # Xnode is a Pandas Series representing the feature of one nodenumber
        for nn in range(NNODES-1):
            Xnode = X.iloc[:,nn]
            #print('Xnode : ', Xnode)

            # Now we get the correlation between Xnode and Y
            cor =  pd.Series.cov(Xnode,pd.Series(Y))
            if math.isnan(cor) :
                cor = 0 
            else : cor = Xnode.corr(Y)
            study_i[output_feature][nn] += abs(cor)*inv_i
        


inv2 = 1/NDP2 
inv3 = 1/NDP3

inv = {2:inv2,
       3:inv3}

study2 = {key : np.asarray([0 for i in range(1,NNODES)]) for key in output}
study3 = {key : np.asarray([0 for i in range(1,NNODES)]) for key in output}

study = {2 : study2,
         3 : study3}

def main_corr(study_number) :
    for file_number, PATH in enumerate(paths.path_dict[study_number]) : 
        print('Processing file {}'.format(PATH))
        for i, df in enumerate(pd.read_csv(PATH, usecols = cols, chunksize=NROWS)):
            print('Processing new chunk')
            df = df.dropna()    
            df[['time [s]']] = df[['time [s]']].astype('float64')
            df = df.drop(df[2.0 > df['time [s]']].index) 
            df = df.drop(df[df['time [s]'] > 12.0].index)
            
            correlation(df,i,study_number,file_number)

main_corr(2)
np.save('loc_corr_study2.npy', study2)

# to load it back :
# study2 = np.load('loc_corr_study2.npy',allow_pickle='TRUE').item()


