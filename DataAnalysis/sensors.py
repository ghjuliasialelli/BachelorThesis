# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:02:20 2020

@author: Administrateur
"""

"""

Which sensors (i.e. where they are located) are best for predicting ?

"""
import pandas as pd
import xgboost
import shap
import paths

NNODES = 818        # number of nodes for one dp
NPOINTS = 100       # number of points in a minute 
LENFRAME = 12       # length of a time frame to study (drop first 2 seconds)
NROWS = NNODES * (NPOINTS * LENFRAME + 1)   # number of rows to read for one dp
NDP2 = 296          # number of dp points for study2
NPD3 = 200          # number of dp points for study3


relevant_sensors_SHAP = {'inlet_velocity [m/s]':['velocity-magnitude'],
                    'coef_drag [-]':['acoustic-source-power-db'],
                    'coef_lift [-]':['pressure'],
                    'drag_force [N]':['acoustic-source-power-db'],
                    'lift_force [N]':['pressure'],
                    'angle_attack [°]':['pressure']
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
def create_X(df):
    X1,X2,X3 = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()  
    for m in range(1,NNODES+1):
        tmpdf = df.drop(df[df.nodenumber != m].index)
        #tmpdf = tmpdf.drop('nodenumber', axis='columns')
        X1['node{}'.format(m)] = pd.Series(list(tmpdf['velocity-magnitude']))
        X2['node{}'.format(m)] = pd.Series(list(tmpdf['acoustic-source-power-db']))
        X3['node{}'.format(m)] = pd.Series(list(tmpdf['pressure']))
    
    X4 = X2
    X5 = X3
    X6 = X3
    # remark : we store more than necessary, fix this within call inside SHAP 
    return [X1,X2,X3,X4,X5,X6]


"""
Returns a pandas DataFrame. Since the truth values are the same for all 
nodenumbers, we just take one from the dataframe and filter out every column
but that of the output_feature we're interested in (e.g. 'drag force').
"""
def create_Y(df, output_feature):
    Y = pd.DataFrame()
    tmpdf = df.drop(df[df.nodenumber != 1].index)
    Y[output_feature] = tmpdf[output_feature]
    return Y
        


"""
Obtain the SHAP values for all sensors for all output features.

dp_num : dp_number being processed
file_num : file being processed

From there, can obtain actual dp to index the dictionary on of the study 



"""
def SHAP(df, dp_num):
    
    # Processing
    df = df.dropna()    
    df[['time [s]']] = df[['time [s]']].astype('float64')
    df = df.drop(df[2.0 > df['time [s]']].index) 
    df = df.drop(df[df['time [s]'] > 12.0].index) 
    df = df.drop(df[df.dp_number != dp_num].index)
    df = df.drop(columns=['dp_number','time [s]'])
    
    print('\n DF : \n {} \n'.format(df))
    
    print('Creating Xs...')
    Xs = create_X(df)
    print('Xs done!')
    
    for i, output_feature in enumerate(output) :
        print('Processing for output feature {}'.format(output_feature))
        
        X = Xs[i]

        print('Creating Y...')
        Y = create_Y(df, output_feature)
        print('Y done!')
        model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=Y), 100)
        print('Shap values for output {}'.format(output_feature))
        print('Calculating SHAP values...')
        exp = shap.TreeExplainer(model)
        print('Done!')
        shap_values = exp.shap_values(X)
        #shap.summary_plot(shap_values, X, plot_type="bar")
        

""" 
Script to load chunks of data from the .csv file, to execute the above code for
different values of the dp_number, so as to average the importance of the node
sensors over different experiments.

study_number : int, specifies for which study the code should be run 

"""

def main_SHAP(study_number) :
    for PATH in paths.path_dict[study_number] : 
        print('Processing file {}'.format(PATH))
        # not sure about the skip rows 
        for i,df in enumerate(pd.read_csv(PATH, usecols = cols, chunksize=NROWS)):
            print('Processing chunk {}'.format(i))
            SHAP(df, i)




















        
