# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 16:32:04 2020

@author: Ghjulia Sialelli
"""

NNODES = 818        # number of nodes for one dp
NPOINTS = 100       # number of points in a minute 
LENFRAME = 12       # length of a time frame to study (drop first 2 seconds)
NROWS = NNODES * (NPOINTS * LENFRAME + 1)   # number of rows to read for one dp
import paths

import pandas as pd

output = ['inlet_velocity [m/s]','coef_drag [-]','coef_lift [-]','drag_force [N]',	
       'lift_force [N]','angle_attack [°]']

inputs = ['pressure', 'velocity-magnitude','acoustic-source-power-db']

keys = [('pressure', 'inlet_velocity [m/s]'),
 ('velocity-magnitude', 'inlet_velocity [m/s]'),
 ('acoustic-source-power-db', 'inlet_velocity [m/s]'),
 ('pressure', 'coef_drag [-]'),
 ('velocity-magnitude', 'coef_drag [-]'),
 ('acoustic-source-power-db', 'coef_drag [-]'),
 ('pressure', 'coef_lift [-]'),
 ('velocity-magnitude', 'coef_lift [-]'),
 ('acoustic-source-power-db', 'coef_lift [-]'),
 ('pressure', 'drag_force [N]'),
 ('velocity-magnitude', 'drag_force [N]'),
 ('acoustic-source-power-db', 'drag_force [N]'),
 ('pressure', 'lift_force [N]'),
 ('velocity-magnitude', 'lift_force [N]'),
 ('acoustic-source-power-db', 'lift_force [N]'),
 ('pressure', 'angle_attack [°]'),
 ('velocity-magnitude', 'angle_attack [°]'),
 ('acoustic-source-power-db', 'angle_attack [°]')]


def file(PATH):
    file = {key: 0 for key in keys}
    count = 0
    for df in pd.read_csv(PATH, usecols = inputs+output,
                          chunksize = NROWS):
        print('New chunk')
        count += 1
        df.dropna()
        for in_feat in inputs :
            for out_feat in output :
                s1 = df[in_feat]
                s2 = df[out_feat]
                file[(in_feat,out_feat)] += abs(s1.corr(s2))
    print('Done with this file!')    
    return file, count


# results for study i : to change : PATHS{i}
dicts = []
count = 0
for PATH in paths.path_dict[3] :
    print('\n New file')
    res, cnt = file(PATH)[0], file(PATH)[1]
    dicts.append(res)
    count += cnt
    
res = {key: sum([dic[key] for dic in dicts ])/count for key in keys}

print('\n RESULT FOR STUDY : \n', res)



''' 

 RESULT FOR STUDY 2 : 
 {('pressure', 'inlet_velocity [m/s]'): 0.4580796810502228, 
 ('velocity-magnitude', 'inlet_velocity [m/s]'): 0.01577132210489518, 
 ('acoustic-source-power-db', 'inlet_velocity [m/s]'): 0.028596328066400195, 
 
 ('pressure', 'coef_drag [-]'): 0.04510690478321687, 
 ('velocity-magnitude', 'coef_drag [-]'): 0.215499302056421, 
 ('acoustic-source-power-db', 'coef_drag [-]'): 0.11696950906765206, 
 
 ('pressure', 'coef_lift [-]'): 0.004531727024792427, 
 ('velocity-magnitude', 'coef_lift [-]'): 0.008468201396344363, 
 ('acoustic-source-power-db', 'coef_lift [-]'): 0.004898887051297535, 
 
 ('pressure', 'drag_force [N]'): 0.0451069036637124, 
 ('velocity-magnitude', 'drag_force [N]'): 0.21549930233743037, 
 ('acoustic-source-power-db', 'drag_force [N]'): 0.1169695096762789, 
 
 ('pressure', 'lift_force [N]'): 0.004531727072164346, 
 ('velocity-magnitude', 'lift_force [N]'): 0.008468201294971025, 
 ('acoustic-source-power-db', 'lift_force [N]'): 0.004898887067705152, 
 
 ('pressure', 'angle_attack [°]'): 0.003925993869511436, 
 ('velocity-magnitude', 'angle_attack [°]'): 0.012455319476385734, 
 ('acoustic-source-power-db', 'angle_attack [°]'): 0.006373328574278924}

''' 

relevant_sensors_correlation = {'inlet_velocity [m/s]':['pressure'],
                    'coef_drag [-]':['velocity-magnitude'],
                    'coef_lift [-]':['velocity-magnitude'],
                    'drag_force [N]':['velocity-magnitude'],
                    'lift_force [N]':['velocity-magnitude'],
                    'angle_attack [°]':['velocity-magnitude']
                    }