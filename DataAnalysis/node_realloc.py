# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:41:01 2020

@author: Ghjulia Sialelli
"""

import pandas as pd
from math import sqrt, atan2, pi, degrees
import numpy as np
import paths
from sklearn.metrics import pairwise_distances

NNODES = 818        # number of nodes for one dp
NPOINTS = 100       # number of points in a minute 
LENFRAME = 12       # length of a time frame to study (drop first 2 seconds)
NROWS = NNODES * (NPOINTS * LENFRAME + 1)   # number of rows to read for one dp


ORDER = [410,408,407,406,405,404,402,401,400,399,398,397,395,394,393,392,391,390,
         388,387,386,385,384,382,381,380,379,378,377,375,374,373,372,371,369,368,
         367,366,365,363,362,361,360,359,357,356,355,354,353,351,350,349,348,347,
         345,344,343,342,340,339,338,337,335,334,333,332,331,329,328,327,326,324,
         323,322,321,319,318,317,316,314,313,312,310,309,308,307,305,304,303,301,
         300,299,298,296,295,294,292,291,290,289,287,286,285,283,282,281,279,278,
         277,275,274,273,271,270,269,267,266,265,263,262,261,259,258,256,255,254,
         252,251,250,248,247,245,244,243,241,240,239,237,236,234,233,231,230,229,
         227,226,224,223,222,220,219,217,216,214,213,211,210,209,207,206,204,203,
         201,200,198,197,195,194,192,191,189,188,186,185,183,182,180,179,177,176,
         174,173,171,170,168,167,165,164,162,161,159,157,156,154,153,151,150,148,
         146,145,143,142,140,138,137,135,134,132,130,129,127,126,124,122,121,119,
         117,116,114,112,111,109,107,106,104,102,101,99,97,95,94,92,90,89,87,85,
         83,82,80,78,76,75,73,71,69,67,66,64,62,60,58,57,55,53,51,49,48,46,44,42,
         40,38,36,34,33,31,29,27,25,23,21,19,17,15,13,11,9,7,6,4,2,1,3,5,8,10,12,
         14,16,18,20,22,24,26,28,30,32,35,37,39,41,43,45,47,50,52,54,56,59,61,63,
         65,68,70,72,74,77,79,81,84,86,88,91,93,96,98,100,103,105,108,110,113,115,
         118,120,123,125,128,131,133,136,139,141,144,147,149,152,155,158,160,163,
         166,169,172,175,178,181,184,187,190,193,196,199,202,205,208,212,215,218,221,
         225,228,232,235,238,242,246,249,253,257,260,264,268,272,276,280,284,288,
         293,297,302,306,311,315,320,325,330,336,341,346,352,358,364,370,376,383,389,
         396,403,409,416,423,429,436,442,449,455,461,467,473,478,483,489,494,499,503,
         508,513,517,522,526,530,535,539,543,547,551,555,559,562,566,570,573,577,
         580,584,587,591,594,598,601,604,607,611,614,617,620,623,626,629,632,635,
         638,641,644,647,650,653,656,659,661,664,667,670,672,675,678,680,683,686,
         688,691,693,696,699,701,704,706,709,711,714,716,719,721,723,726,728,730,
         733,735,738,740,742,745,747,749,751,754,756,758,760,763,765,767,769,772,
         774,776,778,780,782,784,787,789,791,793,795,797,799,801,803,805,807,809,
         811,813,815,817,818,816,814,812,810,808,806,804,802,800,798,796,794,792,
         790,788,786,785,783,781,779,777,775,773,771,770,768,766,764,762,761,759,
         757,755,753,752,750,748,746,744,743,741,739,737,736,734,732,731,729,727,
         725,724,722,720,718,717,715,713,712,710,708,707,705,703,702,700,698,697,
         695,694,692,690,689,687,685,684,682,681,679,677,676,674,673,671,669,668,
         666,665,663,662,660,658,657,655,654,652,651,649,648,646,645,643,642,640,
         639,637,636,634,633,631,630,628,627,625,624,622,621,619,618,616,615,613,
         612,610,609,608,606,605,603,602,600,599,597,596,595,593,592,590,589,588,
         586,585,583,582,581,579,578,576,575,574,572,571,569,568,567,565,564,563,
         561,560,558,557,556,554,553,552,550,549,548,546,545,544,542,541,540,538,
         537,536,534,533,532,531,529,528,527,525,524,523,521,520,519,518,516,515,514,
         512,511,510,509,507,506,505,504,502,501,500,498,497,496,495,493,492,491,
         490,488,487,486,485,484,482,481,480,479,477,476,475,474,472,471,470,469,
         468,466,465,464,463,462,460,459,458,457,456,454,453,452,451,450,448,447,
         446,445,444,443,441,440,439,438,437,435,434,433,432,431,430,428,427,426,
         425,424,422,421,420,419,418,417,415,414,413,412,411]

CR = (0.1,0)        # coordinates of the center of rotation 


'''
Function to obtain the distances between the center of rotation (RC) and each
individual nodes, given their (x,y) coordinates. Distance computed using 
scipy.spatial Euclidean distance. 

input : 
    nodes, an array of length #nodes with the coordinates of each node
    
output :
    dist, an array of length #nodes with the distance for each node
'''
def nodes_CR_dist(nodes):
    dist = np.diag(pairwise_distances(nodes, [CR for _ in range(len(nodes))], 'euclidean'))
    return dist
    


'''
Given an array of coordinates, return an array where each entry is the angle 
between (tail, CR) and (current point, CR).

input :
    tail, coordinates of the tail point
    nodes, an array of length #nodes with the coordinates of each node

output :
    angles, an array of length #nodes with the angles of rotation
'''
def angles_nodes(tail,nodes):
    angles = []
    for p in nodes : 
        # Angle between the 3 points tail, CR, and p 
        angles.append(degrees(atan2(p[1]-CR[1], p[0]-CR[0]) - atan2(tail[1]-CR[1], tail[0]-CR[0])))
    return angles 


def create_base():
    # We load the data for the first 818 nodes, that we consider to be our base
    # data, since the angle of attack is 0
    base_df = pd.read_csv('naca_study2_training_data1.csv', nrows = 819)
    
    
    # Isolate x and y coordinates, and remove columns names
    Xs = np.asarray(base_df.iloc[:]['x-coordinate'])
    Ys = np.asarray(base_df.iloc[:]['y-coordinate'])
    coords = np.stack((Xs,Ys), axis=1).tolist()
    
    '''
    # Get the distances between each node and the center of rotation CR
    dist = nodes_CR_dist(coords)
    
    # Find the coordinates of the node for which the distance is maximal, and 
    # which has greatest x coordinates (corresponds to the tail of the airfoil)
    maximum_indices = np.where(dist == max(dist))[0]
    tail = max([coords[ind] for ind in maximum_indices], key = lambda t : t[0] )
    
    # What is the index of the tail in coords ?
    tail_idx = coords.index(tail)
    
    # Find the angles needed to go from one node to the next
    angles = angles_nodes(coords[tail_idx:] + coords[:tail_idx])
    '''
    
    # Find the coordinates of the tail 
    tail = (0.2,0)
    ordered_coords = [coords[i-1] for i in ORDER]
    
    angles = angles_nodes(tail,ordered_coords)
    np.save('transform_data/base_angles.npy', angles)
    
    return 
    


def redo(df, angle):
    Xs = np.asarray(df['x-coordinate'])
    Ys = np.asarray(df['y-coordinate'])
    coords = np.stack((Xs,Ys), axis=1).tolist()
    
    # Get the distances between each node and the center of rotation CR
    dist = nodes_CR_dist(coords)
    
    # Find the coordinates of the node for which the distance is maximal, and 
    # which has greatest x coordinates (corresponds to the tail of the airfoil)
    maximum_indices = np.where(dist == max(dist))[0]
    tail = max([coords[ind] for ind in maximum_indices], key = lambda t : t[0] )
    
    # What is the index of the tail in coords ?
    tail_idx = coords.index(tail)
    
    
    










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
        ??? 
        
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