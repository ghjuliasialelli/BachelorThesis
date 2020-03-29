# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:02:26 2020

@author: Ghjulia Sialelli

TCN architecture implementation
"""

from keras.layers import Conv1D, Input, ReLU, Concatenate, Dense, MaxPooling1D
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from keras.models import Model
from keras import optimizers
import pandas as pd
import numpy as np
import os.path

#############################################################################################
inputs = 'pressure'
outputs = 'pressure'

######## 4P_SHAP ################################################
#rel_ids = [7223.5, 7221.497170000001, 7225.50265, 7227.5, 7231.5]
#rel_types = ['pressure','pressure','pressure','pressure']
#################################################################

##### 4P_RANDOM ##########################################
#rel_ids = [7435.5, 7243.5, 6613.49682, 7889.50247]      
#rel_types = ['pressure','pressure','pressure','pressure']
##########################################################

######## 100P ##############################################################################
#rel_ids = [7783.5    , 7229.50231, 8057.501690000001, 6961.497170000001, 6933.502579999999,
#           7467.5    , 7645.4986 , 6643.5    , 7291.5    , 7013.49951,
#           6759.5    , 6747.5    , 7559.5    , 7217.496279999999, 6509.48116,
#           6815.5    , 8101.50508, 6887.5    , 6581.50339, 8061.505090000001,
#           6875.5    , 6773.496940000001, 7391.5    , 6805.49917, 6913.4985400000005,
#           7099.5    , 7347.5    , 6711.5    , 7409.49568, 7359.5    ,
#           6883.5    , 7033.4993 , 7261.5018900000005, 8037.50267, 7067.5    ,
#           6583.5    , 7043.5    , 6747.5    , 7187.5    , 6903.5    ,
#           7645.4986 , 6963.5    , 7553.499379999999, 6553.50361, 7195.5    ,
#           7281.5021799999995, 6643.5    , 7531.5    , 7737.50252, 6633.49952,
#           7915.5    , 8041.50162, 6973.49915, 6845.5005200000005, 7553.499379999999,
#           7109.501740000001, 6949.5    , 6529.49715, 7529.5018 , 6869.499629999999,
#           7455.5    , 6671.5    , 6537.50187, 6963.5    , 6519.49998,
#           7627.5    , 8015.5    , 7717.49777, 6973.49915, 6903.5    ,
#           7103.5    , 6675.5    , 7317.501509999999, 7705.499640000001, 7327.49997,
#           7957.49608, 6989.5005599999995, 7057.4999099999995, 7629.5016 , 6703.5    ,
#           8111.5    , 7843.5    , 7483.5    , 8135.4992 , 7747.5    ,
#           7317.501509999999, 7099.5    , 6723.5    , 8021.501029999999, 7489.4994799999995,
#           7945.50285, 7007.5    , 7411.5    , 7151.5    , 7897.50247,
#           7623.5    , 7679.5    , 7315.49998, 7411.5    , 7035.5    ]
# rel_types = ['pressure' for _ in range(len(rel_ids))]
#############################################################################################

NNODES = len(rel_ids)                       # number of sensors to consider
NPOINTS = 100                               # number of points in a second 
LENFRAME = 12                               # length of a time frame to study 
NROWS = NNODES * (NPOINTS * LENFRAME + 1)   # number of rows to read for one dp
RATIO = 0.25                                # Ratio of testing to training set

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

NNODES = len(rel_ids)                       # number of sensors we consider
NPOINTS = 100                               # number of points in a second 
LENFRAME = 12                               # length of a time frame to study 
NROWS = NNODES * (NPOINTS * LENFRAME + 1)   # equivalent of one experiment (in rows), after preprocessing
RATIO = 0.25                                # Ratio of testing to training set
TOTROWS = 818 * (NPOINTS * LENFRAME + 1)    # number of rows for one dp

#############################################################################################
'''
Standardizing the dataset
'''
def standardize(X, std = None, mean = None):
    Xresh = X.reshape((X.shape[0]*X.shape[1],))
    if mean == None : mean = np.mean(Xresh)
    if std == None : std = np.std(Xresh)
    return (X-mean)/std, std, mean

'''
Post-processing the data to feed it to Wavenet
'''
def transform_dataset(PATHS):
    Xs = []
    Ys = []
    
    for PATH in PATHS :
        for df in pd.read_csv(PATH, chunksize = TOTROWS):             
            # Process dataframe
            df = df.dropna() 
            df[['time [s]']] = df[['time [s]']].astype('float64')
            df = df.drop(df[2.0 > df['time [s]']].index) 
            
            ######## modify this depending on the # of relevant nodes #############
            df = df[(df['cell-id'] == rel_ids[0]) | (df['cell-id'] == rel_ids[1]) | 
                    (df['cell-id'] == rel_ids[2]) | (df['cell-id'] == rel_ids[3]) ]
            #######################################################################
            
            for time in df['time [s]'].unique() :
                dftime = df[df['time [s]'] == time]
                
                df1 = dftime[dftime['cell-id'] == rel_ids[0]][[rel_types[0]]].T.values
                df2 = dftime[dftime['cell-id'] == rel_ids[1]][[rel_types[1]]].T.values
                df3 = dftime[dftime['cell-id'] == rel_ids[2]][[rel_types[2]]].T.values
                df4 = dftime[dftime['cell-id'] == rel_ids[3]][[rel_types[3]]].T.values
                
                X = np.concatenate((df1,df2,df3,df4), axis = 1)[0]
                Xs.append(X)
                Y = dftime[[outputs]].T.values[0][:1]
                Ys.append(Y)
                    
    return np.array(Xs), np.array(Ys)


#############################################################################################
KERNEL_SIZE = 5
N_FILTERS = 10
EPOCHS = 50
BS = NNODES
LR = 0.0001
#############################################################################################

class TCN():
    '''
    Parameters :
        input_shape : tuple of input shape
        output_shape : tuple of output shape
        kernel_size : size of convolutional operations in residual blocks
    '''
    def __init__(self, input_shape, output_shape, batch_size = BS, epochs = EPOCHS, 
                 kernel_size = KERNEL_SIZE, n_filters = N_FILTERS):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.regularizer = l2(LR)
        self.history = None
        self.model = self.model()
    
    
    '''
    Construction of the model.
    '''
    def model(self):
        x = Input(shape = self.input_shape)
        
        conv11 = Conv1D(self.n_filters, self.kernel_size, kernel_regularizer = self.regularizer, padding='same')(x)
        relu11 = ReLU()(conv11)
        
        pooling11 = MaxPooling1D(pool_size = 2, strides = 2, padding='same')(relu11)
        conv12 = Conv1D(self.n_filters, self.kernel_size, kernel_regularizer = self.regularizer, padding='same')(pooling11)
        relu12 = ReLU()(conv12)
        fc12 = Dense(self.n_filters, kernel_regularizer = self.regularizer)(relu12)
        
        conv21 = Conv1D(self.n_filters, self.kernel_size, kernel_regularizer = self.regularizer, padding='same')(relu11) 
        relu21 = ReLU()(conv21)
        
        pooling21 = MaxPooling1D(pool_size = 2, strides = 2, padding='same')(relu21)
        conv22 = Conv1D(self.n_filters, self.kernel_size, kernel_regularizer = self.regularizer, padding='same')(pooling21)
        relu22 = ReLU()(conv22)
        fc22 = Dense(self.n_filters, kernel_regularizer = self.regularizer)(relu22)
        
        conv31 = Conv1D(self.n_filters, self.kernel_size, kernel_regularizer = self.regularizer, padding='same')(relu21)
        relu31 = ReLU()(conv31)
        fc31 = Dense(self.n_filters, kernel_regularizer = self.regularizer)(relu31)
    
        concat = Concatenate()([fc12, fc22, fc31])
        relu4 = ReLU()(concat)
        fc4 = Dense(self.output_shape[1], kernel_regularizer = self.regularizer)(relu4)
        
        model = Model(inputs=[x], outputs=[fc4])
        model.summary()
        
        return model
        

    '''
    Evaluating the model after training on the testing set.
    Returns : scalar test loss
    '''
    def evaluate(self, X, Y):
        return self.model.evaluate(X, Y, batch_size = self.batch_size)
    
#############################################################################################
print('Transforming / Loading dataset........')
if os.path.isfile('X_tcn.npy') and os.path.isfile('Y_tcn.npy') : 
    X = np.load('X_tcn.npy', allow_pickle='TRUE')
    Y = np.load('Y_tcn.npy', allow_pickle='TRUE')
else:    
    X, Y = transform_dataset(PATHS)
    np.save('X_tcn.npy', X)
    np.save('Y_tcn.npy', Y)
print('Done!')

Xtr,Xte,Ytr,Yte = train_test_split(X,Y, test_size = RATIO, shuffle = True)

Xtr, Xtr_std, Xtr_mean = standardize(Xtr)
Ytr, Ytr_std, Ytr_mean = standardize(Ytr)
Xte, _, _              = standardize(Xte, Xtr_std, Xtr_mean)
Yte, _, _              = standardize(Yte, Ytr_std, Ytr_mean)

np.savetxt('dataset_info.out', np.asarray([Xtr_std, Xtr_mean, Ytr_std, Ytr_mean]), delimiter=',')

Xtr = Xtr.reshape((Xtr.shape[0], 1, 4))
Ytr = Ytr.reshape((Ytr.shape[0], 1, 1))
Xte = Xte.reshape((Xte.shape[0], 1, 4))
Yte = Yte.reshape((Yte.shape[0], 1, 1))

print('Xtr shape : ', Xtr.shape)
print('Xte shape : ', Xte.shape)
print('Ytr shape : ', Ytr.shape)
print('Yte shape : ', Yte.shape)

tcn = TCN((1,4),(1,1))
adam = optimizers.Adam(lr=0.00005, beta_1 = 0.99, beta_2 = 0.999, amsgrad = False)
tcn.model.compile(adam, loss = 'mean_squared_error', metrics = None)
tcn.history = tcn.model.fit(Xtr, Ytr, batch_size = tcn.batch_size, epochs = tcn.epochs, validation_split = 0.2, shuffle = True)

MSE_train = tcn.history.history['loss']
MSE_validation = tcn.history.history['val_loss']


# Saving the losses to files
np.savetxt('MSE_train_tcn.out', np.asarray(MSE_train), delimiter=',')
np.savetxt('MSE_val_tcn.out', np.asarray(MSE_validation), delimiter=',')

print('Writing the model to file')
tcn.model.save('model_tcn.h5') 
