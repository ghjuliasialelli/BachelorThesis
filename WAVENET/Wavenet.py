# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 09:13:55 2020

@author: Ghjulia Sialelli

Wavenet architecture implementation

"""

from keras.layers import Conv1D, Multiply, Add, Input, Activation, TimeDistributed
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras import optimizers
import pandas as pd
import numpy as np
import os.path
import sys

# 0 or 1
index = int(sys.argv[1])

corres = ['drag_force [N]','lift_force [N]', 'angle_attack [°]', 'inlet_velocity [m/s]']

outputs = corres[index]

#############################################################################################
bij_type = {'inlet_velocity [m/s]' : 'velocity-magnitude',
             'drag_force [N]' : 'velocity-magnitude',	
             'lift_force [N]' : 'pressure',
             'angle_attack [°]' : 'acoustic-source-power-db'}

inputs = bij_type[outputs]

##### 4P_RANDOM ##########################################
#rel_ids = [7435.5, 7243.5, 6613.49682, 7889.50247]      
#rel_types = ['pressure','pressure','pressure','pressure']
##########################################################

### 2P2V #########################################################################
#rel_ids = [7347.5, 7301.49452, 6527.5, 7327.49997]    
#rel_types = ['pressure', 'pressure', 'velocity-magnitude', 'velocity-magnitude']
##################################################################################

#####  4P_SHAP  #########################################
#rel_ids = [7347.5, 7301.49452, 7295.5, 7349.4958]
#rel_types = ['pressure','pressure','pressure','pressure']
# all pressure ##########################################

######## 2P1V1N ####################################################################
#rel_ids = [7347.5, 7301.49452, 6527.5, 7315.49998]    
#rel_types = ['pressure','pressure','velocity-magnitude','acoustic-source-power-db']
####################################################################################

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
#rel_types = ['pressure' for _ in range(len(rel_ids))]
#############################################################################################

NNODES = len(rel_ids)                       # number of sensors we consider
NPOINTS = 100                               # number of points in a second 
LENFRAME = 12                               # length of a time frame to study 
NROWS = NNODES * (NPOINTS * LENFRAME + 1)   # equivalent of one experiment (in rows), after preprocessing
RATIO = 0.25                                # Ratio of testing to training set
TOTROWS = 818 * (NPOINTS * LENFRAME + 1)    # number of rows for one dp

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
            #df = df.dropna() 
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
KERNEL_SIZE = 2
DILATION_DEPTH = 5
N_FILTERS = 32
ACTIVATION_FUN = 'linear'
EPOCHS = 5
BS = 8                             
#############################################################################################

class WaveNetRegressor():
    '''
    Parameters :
        input_shape : tuple of input shape
        output_shape : tuple of output shape
        kernel_size : size of convolutional operations in residual blocks
        dilation_depth : depth of residual blocks
        activation : activation function used 
    '''
    def __init__(self, input_shape, output_shape, batch_size = BS, 
                 epochs = EPOCHS, kernel_size = KERNEL_SIZE, 
                 dilation_depth = DILATION_DEPTH, n_filters = N_FILTERS,
                 activation = ACTIVATION_FUN):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.kernel_size = kernel_size
        self.dilation_depth = dilation_depth
        self.n_filters = n_filters
        self.activation = activation
        self.history = None
        self.model = self.model()
    
    '''
    Implementation of the Residual and Skip connections.
    '''
    def skip_connections(self, x, i):
        tanh = Conv1D(self.n_filters, self.kernel_size, dilation_rate = self.kernel_size ** i,
                      padding = 'causal', activation = 'tanh')(x)
        sigmoid = Conv1D(self.n_filters, self.kernel_size, dilation_rate = self.kernel_size ** i,
                         padding = 'causal', activation = 'sigmoid')(x)
        z = Multiply()([tanh, sigmoid])
        skip = Conv1D(self.n_filters, kernel_size = 1)(z)
        out = Add()([skip,x])
        return out, skip

    '''
    Construction of the model.
    '''
    def model(self):
        x = Input(shape = self.input_shape)
        out = Conv1D(self.n_filters, kernel_size = 2, padding = 'causal')(x)
   
        skip_connections = []
        for i in range(self.dilation_depth):
            out, skip = self.skip_connections(out,i)
            skip_connections.append(skip)
        out = Add()(skip_connections)
   
        out = Activation('relu')(out)
        out = Conv1D(self.n_filters, kernel_size = 2, padding = 'same', activation = 'relu')(out)

        out = Conv1D(self.n_filters, self.output_shape[0], activation='relu')(out)
        out = Conv1D(self.output_shape[1], self.output_shape[0])(out)

        out = TimeDistributed(Activation(self.activation))(out)
        
        model = Model(inputs=x, outputs=out)
        model.summary()

        return model

    
    '''
    Evaluating the model.
    output : scalar test loss 
    '''
    def evaluate(self, X, Y):
        return self.model.evaluate(X, Y, batch_size = self.batch_size)


#############################################################################################
print('Transforming / Loading dataset........')
if os.path.isfile('X_{}.npy'.format(outputs[:-6])) and os.path.isfile('Y_{}.npy'.format(outputs[:-6])) : 
    X = np.load('X_{}.npy'.format(outputs[:-6]),allow_pickle='TRUE')
    Y = np.load('Y_{}.npy'.format(outputs[:-6]),allow_pickle='TRUE')
else:    
    X, Y = transform_dataset(PATHS)
    np.save('X_{}.npy'.format(outputs[:-6]), X)
    np.save('Y_{}.npy'.format(outputs[:-6]), Y)
print('Done!')

print('X shape : ', X.shape)
print('Y shape : ', Y.shape)

#indices = []
#for i,x in enumerate(X):
#    if len(x) != 4 : indices.append(i)

#X = np.delete(X,indices)
#X = np.stack(X)
#Y = np.delete(Y,indices)
#Y = Y.reshape((Y.shape[0],1))

#print('after this')
#print('X shape : ', X.shape)
#print('Y shape : ', Y.shape)

Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size = RATIO, shuffle = True)


print('Xtr shape : ', Xtr.shape)
print('Xte shape : ', Xte.shape)
print('Ytr shape : ', Ytr.shape)
print('Yte shape : ', Yte.shape)

# Now, we standardize the data. We obtain the std and mean of the training data, and apply it to the testing data
Xtr, Xtr_std, Xtr_mean = standardize(Xtr)
Ytr, Ytr_std, Ytr_mean = standardize(Ytr)
Xte, _, _              = standardize(Xte, Xtr_std, Xtr_mean)
Yte, _, _              = standardize(Yte, Ytr_std, Ytr_mean)


Xtr = Xtr.reshape((Xtr.shape[0], 1, 4))
Ytr = Ytr.reshape((Ytr.shape[0], 1, 1))
Xte = Xte.reshape((Xte.shape[0], 1, 4))
Yte = Yte.reshape((Yte.shape[0], 1, 1))

print('Xtr shape : ', Xtr.shape)
print('Xte shape : ', Xte.shape)
print('Ytr shape : ', Ytr.shape)
print('Yte shape : ', Yte.shape)

wn = WaveNetRegressor((1,4),(1,1))
adam = optimizers.Adam(lr = 0.00075, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
wn.model.compile(adam, loss = 'mean_squared_error', metrics = None)
wn.history = wn.model.fit(Xtr, Ytr, batch_size = wn.batch_size, epochs = wn.epochs, validation_split = 0.2, shuffle = True)

MSE_train = wn.history.history['loss']
MSE_validation = wn.history.history['val_loss']

# Saving the losses to files
np.savetxt('MSE_train_wn_{}.out'.format(outputs[:-6]), np.asarray(MSE_train), delimiter=',')
np.savetxt('MSE_val_wn_{}.out'.format(outputs[:-6]), np.asarray(MSE_validation), delimiter=',')
np.savetxt('dataset_info_{}.out'.format(outputs[:-6]), np.asarray([Xtr_std, Xtr_mean, Ytr_std, Ytr_mean]), delimiter=',')

print('Writing the model to file')
wn.model.save('model_wn_{}.h5'.format(outputs[:-6]))
