from keras.model import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np 
import pandas as pd
import paths 

#paths = []
#for std_num in [2,3]:
#    paths.append(paths.path_dict[std_num])

study_number = 2

X_train, Y_train, X_test, Y_test = None

data_dim = X_train.shape[]
timesteps = 
batch_size = 50  
EPOCHS = 100 

# input format expected : (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(data_dim, return_sequences = True, stateful = True,
    batch_input_shape = (batch_size, timesteps, data_dim)))
model.add(LSTM(data_dim, return_sequences = True, stateful = True))
model.add(LSTM(data_dim, return_sequences = True, stateful = True))

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, Y_train, batch_size = batch_size, epochs = EPOCHS, 
    shuffle = False, validation_split = 0.2) 




