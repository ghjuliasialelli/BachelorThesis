# -*- coding: utf-8 -*-
"""
Test SHAP Values
"""

import pandas as pd
import numpy as np
import wavenet as net
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from sklearn.model_selection import train_test_split
import shap
import tcn as tcn

logging.basicConfig(level=logging.DEBUG)
np.random.seed(0)


cols = ['time [s]', 'dp_number', 'inlet_velocity [m/s]','rotational_speed [rad/s]',	
'coef_drag [-]'	,'coef_lift [-]'	,'drag_force [N]',	'lift_force [N]'	,
'angle_attack [Â°]','moving_drag_force_x [N]',	'moving_drag_force_y [N]'	,
'moving_lift_force_x [N]',	'moving_lift_force_y [N]','nodenumber','x-coordinate',
'y-coordinate','pressure','acoustic-source-power-db','velocity-magnitude',
'x-velocity','y-velocity','velocity-angle']

output = ['inlet_velocity [m/s]','coef_drag [-]','coef_lift [-]','drag_force [N]',	
       'lift_force [N]','angle_attack [°]']

def read_csv():
    df = pd.read_csv('naca_study2_training_data1.csv', skiprows= lambda x: x in [1, 163602], nrows=1000) # Load the data
    return df



def create_tensors(features_train,features_test,labels_train,labels_test):
    features_train = np.reshape(features_train, (int(features_train.shape[0] / 3000), -1, features_train.shape[1]))
    features_test = features_test.reshape((int(features_test.shape[0] / 3000), -1, features_test.shape[1]))
    labels_train = labels_train.reshape((int(labels_train.shape[0] / 3000), -1, labels_train.shape[1]))
    labels_test = labels_test.reshape((int(labels_test.shape[0] / 3000), -1, labels_test.shape[1]))
    
    train_TCN_input = torch.zeros(features_train.shape[0], 1, features_train.shape[1], features_train.shape[2])
    test_TCN_input = torch.zeros(features_test.shape[0], 1, features_test.shape[1], features_test.shape[2])
    train_target_tensor = torch.zeros(labels_train.shape[0], labels_train.shape[1], labels_train.shape[2])
    test_target_tensor = torch.zeros(labels_test.shape[0], labels_test.shape[1], labels_test.shape[2])
    
    for i in range(features_train.shape[0]):
        train_TCN_input[i, 0] = torch.from_numpy(features_train[i]).float()
        train_target_tensor[i] = torch.from_numpy(labels_train[i]).float()
    
    for i in range(features_test.shape[0]):
        test_TCN_input[i, 0] = torch.from_numpy(features_test[i]).float()
        test_target_tensor[i] = torch.from_numpy(labels_test[i]).float()
    
    print('Creating tensors... Done!')
    
    return train_TCN_input ,test_TCN_input ,train_target_tensor ,test_target_tensor 




def split(df):
    # output parameters
    Y = df[output]
    
    # input parameters
    X = df[['pressure', 'velocity-magnitude','acoustic-source-power-db']]
    
    # Split the data into train and test data:
    Xdf_train, Xdf_test, Ydf_train, Ydf_test = train_test_split(X, Y, test_size = 0.2)
    
    # Turn them into pytorch tensors
    X_train = torch.from_numpy(Xdf_train.values)
    X_test = torch.from_numpy(Xdf_test.values)
    Y_train = torch.from_numpy(Ydf_train.values)
    Y_test = torch.from_numpy(Ydf_test.values)
    
    #X_train,X_test,Y_train,Y_test = create_tensors(Xdf_train, Xdf_test, Ydf_train, Ydf_test)
    
    return X_train,X_test,Y_train,Y_test




def create_wavenet(X_train,Y_train):
    conv_size = 64
    model = net.Wavenet(conv_size=conv_size, nb_of_measurements=X_train.shape[1], 
                     nb_of_outputs=Y_train.shape[1])
    return model 

def train_wavenet(model,X_train,X_test,Y_train,Y_test):
    batch_size = 10
    n_epoch = 50
    model.reset_parameters()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.01)
    train_losses = []
    test_losses = []
    
    logging.debug('Training Start')
    for epoch in range(n_epoch):
        running_loss = 0
        for mini_batch in net.create_mini_batches(X_train,Y_train,batch_size):
            # zero the parameter gradients
            optimizer.zero_grad()
            # calculate outputs
            outputs = model(torch.from_numpy(mini_batch[0]))
            # calculate loss
            train_loss = loss_fn(outputs, torch.from_numpy(mini_batch[1]))
            # perform backpropagation
            train_loss.backward()
            # optimize
            optimizer.step()
    
            running_loss += train_loss.item()
    
        epoch_loss = running_loss / int(X_train.shape[0] / batch_size)
    
        with torch.no_grad():
            pred = model(X_test)
            test_loss = loss_fn(pred, Y_test)
    
        train_losses.append(epoch_loss)
        test_losses.append(test_loss.item())
    
        # logging.debug statistics
        logging.debug(f'epoch: {epoch}, train loss: {epoch_loss}, test loss: {test_loss.item()}')
    logging.debug('Training finished!')




def create_tcn(X_train,Y_train):
    hidden = 100
    out = Y_train.shape[0]
    in_dim = X_train.shape[1]
    model = tcn.TemporalConvNet(in_dim=in_dim, hid_dim=hidden, out_dim=out, time_len=X_train.shape[0])
    return model

def train_tcn(model,X_train,X_test,Y_train,Y_test):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.01)
    n_epoch = 50
    batch_size = 10
    train_losses = []
    test_losses = []

    print('Start training...')
    for epoch in range(n_epoch):
        train_loss = tcn.train(model, X_train, Y_train, loss_fn, optimizer, batch_size)
        train_losses.append(train_loss)
        test_loss = tcn.test(model, X_test, Y_test, loss_fn, batch_size)
        test_losses.append(test_loss)
        print('epoch:', epoch + 1, 'train_loss:', ', ', float(train_loss), 'test_loss:', float(test_loss))
    print('...end training')



def get_shap(model, X_train):
    exp = shap.DeepExplainer(model,X_train)
    shap_values = exp.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    shap.summary_plot(shap_values, X_train)
    
