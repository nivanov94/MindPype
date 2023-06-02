# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:54:43 2020

@author: Nick
"""

import numpy as np
from scipy.io import loadmat
from scipy import signal

def load_training_data():
    class1_data = loadmat('data/class1_trials.mat')['class1_trials']
    class2_data = loadmat('data/class2_trials.mat')['class2_trials']
    class3_data = loadmat('data/class3_trials.mat')['class3_trials']
    
    channels = (5,10,12,13,14,18,20,21,31,32,38,40,45,47,48,49,50,51,55,57,58)
    Nc = len(channels)
    
    class1_data = class1_data[:,:,channels]
    class2_data = class2_data[:,:,channels]
    class3_data = class3_data[:,:,channels]
    
    # filter the data
    sos = signal.butter(4,(8,35),btype='bandpass',output='sos',fs=250)
    
    class1_data = signal.sosfiltfilt(sos,class1_data,axis=1)
    class2_data = signal.sosfiltfilt(sos,class2_data,axis=1)
    class3_data = signal.sosfiltfilt(sos,class3_data,axis=1)
    
    # use the last sixty trials of each for training
    X_train = np.zeros((180,Nc,Nc))
    for i in range(1,61):
        X_train[i-1,:,:] = np.cov(class1_data[i,:,:],rowvar=False)
        X_train[i+59,:,:] = np.cov(class2_data[i,:,:],rowvar=False)
        X_train[i+119,:,:] = np.cov(class3_data[i,:,:],rowvar=False)
        
    y_train = np.asarray([0]*60 + [1]*60 + [2]*60)
    
    return X_train, y_train