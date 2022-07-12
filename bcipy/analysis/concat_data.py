# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 11:45:16 2020

@author: Nick
"""


import numpy as np
from glob import glob
from scipy.io import loadmat, savemat

if __name__ == "__main__":
    
    
    dataset = 'High'
    
    Fbands =      ((4,38),
                   (4,10),
                   (7,13),
                   (10,18),
                   (14,22),
                   (18,26),
                   (22,30),
                   (26,34),
                   (30,38),
                   (8,10),
                   (10,12),
                   (12,18),
                   (18,30),
                   (8,30))
    
    for i_p in range(1,10):
        print("Concatenating data for participant {}".format(i_p))
        
        train_data = loadmat("cropped-A0{}T.mat".format(i_p))
        test_data = loadmat("cropped-A0{}E.mat".format(i_p))
        
        Nclasses = 4
        Ntrials = {}
        data = {'num_trials' : Ntrials,
                'num_classes' : Nclasses,
                'freq_bands' : Fbands}
        for i_t in range(1,Nclasses+1):
            class_train = train_data['class{}_trials'.format(i_t)]
            class_test = test_data['class{}_trials'.format(i_t)]
            
            Ntrials[i_t] = class_train[0] + class_test[0]
            
            data['class1_trials'.format(i_t)] = np.concatenate((class_train,
                                                                class_test),
                                                               axis=0)
        
        savemat('cropped-A0{}.mat'.format(i_p),data)
        