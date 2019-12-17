# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:37:58 2019

full block test

@author: ivanovn
"""

from classes.session import Session
from classes.tensor import Tensor
from classes.scalar import Scalar
from classes.filter import Filter
from classes.block import Block
from classes.source import BcipMatFile
from classes.bcip_enums import BcipEnums

from kernels.filtfilt import FiltFiltKernel
from kernels.covariance import CovarianceKernel
from kernels.riemann_mdm_classifier_kernel import RiemannMDMClassifierKernel

import numpy as np
from scipy.io import loadmat
from scipy import signal

from random import shuffle

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
        X_train[i-1,:,:] = np.cov(class1_data[-i,:,:],rowvar=False)
        X_train[i+59,:,:] = np.cov(class2_data[-i,:,:],rowvar=False)
        X_train[i+119,:,:] = np.cov(class3_data[-i,:,:],rowvar=False)
        
    y_train = np.asarray([0]*60 + [1]*60 + [2]*60)
    
    return X_train, y_train

def main():
    # create a session
    s = Session.create()

    # add a block and some tensors
    b = Block.create(s,3,(4,4,4))

    # initialize the classifier
    # grab some data for training
    X_train, y_train = load_training_data()
        
    X = Tensor.create_from_data(s,X_train.shape,X_train)
    y = Tensor.create_from_data(s,y_train.shape,y_train)


    # create the volatile data source obj
    channels = (5,10,12,13,14,18,20,21,31,32,38,40,45,47,48,49,50,51,55,57,58)
    time_samples = tuple([i for i in range(500)])
    Ns = len(time_samples)
    Nc = len(channels)
    label_varname_map = {0 : 'class1_trials', 1 : 'class2_trials', 2 : 'class3_trials'}
    dims = (time_samples,channels)
    data_src = BcipMatFile.create('class_trials.mat','data/',label_varname_map,dims)
    
    # create the input data tensor
    t_in = Tensor.create_from_handle(s, (Ns,Nc), data_src)

    # create virtual tensor (filtered data & covariance matrix)
    t_virt = [Tensor.create_virtual(s), \
              Tensor.create_virtual(s)]

    # create the output label
    s_out = Scalar.create_from_value(s,-1)
    
    
    # create a filter object
    order = 4
    bandpass = (8,35) # in Hz
    fs = 250
    f = Filter.create_butter(s,order,bandpass,btype='bandpass',fs=fs,implementation='sos')


    # add the nodes to the block
    CovarianceKernel.add_covariance_node(b.trial_processing_graph,t_virt[0],t_virt[1])
    FiltFiltKernel.add_filtfilt_node(b.trial_processing_graph,t_in,f,t_virt[0])
    RiemannMDMClassifierKernel.add_untrained_riemann_MDM_node(b.trial_processing_graph,
                                                              t_virt[1],
                                                              s_out,X,y)

    # verify the session (i.e. schedule the nodes)
    sts = s.verify()

    if sts != BcipEnums.SUCCESS:
        print(sts)
        print("Test Failed D=")
        return sts
    
    sts = s.start_block()
    if sts != BcipEnums.SUCCESS:
        print(sts)
        print("Test Failed D=")
        return sts
    
    trial_seq = [0]*4 + [1]*4 + [2]*4
    shuffle(trial_seq)
    
    # RUN!
    t_num = 1
    correct_labels = 0
    while s.remaining_blocks != 0 and sts == BcipEnums.SUCCESS:
        y = trial_seq[t_num-1]
        sts = s.execute_trial(y)
        
        if sts == BcipEnums.SUCCESS:
            # print the value of the most recent trial
            y_bar = s_out.data
            print("Trial {}: Label = {}, Predicted label = {}".format(t_num,y,y_bar))
            
            if y == y_bar:
                correct_labels += 1
        
        t_num += 1
        
        b = s.current_block
        if sum(b.remaining_trials()) == 0:
            # run the block closing graph
            s.close_block()
        
    print("Accuracy = {:.2f}%.".format(100 * correct_labels/len(trial_seq)))
    
    print("Test Passed =D")


if __name__ == "__main__":
    main()
