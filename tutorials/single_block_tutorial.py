# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:37:58 2019

full block test

@author: ivanovn
"""
from bcipy import bcipy

from test.utils import load_training_data

from random import shuffle

def main():
    # create a session
    s = bcipy.Session.create()
    g = bcipy.Graph.create()

    # grab some data for training the classifier
    X_train, y_train = load_training_data()
        
    X = bcipy.Tensor.create_from_data(s,X_train.shape,X_train)
    y = bcipy.Tensor.create_from_data(s,y_train.shape,y_train)


    # create the volatile data source obj
    channels = (5,10,12,13,14,18,20,21,31,32,38,40,45,47,48,49,50,51,55,57,58)
    time_samples = tuple([i for i in range(500)])
    Ns = len(time_samples)
    Nc = len(channels)
    label_varname_map = {0 : 'class1_trials', 1 : 'class2_trials', 2 : 'class3_trials'}
    dims = (time_samples,channels)
    data_src = bcipy.source.BcipMatFile.create(s,'class_trials.mat','data/',label_varname_map,dims)
    
    # create the input data tensor
    eeg = bcipy.Tensor.create_from_handle(s, (Ns,Nc), data_src)

    # create virtual tensor (filtered data & covariance matrix)
    t_virt = [bcipy.Tensor.create_virtual(s), \
              bcipy.Tensor.create_virtual(s)]

    # create the output label
    label = bcipy.Scalar.create_from_value(s,-1)
    
    
    # create a filter object
    order = 4
    bandpass = (8,35) # in Hz
    fs = 250
    f = bcipy.Filter.create_butter(s,order,bandpass,btype='bandpass',fs=fs,implementation='sos')


    # add the nodes to the block
    bcipy.kernels.CovarianceKernel.add_covariance_node(g,t_virt[0],t_virt[1])
    
    bcipy.kernels.FiltFiltKernel.add_filtfilt_node(g,eeg,f,t_virt[0])
    
    bcipy.kernels.RiemannMDMClassifierKernel.add_untrained_riemann_MDM_node(g,
                                                              t_virt[1],
                                                              label,X,y)

    # verify the session (i.e. schedule the nodes)
    sts = s.verify()

    if sts != bcipy.BcipEnums.SUCCESS:
        print(sts)
        print("Test Failed D=")
        return sts

    
    trial_seq = [0]*4 + [1]*4 + [2]*4
    shuffle(trial_seq)
    
    # RUN!
    t_num = 0
    correct_labels = 0
    while t_num and sts == bcipy.BcipEnums.SUCCESS:
        print(f"t_num {t_num}, length of trials: {len(trial_seq)}")
        y = trial_seq[t_num]
        sts = g.execute()
        
        if sts == bcipy.BcipEnums.SUCCESS:
            # print the value of the most recent trial
            y_bar = label.data
            print("Trial {}: Label = {}, Predicted label = {}".format(t_num+1,y,y_bar))
            
            if y == y_bar:
                correct_labels += 1
        
        else:
            print(f"Trial {t_num+1} raised error, status code: {sts}")
            break

        t_num += 1
        
    print("Accuracy = {:.2f}%.".format(100 * correct_labels/len(trial_seq)))
    
    print("Test Passed =D")


if __name__ == "__main__":
    main()
