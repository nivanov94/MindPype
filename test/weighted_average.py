# -*- coding: utf-8 -*-
"""
Created on Tues July 26 16:12:30 2022

@author: aaronlio
"""
# This test is broken, do not reference
# TODO: Design this before building

# For debugging 
import sys, os
sys.path.insert(0, os.getcwd())

# Create a simple graph for testing
from bcipy import bcipy
import numpy as np
import scipy.io as sio
from random import shuffle

def main():
    # create a session
    session = bcipy.Session.create()
    trial_graph = bcipy.Graph.create(session)
    trial_graph_1b = bcipy.Graph.create(session)
    trial_graph_2 = bcipy.Graph.create(session)

    #data
    training_data = np.random.random((120,12,500))

    init_data = sio.loadmat('test_data\init_data.mat')['init_data']
    init_labels = sio.loadmat('test_data\init_labels.mat')['labels']
    

    X = bcipy.Tensor.create_from_data(session,np.shape(init_data), init_data)
    y = bcipy.Tensor.create_from_data(session,np.shape(init_labels),init_labels)


    input_data = bcipy.source.BcipContinuousMat.create_continuous(session, 2, 500, 0, 4000, 0, 'input_data', 'input_labels', 'test_data\input_data.mat', 'test_data\input_labels.mat')

    input_data = bcipy.Tensor.create_from_handle(session, (12, 500), input_data)
    
    #t_in = Tensor.create_from_data(session,(12,500),input_data)
    s_out = bcipy.Scalar.create_from_value(session,-1)
    t_virt = [bcipy.Tensor.create_virtual(session), \
              bcipy.Tensor.create_virtual(session)]


    t_virt_1b = [bcipy.Tensor.create_virtual(session), \
              bcipy.Tensor.create_virtual(session)
              ]
    
    t_out_1a = bcipy.Tensor.create(session, (4,12,12))
    t_out_1b = bcipy.Tensor.create(session, (4,12,12))
    # create a filter
    order = 4
    bandpass = (8,35) # in Hz
    fs = 250
    f = bcipy.Filter.create_butter(session,order,bandpass,btype='bandpass',fs=fs,implementation='sos')
    
    # add the nodes
    

    bcipy.kernels.FilterKernel.add_filter_node(trial_graph,input_data,f,t_virt[0])
    bcipy.kernels.CovarianceKernel.add_covariance_node(trial_graph, t_virt[0], t_virt[1], 0)
    bcipy.kernels.RiemannMeanKernel.add_riemann_mean_node(trial_graph, t_virt[1], t_out_1a)


    bcipy.kernels.FilterKernel.add_filter_node(trial_graph_1b,input_data,f,t_virt_1b[0])
    bcipy.kernels.CovarianceKernel.add_covariance_node(trial_graph_1b, t_virt_1b[0], t_virt_1b[1], 0)
    bcipy.kernels.RiemannMeanKernel.add_riemann_mean_node(trial_graph_1b, t_virt_1b[1], t_out_1b)
    
    bcipy.kernels.RiemannDistanceKernel.add_riemann_distance_node(trial_graph_2, t_out_1a, t_out_1b, s_out)

    # verify the session (i.e. schedule the nodes)
    verify = session.verify()

    if verify != bcipy.BcipEnums.SUCCESS:
        print(verify)
        print("Test Failed D=")
        return verify
    
    
    # RUN!
    trial_seq = [0]*4 + [1]*4
    
    shuffle(trial_seq)

    t_num = 0
    sts = bcipy.BcipEnums.SUCCESS
    correct_labels = 0

    
    while t_num < 8 and sts == bcipy.BcipEnums.SUCCESS:
        print(f"t_num {t_num}, length of trials: {len(trial_seq)}")
        y = trial_seq[t_num]
        sts = session.execute_trial(y, trial_graph)
        if sts == bcipy.BcipEnums.SUCCESS:
            # print the value of the most recent trial
            y_bar = s_out.data
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
