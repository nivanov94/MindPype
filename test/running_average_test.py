# -*- coding: utf-8 -*-
"""
Created on Tues July 26 16:12:30 2022

@author: aaronlio
"""
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
    trial_graph_2 = bcipy.Graph.create(session)
    trial_graph_3 = bcipy.Graph.create(session)

    #data
    training_data = np.random.random((120,12,500))

    init_data = sio.loadmat('test_data\init_data.mat')['init_data']
    init_labels = sio.loadmat('test_data\init_labels.mat')['labels']
    

    X = bcipy.Tensor.create_from_data(session,np.shape(init_data), init_data)
    y = bcipy.Tensor.create_from_data(session,np.shape(init_labels),init_labels)


    input_data = bcipy.source.BcipClassSeparatedMat.create_class_separated(session, 2, 500, 0, 4000, 0, 'input_data', 'input_labels', 'test_data\input_data.mat', 'test_data\input_labels.mat')
    input_data = bcipy.Tensor.create_from_handle(session, (12, 500), input_data)
    
    #t_in = Tensor.create_from_data(session,(12,500),input_data)
    t_out = bcipy.Tensor.create_virtual(session)
    
    t_virt = [bcipy.Tensor.create_virtual(session), \
              bcipy.Tensor.create_virtual(session)]

    t_virt2 = [bcipy.Tensor.create_virtual(session), \
              bcipy.Tensor.create_virtual(session)]

    class1avg = bcipy.Tensor.create_virtual(session)
    class2avg = bcipy.Tensor.create_virtual(session)


    # create a filter
    order = 4
    bandpass = (8,35) # in Hz
    fs = 250
    f = bcipy.Filter.create_butter(session,order,bandpass,btype='bandpass',fs=fs,implementation='sos')
    
    # add the nodes
    bcipy.kernels.FilterKernel.add_filter_node(trial_graph,input_data,f,t_virt[0])
    bcipy.kernels.CovarianceKernel.add_covariance_node(trial_graph, t_virt[0], t_virt[1], 0)
    bcipy.kernels.RunningAverageKernel.add_running_average_node(trial_graph, t_virt[1], class1avg, 5, 0)
    bcipy.kernels.FilterKernel.add_filter_node(trial_graph_2,input_data,f,t_virt2[0])
    bcipy.kernels.CovarianceKernel.add_covariance_node(trial_graph_2, t_virt2[0], t_virt2[1], 0)
    bcipy.kernels.RunningAverageKernel.add_running_average_node(trial_graph_2, t_virt2[1], class2avg, 5, 0)
    bcipy.kernels.RiemannDistanceKernel.add_riemann_distance_node(trial_graph_3, class1avg, class2avg, t_out)

    # verify the session (i.e. schedule the nodes)
    verify = session.verify()

    if verify != bcipy.BcipEnums.SUCCESS:
        print(verify)
        print("Test Failed D=")
        return verify
    
    start = trial_graph.initialize()

    if start != bcipy.BcipEnums.SUCCESS:
        print(start)
        print("Test Failed D=")
        return start
    
    start = trial_graph_2.initialize()

    if start != bcipy.BcipEnums.SUCCESS:
        print(start)
        print("Test Failed D=")
        return start

    start = trial_graph_3.initialize()
    if start != bcipy.BcipEnums.SUCCESS:
        print(start)
        print("Test Failed D=")
        return start

    # RUN!
    trial_seq = [0]*4 + [1]*4
    
    shuffle(trial_seq)

    t_num = 0

    sts = bcipy.BcipEnums.SUCCESS
    
    while t_num < 8 and sts == bcipy.BcipEnums.SUCCESS:
        print(f"t_num {t_num+1}, length of trials: {len(trial_seq)}")
        y = trial_seq[t_num]
        if y == 0:
            sts = trial_graph.execute(y)
        else:
            sts = trial_graph_2.execute(y)

        t_num += 1
        
    sts = trial_graph_3.execute(0, False)

    print(t_out.data)

    if sts != bcipy.BcipEnums.SUCCESS:
        print(sts)
        print("Test Failed D=")
        return sts

    print("Test Passed =D")
    return bcipy.BcipEnums.SUCCESS
if __name__ == "__main__":
    main()
