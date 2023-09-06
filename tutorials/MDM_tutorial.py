# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:12:30 2019

@author: ivanovn
"""

# Create a simple graph for testing

import mindpype as mp
import numpy as np


def main():
    # create a session
    s = mp.Session.create()
    trial_graph = mp.Graph.create(s)
    # add a block and some tensors

    # initialize the classifier
    # fake data for training
    raw_training_data = np.random.normal(loc=0.0,scale=1.0,size=(180,250,12))
    training_data = np.zeros((180,12,12))
    for i in range(180):
        training_data[i,:,:] = np.cov(raw_training_data[i,:,:],rowvar=False)
        
    labels = np.asarray([0]*60 + [1]*60 + [2]*60)
    X = mp.Tensor.create_from_data(s,training_data.shape,training_data)
    y = mp.Tensor.create_from_data(s,labels.shape,labels)

    input_data = np.random.randn(12,500)
    t_in = mp.Tensor.create_from_data(s,(12,500),input_data)
    s_out = mp.Scalar.create_from_value(s,-1)
    t_virt = [mp.Tensor.create_virtual(s), 
              mp.Tensor.create_virtual(s)]
    
    # create a filter
    order = 4
    bandpass = (8,35) # in Hz
    fs = 250
    f = mp.Filter.create_butter(s,order,bandpass,btype='bandpass',fs=fs,implementation='sos')

    # add the nodes
    mp.kernels.CovarianceKernel.add_covariance_node(trial_graph,t_virt[0],t_virt[1])
    mp.kernels.FilterKernel.add_filter_node(trial_graph,t_in,f,t_virt[0])
    mp.kernels.RiemannMDMClassifierKernel.add_riemann_MDM_node(trial_graph,
                                                               t_virt[1],
                                                               s_out,X,y)

    # verify the session (i.e. schedule the nodes)
    trial_graph.verify()
    # intialize the graph (i.e. train the classifier)
    trial_graph.initialize()

    # RUN!
    trial_graph.execute(0)
    
    print(s_out.data)
    
    print("Test Passed =D")


if __name__ == "__main__":
    main()
