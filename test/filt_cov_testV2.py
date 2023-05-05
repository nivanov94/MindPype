# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:30:09 2019

@author: ivanovn
"""

# Create a simple graph for testing
import sys, os
sys.path.insert(0, os.getcwd())

from bcipy import bcipy

import numpy as np
from scipy import signal

def manual_computation(input_data):
    
    # first filter the data
    sos = signal.butter(4,(8,35),btype='bandpass',output='sos',fs=250)
    filtered_data = signal.sosfilt(sos,input_data,axis=0)
    cov_mat = np.cov(filtered_data)
    
    return cov_mat

def main():
    # create a session
    s = bcipy.Session.create()

    trial_graph = bcipy.Graph.create(s)

    input_data = np.random.randn(12,500)
    t_in = bcipy.Tensor.create_from_data(s,(12,500),input_data)
    t_out = bcipy.Tensor.create(s,(12,12))
    t_virt = bcipy.Tensor.create_virtual(s)

    # create a filter
    order = 4
    bandpass = (8,35) # in Hz
    fs = 250
    f = bcipy.Filter.create_butter(s,order,bandpass,btype='bandpass',fs=fs,implementation='sos')

    # add the nodes
    bcipy.kernels.FilterKernel.add_filter_node(trial_graph,t_in,f,t_virt)
    bcipy.kernels.CovarianceKernel.add_covariance_node(trial_graph,t_virt,t_out)
    

    # verify the session (i.e. schedule the nodes)
    sts = trial_graph.verify()

    if sts != bcipy.BcipEnums.SUCCESS:
        print(sts)
        return sts
    
    # initializing the graph
    sts = trial_graph.initialize()

    # RUN!
    sts = trial_graph.execute(0, poll_volatile_sources=False)
    
    
    # compare the output with manual calculation
    ground_truth = manual_computation(input_data)
    
    max_diff = np.max(np.abs(t_out.data - ground_truth))
    print(max_diff)
    
    if max_diff <= np.finfo(np.float64).eps:
        print("Test Passed =D")
    else:
        print("Test Failed D=")


main()
