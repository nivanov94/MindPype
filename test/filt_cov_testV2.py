# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:30:09 2019

@author: ivanovn
"""

# Create a simple graph for testing
import sys, os
sys.path.insert(0, os.getcwd())

from classes.session import Session
from classes.tensor import Tensor
from classes.filter import Filter
from classes.bcip_enums import BcipEnums
from classes.graph import Graph

from kernels.filter_ import FilterKernel
from kernels.covariance import CovarianceKernel

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
    s = Session.create()

    trial_graph = Graph.create(s)

    input_data = np.random.randn(12,500)
    t_in = Tensor.create_from_data(s,(12,500),input_data)
    t_out = Tensor.create(s,(12,12))
    t_virt = Tensor.create_virtual(s)

    # create a filter
    order = 4
    bandpass = (8,35) # in Hz
    fs = 250
    f = Filter.create_butter(s,order,bandpass,btype='bandpass',fs=fs,implementation='sos')

    # add the nodes
    FilterKernel.add_filter_node(trial_graph,t_in,f,t_virt)
    CovarianceKernel.add_covariance_node(trial_graph,t_virt,t_out)
    

    # verify the session (i.e. schedule the nodes)
    sts = trial_graph.verify()

    if sts != BcipEnums.SUCCESS:
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
