# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:30:09 2019

@author: ivanovn
"""

# Create a simple graph for testing


from classes.session import Session
from classes.tensor import Tensor
from classes.filter import Filter
from classes.block import Block
from classes.bcip_enums import BcipEnums

from kernels.filter_kernel import FilterKernel
from kernels.covariance import CovarianceKernel

import numpy as np
from scipy import signal

def manual_computation(input_data):
    
    # first filter the data
    sos = signal.butter(4,(8,35),btype='bandpass',output='sos',fs=250)
    filtered_data = signal.sosfilt(sos,input_data,axis=0)
    cov_mat = np.cov(filtered_data,rowvar=False)
    
    return cov_mat

def main():
    # create a session
    s = Session.create()

    # add a block and some tensors
    b = Block.create(s,4,3)

    input_data = np.random.randn(500,12)
    t_in = Tensor.createFromData(s,(500,12),input_data)
    t_out = Tensor.create(s,(12,12))
    t_virt = Tensor.createVirtual(s)

    # create a filter
    order = 4
    bandpass = (8,35) # in Hz
    fs = 250
    f = Filter.createButter(s,order,bandpass,btype='bandpass',fs=fs,implementation='sos')

    # add the nodes
    cov_node  = CovarianceKernel.addCovarianceNode(b,t_virt,t_out)
    filt_node = FilterKernel.addFilterNode(b,t_in,f,t_virt)

    # verify the session (i.e. schedule the nodes)
    sts = s.verify()

    if sts != BcipEnums.SUCCESS:
        print(sts)
        return sts
    
    # RUN!
    sts = s.execute(0)
    
    # compare the output with manual calculation
    ground_truth = manual_computation(input_data)
    
    max_diff = np.max(np.abs(t_out.data - ground_truth))
    print(max_diff)
    
    if max_diff <= np.finfo(np.float64).eps:
        print("Test Passed =D")
    else:
        print("Test Failed D=")


main()