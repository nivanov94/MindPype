# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:12:30 2019

@author: ivanovn
"""

# Create a simple graph for testing


from classes.session import Session
from classes.tensor import Tensor
from classes.scalar import Scalar
from classes.filter import Filter
from classes.block import Block
from classes.bcip_enums import BcipEnums

from kernels.filter_ import FilterKernel
from kernels.covariance import CovarianceKernel
from kernels.riemann_mdm_classifier_kernel import RiemannMDMClassifierKernel

import numpy as np


def main():
    # create a session
    s = Session.create()

    # add a block and some tensors
    b = Block.create(s,3,(4,4,4))

    # initialize the classifier
    # fake data for training
    raw_training_data = np.random.normal(loc=0.0,scale=1.0,size=(180,250,12))
    training_data = np.zeros((180,12,12))
    for i in range(180):
        training_data[i,:,:] = np.cov(raw_training_data[i,:,:],rowvar=False)
        
    labels = np.asarray([0]*60 + [1]*60 + [2]*60)
    X = Tensor.create_from_data(s,training_data.shape,training_data)
    y = Tensor.create_from_data(s,labels.shape,labels)

    input_data = np.random.randn(500,12)
    t_in = Tensor.create_from_data(s,(500,12),input_data)
    s_out = Scalar.create_from_value(s,-1)
    t_virt = [Tensor.create_virtual(s), \
              Tensor.create_virtual(s)]
    
    # create a filter
    order = 4
    bandpass = (8,35) # in Hz
    fs = 250
    f = Filter.create_butter(s,order,bandpass,btype='bandpass',fs=fs,implementation='sos')

    # add the nodes
    CovarianceKernel.add_covariance_node(b.trial_processing_graph,t_virt[0],t_virt[1])
    FilterKernel.add_filter_node(b.trial_processing_graph,t_in,f,t_virt[0])
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
    
    # RUN!
    sts = s.execute_trial(0)
    
    print(s_out.data)
    
    print("Test Passed =D")


if __name__ == "__main__":
    main()
