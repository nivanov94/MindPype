# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:12:30 2019

@author: ivanovn
"""
# For debugging 
import sys, os
sys.path.insert(0, os.getcwd())

# Create a simple graph for testing

from classes.session import Session
from classes.tensor import Tensor
from classes.scalar import Scalar
from classes.filter import Filter
from classes.block import Block
from classes.bcip_enums import BcipEnums
from classes.graph import Graph
from classes.source import BcipContinuousMat

from kernels.filter_ import FilterKernel
from kernels.covariance import CovarianceKernel
from kernels.riemann_mdm_classifier_kernel import RiemannMDMClassifierKernel

import numpy as np

def main():
    # create a session
    session = Session.create()
    trial_graph = Graph.create(session)

    # add a block and some tensors
    block = Block.create(session,3,(4,4,4))
    
    data = BcipContinuousMat.create_continuous(session, 2, 800, 0, 160000, 'data\eegdata.mat', 'data\labels.mat')

    # need to take 20% of each array to use as training data


    # initialize the classifier
    # raw_training_data = np.random.normal(loc=0.0,scale=1.0,size=(180,250,12))
    # training_data = np.zeros((180,12,12))
    # for i in range(180):
    #    training_data[i,:,:] = np.cov(raw_training_data[i,:,:],rowvar=False)
    
    
    """X = Tensor.create_from_data(session,data.shape,data)
    y = Tensor.create_from_data(session,labels.shape,labels)

    input_data = np.random.randn(500,12)
    t_in = Tensor.create_from_data(session,(500,12),input_data)
    s_out = Scalar.create_from_value(session,-1)
    t_virt = [Tensor.create_virtual(session), \
              Tensor.create_virtual(session)]
    
    # create a filter
    order = 4
    bandpass = (8,35) # in Hz
    fs = 250
    f = Filter.create_butter(session,order,bandpass,btype='bandpass',fs=fs,implementation='sos')

    # add the nodes
    CovarianceKernel.add_covariance_node(trial_graph,t_virt[0],t_virt[1])
    FilterKernel.add_filter_node(trial_graph,t_in,f,t_virt[0])
    RiemannMDMClassifierKernel.add_untrained_riemann_MDM_node(trial_graph,
                                                              t_virt[1],
                                                              s_out,X,y)

    # verify the session (i.e. schedule the nodes)
    verify = session.verify()

    if verify != BcipEnums.SUCCESS:
        print(verify)
        print("Test Failed D=")
        return verify
    
    start = session.start_block(trial_graph)
    if start != BcipEnums.SUCCESS:
        print(start)
        print("Test Failed D=")
        return start
    
    # RUN!
    execute = session.execute_trial(0, trial_graph)
    
    print(s_out.data)
    
    print("Test Passed =D")"""

if __name__ == "__main__":
    main()
