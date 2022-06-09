# -*- coding: utf-8 -*-
"""
Created on Wed May 18 1:01:00 PM 2022

@author: aaronlio
"""
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


from kernels.filter_ import FilterKernel
from kernels.covariance import CovarianceKernel
from kernels.csp import CommonSpatialPatternKernel
from kernels.riemann_mdm_classifier_kernel import RiemannMDMClassifierKernel
from kernels.lda import LDAClassifierKernel

import numpy as np
from scipy import signal

def filter(input_data):
    filter = signal.butter(4, [8, 30], 'bandpass', output='sos', fs=200)
    filteredData = signal.sosfilt(filter, input_data)


def main():
    session = Session.create()
    trial_graph = Graph.create(session)
    block = Block.create(session, 3, (4,4,4))

    #data
    raw_training_data = np.random.normal(loc=0.0,scale=1.0,size=(180,250,12))
    training_data = np.zeros((180,12,12))
    for i in range(180):
        training_data[i,:,:] = np.cov(raw_training_data[i,:,:],rowvar=False)
        
    labels = np.asarray([0]*60 + [1]*60 + [2]*60)
    X = Tensor.create_from_data(session,training_data.shape,training_data)
    y = Tensor.create_from_data(session,labels.shape,labels)

    input_data = np.random.randn(500,12)
    input = Tensor.create_from_data(session,(500,12),input_data)
    output = Scalar.create_from_value(session,"left")
    virtual_edges = [Tensor.create_virtual(session), \
              Tensor.create_virtual(session)]

    order = 4
    bandpass = (8,30) # in Hz
    fs = 250
    butterworth = Filter.create_butter(session,order,bandpass,btype='bandpass',fs=fs,implementation='sos')


    FilterKernel.add_filter_node(trial_graph, input, butterworth, virtual_edges[0])
    CommonSpatialPatternKernel.add_uninitialized_CSP_node(trial_graph, virtual_edges[0], virtual_edges[1], training_data, labels, 2)
    
    LDAClassifierKernel.add_untrained_LDA_node(trial_graph, X, y, training_data, labels, pred_proba=80)

    verification = session.verify()

    if verification != BcipEnums.SUCCESS:
        print(verification)
        print("Test Failed D=")
        return verification
    

    verification = session.start_block(trial_graph)
    if verification != BcipEnums.SUCCESS:
        print(verification)
        print("Test Failed D=")
        return verification


    execute = session.execute_trial(0, trial_graph)
    
    print(execute.data)
    
    print("Test Passed =D")

if __name__ == '__main__':
    main()