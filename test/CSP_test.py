# -*- coding: utf-8 -*-
"""
Created on Wed May 18 1:01:00 PM 2022

@author: aaronlio
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

    block = Block.create(session, 2, (2))

    #data

    order = 4
    bandpass = (8,30) # in Hz
    fs = 250
    butterworth = Filter.create_butter(session,order,bandpass,btype='bandpass',fs=fs,implementation='sos')


    input = Tensor.create_from_data(session,(500,12),input_data)
    output = Scalar.create_from_value(session,"left")
    virtual_edges = [Tensor.create_virtual(session), \
              Tensor.create_virtual(session)]

    FilterKernel.add_filter_node(block.trial_processing_graph, input, butterworth, virtual_edges[0])
    CommonSpatialPatternKernel.add_uninitialized_CSP_node(block.trial_processing_graph, virtual_edges[0], virtual_edges[1], 'data', 'labels', 2)
    LDAClassifierKernel.add_untrained_LDA_node(block.trial_processing_graph, virtual_edges[1], output=output)

    verification = session.verify()

    if verification != BcipEnums.SUCCESS:
        print(verification)
        print("Test Failed D=")
        return verification
    

    verification = session.start_block()
    if verification != BcipEnums.SUCCESS:
        print(verification)
        print("Test Failed D=")
        return verification


    execute = session.execute_trial(0)
    
    print(execute.data)
    
    print("Test Passed =D")