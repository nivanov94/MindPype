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
from classes.classifier import Classifier
from classes.session import Session
from classes.tensor import Tensor
from classes.scalar import Scalar
from classes.filter import Filter
from classes.block import Block
from classes.bcip_enums import BcipEnums
from classes.graph import Graph
from classes.source import BcipContinuousMat

from kernels.csp import CommonSpatialPatternKernel
from kernels.filter_ import FilterKernel
from kernels.classifier_ import ClassifierKernel
from kernels.covariance import CovarianceKernel
from kernels.riemann_mdm_classifier_kernel import RiemannMDMClassifierKernel
from kernels.riemann_distance import RiemannDistanceKernel
from kernels.mean import MeanKernel
from kernels.riemann_mean import RiemannMeanKernel

import numpy as np
import scipy.io as sio
from random import shuffle

def main():
    # create a session
    session = Session.create()
    trial_graph = Graph.create(session)
    trial_graph_1b = Graph.create(session)
    trial_graph_2 = Graph.create(session)
    block = Block.create(session, 2, (4,4))

    #data
    training_data = np.random.random((120,12,500))

    init_data = sio.loadmat('test_data\init_data.mat')['init_data']
    init_labels = sio.loadmat('test_data\init_labels.mat')['labels']
    

    X = Tensor.create_from_data(session,np.shape(init_data), init_data)
    y = Tensor.create_from_data(session,np.shape(init_labels),init_labels)


    input_data = BcipContinuousMat.create_continuous(session, 2, 500, 0, 4000, 0, 'input_data', 'input_labels', 'test_data\input_data.mat', 'test_data\input_labels.mat')

    input_data = Tensor.create_from_handle(session, (12, 500), input_data)
    
    #t_in = Tensor.create_from_data(session,(12,500),input_data)
    s_out = Scalar.create_from_value(session,-1)
    t_virt = [Tensor.create_virtual(session), \
              Tensor.create_virtual(session)]


    t_virt_1b = [Tensor.create_virtual(session), \
              Tensor.create_virtual(session)
              ]
    
    t_out_1a = Tensor.create(session, (4,12,12))
    t_out_1b = Tensor.create(session, (4,12,12))
    # create a filter
    order = 4
    bandpass = (8,35) # in Hz
    fs = 250
    f = Filter.create_butter(session,order,bandpass,btype='bandpass',fs=fs,implementation='sos')
    
    # add the nodes
    

    FilterKernel.add_filter_node(trial_graph,input_data,f,t_virt[0])
    CovarianceKernel.add_covariance_node(trial_graph, t_virt[0], t_virt[1], 0)
    RiemannMeanKernel.add_riemann_mean_node(trial_graph, t_virt[1], t_out_1a)


    FilterKernel.add_filter_node(trial_graph_1b,input_data,f,t_virt_1b[0])
    CovarianceKernel.add_covariance_node(trial_graph_1b, t_virt_1b[0], t_virt_1b[1], 0)
    RiemannMeanKernel.add_riemann_mean_node(trial_graph_1b, t_virt_1b[1], t_out_1b)
    
    RiemannDistanceKernel.add_riemann_distance_node(trial_graph_2, t_out_1a, t_out_1b, s_out)

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
    trial_seq = [0]*4 + [1]*4
    
    shuffle(trial_seq)

    t_num = 0
    sts = BcipEnums.SUCCESS
    correct_labels = 0

    
    while sum(block.remaining_trials()) != 0 and sts == BcipEnums.SUCCESS:
        print(f"t_num {t_num}, length of trials: {len(trial_seq)}")
        y = trial_seq[t_num]
        sts = session.execute_trial(y, trial_graph)
        if sts == BcipEnums.SUCCESS:
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
