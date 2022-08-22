# -*- coding: utf-8 -*-
"""
Created on Tues July 26 16:12:30 2022

@author: aaronlio
"""
# For debugging 
import sys, os
sys.path.insert(0, os.getcwd())

# Create a simple graph for testing
from classes.classifier import Classifier
from classes.session import Session
from classes.tensor import Tensor
from classes.scalar import Scalar
from classes.filter import Filter
from classes.bcip_enums import BcipEnums
from classes.graph import Graph
from classes.source import BcipClassSeparated
from classes.circle_buffer import CircleBuffer

from kernels.csp import CommonSpatialPatternKernel
from kernels.filter_ import FilterKernel
from kernels.classifier_ import ClassifierKernel
from kernels.covariance import CovarianceKernel
from kernels.riemann_mdm_classifier_kernel import RiemannMDMClassifierKernel
from kernels.riemann_distance import RiemannDistanceKernel
from kernels.mean import MeanKernel
from kernels.running_average import RunningAverageKernel

import numpy as np
import scipy.io as sio
from random import shuffle


def main():
    # create a session
    session = Session.create()
    trial_graph = Graph.create(session)
    trial_graph_2 = Graph.create(session)
    trial_graph_3 = Graph.create(session)

    #data
    training_data = np.random.random((120,12,500))

    init_data = sio.loadmat('test_data\init_data.mat')['init_data']
    init_labels = sio.loadmat('test_data\init_labels.mat')['labels']
    

    X = Tensor.create_from_data(session,np.shape(init_data), init_data)
    y = Tensor.create_from_data(session,np.shape(init_labels),init_labels)


    input_data = BcipClassSeparated.create_continuous(session, 2, 500, 0, 4000, 0, 'input_data', 'input_labels', 'test_data\input_data.mat', 'test_data\input_labels.mat')
    input_data = Tensor.create_from_handle(session, (12, 500), input_data)
    
    #t_in = Tensor.create_from_data(session,(12,500),input_data)
    t_out = Tensor.create_virtual(session)
    
    t_virt = [Tensor.create_virtual(session), \
              Tensor.create_virtual(session)]

    t_virt2 = [Tensor.create_virtual(session), \
              Tensor.create_virtual(session)]

    class1avg = Tensor.create_virtual(session)
    class2avg = Tensor.create_virtual(session)


    # create a filter
    order = 4
    bandpass = (8,35) # in Hz
    fs = 250
    f = Filter.create_butter(session,order,bandpass,btype='bandpass',fs=fs,implementation='sos')
    
    # add the nodes
    FilterKernel.add_filter_node(trial_graph,input_data,f,t_virt[0])
    CovarianceKernel.add_covariance_node(trial_graph, t_virt[0], t_virt[1], 0)
    RunningAverageKernel.add_running_average_node(trial_graph, t_virt[1], class1avg, 5, 0)

    FilterKernel.add_filter_node(trial_graph_2,input_data,f,t_virt2[0])
    CovarianceKernel.add_covariance_node(trial_graph_2, t_virt2[0], t_virt2[1], 0)
    RunningAverageKernel.add_running_average_node(trial_graph_2, t_virt2[1], class2avg, 5, 0)
    
    RiemannDistanceKernel.add_riemann_distance_node(trial_graph_3, class1avg, class2avg, t_out)

    # verify the session (i.e. schedule the nodes)
    verify = session.verify()

    if verify != BcipEnums.SUCCESS:
        print(verify)
        print("Test Failed D=")
        return verify
    
    start = trial_graph.initialize()

    if start != BcipEnums.SUCCESS:
        print(start)
        print("Test Failed D=")
        return start
    
    start = trial_graph_2.initialize()

    if start != BcipEnums.SUCCESS:
        print(start)
        print("Test Failed D=")
        return start

    start = trial_graph_3.initialize()
    if start != BcipEnums.SUCCESS:
        print(start)
        print("Test Failed D=")
        return start

    # RUN!
    trial_seq = [0]*4 + [1]*4
    
    shuffle(trial_seq)

    t_num = 0

    sts = BcipEnums.SUCCESS
    
    while t_num < 8 and sts == BcipEnums.SUCCESS:
        print(f"t_num {t_num+1}, length of trials: {len(trial_seq)}")
        y = trial_seq[t_num]
        if y == 0:
            sts = trial_graph.execute(y)
        else:
            sts = trial_graph_2.execute(y)

        t_num += 1
        
    sts = trial_graph_3.execute(0, False)

    print(t_out.data)

    if sts != BcipEnums.SUCCESS:
        print(sts)
        print("Test Failed D=")
        return sts

    print("Test Passed =D")
    return BcipEnums.SUCCESS
if __name__ == "__main__":
    main()
