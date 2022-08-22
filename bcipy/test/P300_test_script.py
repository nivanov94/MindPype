# -*- coding: utf-8 -*-
"""
Created on Tues July 26 16:12:30 2022

@author: aaronlio
"""
# For debugging 
import sys, os
from tkinter import N
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

from kernels.tangent_space import TangentSpaceKernel
from kernels.xdawn_covariances import XDawnCovarianceKernel
from kernels.classifier_ import ClassifierKernel

import numpy as np
import scipy.io as sio
from random import sample, shuffle

def main():
    # create a session
    session = Session.create()
    trial_graph = Graph.create(session)

    #data
    init_data = sio.loadmat('test_data\init_data.mat')['init_data']
    init_labels = sio.loadmat('test_data\init_labels.mat')['labels']
    

    X = Tensor.create_from_data(session,np.shape(init_data), init_data)
    y = Tensor.create_from_data(session,np.shape(init_labels),init_labels)


    input_data = BcipClassSeparated.create_continuous(session, 2, 500, 0, 4000, 0, 'input_data', 'input_labels', 'test_data\input_data.mat', 'test_data\input_labels.mat')

    input_data = Tensor.create_from_handle(session, (12, 500), input_data)
    
    #t_in = Tensor.create_from_data(session,(12,500),input_data)
    s_out = Scalar.create_from_value(session,-1)
    t_virt = [Tensor.create_virtual(session), \
              Tensor.create_virtual(session)]
    
    # create a filter
    order = 4
    bandpass = (8,35) # in Hz
    fs = 250

    classifier = Classifier.create_logistic_regression(session)
    
    #TangentSpaceWeight = [None]*120
    # add the nodes
    XDawnCovarianceKernel.add_xdawn_covariance_kernel(trial_graph, input_data, t_virt[0], X, y, 4)
    TangentSpaceKernel.add_tangent_space_kernel(trial_graph, t_virt[0], t_virt[1], None, metric='riemann', tsupdate=False, sample_weight=None)
    ClassifierKernel.add_classifier_node(trial_graph, t_virt[1], classifier, s_out, None, None)

    # verify the session (i.e. schedule the nodes)
    verify = trial_graph.verify()

    if verify != BcipEnums.SUCCESS:
        print(verify)
        print("Test Failed D=")
        return verify
    
    start = trial_graph.initialize()


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

    
    while t_num < 8 and sts == BcipEnums.SUCCESS:
        print(f"t_num {t_num}, length of trials: {len(trial_seq)}")
        y = trial_seq[t_num]
        sts = trial_graph.execute(y)
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
