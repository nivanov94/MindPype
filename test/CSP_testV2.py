# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:12:30 2019

@author: ivanovn
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

from kernels.csp import CommonSpatialPatternKernel
from kernels.filter_ import FilterKernel
from kernels.classifier_ import ClassifierKernel
from kernels.covariance import CovarianceKernel
from kernels.riemann_mdm_classifier_kernel import RiemannMDMClassifierKernel

import numpy as np
from random import shuffle

def main():
    # create a session
    session = Session.create()
    trial_graph = Graph.create(session)

    #data
    training_data = np.random.random((120,12,500))
        
    labels = np.asarray([0]*60 + [1]*60)

    X = Tensor.create_from_data(session,training_data.shape,training_data)
    y = Tensor.create_from_data(session,labels.shape,labels)

    y_LDA = Tensor.create_from_data(session, labels.shape, labels)


    input_data = np.random.randn(12, 500)

    t_in = Tensor.create_from_data(session,(12,500),input_data)
    s_out = Scalar.create_from_value(session,-1)
    t_virt = [Tensor.create_virtual(session), \
              Tensor.create_virtual(session)]
    
    # create a filter
    order = 4
    bandpass = (8,35) # in Hz
    fs = 250
    f = Filter.create_butter(session,order,bandpass,btype='bandpass',fs=fs,implementation='sos')

    classifier = Classifier.create_LDA(session)
    
    # add the nodes
    FilterKernel.add_filter_node(trial_graph,t_in,f,t_virt[0])
    CommonSpatialPatternKernel.add_uninitialized_CSP_node(trial_graph, t_virt[0], t_virt[1], X, y, 2)
    ClassifierKernel.add_classifier_node(trial_graph, t_virt[1], classifier, s_out, None, None)
    


    # verify the session (i.e. schedule the nodes)
    verify = trial_graph.verify()
    print(trial_graph._missing_data)

    if verify != BcipEnums.SUCCESS:
        print(verify)
        print("Test Failed D=")
        return verify
    
    start = trial_graph.initialize()

    print(trial_graph._nodes[1]._kernel._init_params['labels']._id)
    print(trial_graph._nodes[2].kernel._labels._id)

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