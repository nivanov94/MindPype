# -*- coding: utf-8 -*-
"""
Created on Tues July 26 16:12:30 2022

@author: aaronlio
"""
# For debugging 
import sys, os
sys.path.insert(0, os.getcwd())

# Create a simple graph for testing
from bcipy import bcipy

import numpy as np
import scipy.io as sio
from random import shuffle

def main():
    # create a session
    session = bcipy.Session.create()
    trial_graph = bcipy.Graph.create(session)

    init_data = sio.loadmat('test_data\init_data.mat')['init_data']
    init_labels = sio.loadmat('test_data\init_labels.mat')['labels']

    X = bcipy.Tensor.create_from_data(session,np.shape(init_data), init_data)
    y = bcipy.Tensor.create_from_data(session,np.shape(init_labels),init_labels)

    input_data = bcipy.source.BcipClassSeparatedMat.create_class_separated(session, 2, 500, 0, 4000, 0, 'input_data', 'input_labels', \
                                                              'test_data\input_data.mat', 'test_data\input_labels.mat')
    
    input_data = bcipy.Tensor.create_from_handle(session, (12, 500), input_data)

    s_out = bcipy.Scalar.create_from_value(session,-1)
    t_virt = [bcipy.Tensor.create_virtual(session), \
              bcipy.Tensor.create_virtual(session)]
    
    # create a filter
    f = bcipy.Filter.create_butter(session,4,(8,35),btype='bandpass',fs=250,implementation='sos')
    classifier = bcipy.Classifier.create_LDA(session)
    
    # add the nodes
    bcipy.kernels.FilterKernel.add_filter_node(trial_graph,input_data,f,t_virt[0])
    bcipy.kernels.CommonSpatialPatternKernel.add_uninitialized_CSP_node(trial_graph, t_virt[0], t_virt[1], X, y, 2)
    bcipy.kernels.ClassifierKernel.add_classifier_node(trial_graph, t_virt[1], classifier, s_out, None, None)

    # verify the session (i.e. schedule the nodes)
    verify = trial_graph.verify()

    if verify != bcipy.BcipEnums.SUCCESS:
        print(verify)
        print("Test Failed D=")
        return verify
    
    start = trial_graph.initialize()

    if start != bcipy.BcipEnums.SUCCESS:
        print(start)
        print("Test Failed D=")
        return start
    
    # RUN!
    trial_seq = [0]*4 + [1]*4
    
    shuffle(trial_seq)

    t_num = 0
    sts = bcipy.BcipEnums.SUCCESS
    correct_labels = 0

    
    while t_num < 8 and sts == bcipy.BcipEnums.SUCCESS:
        print(f"t_num {t_num}, length of trials: {len(trial_seq)}")
        y = trial_seq[t_num]
        sts = trial_graph.execute(y)
        if sts == bcipy.BcipEnums.SUCCESS:
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
