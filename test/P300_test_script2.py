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


def main():
    # create a session
    session = bcipy.Session.create()
    trial_graph = bcipy.Graph.create(session)

    #data
    init_data = sio.loadmat('test_data\init_data.mat')['init_data']
    init_labels = sio.loadmat('test_data\init_labels.mat')['labels']
    

    X = bcipy.Tensor.create_from_data(session,np.shape(init_data), init_data)
    y = bcipy.Tensor.create_from_data(session,np.shape(init_labels),init_labels)


    input_data = bcipy.source.BcipContinuousMat.create_continuous(session, 500, 0, 4000, 0, 'input_data', 'input_labels', 'test_data\input_data.mat', 'test_data\input_labels.mat')
    input_data = bcipy.Tensor.create_from_handle(session, (12, 500), input_data)
    
    s_out = bcipy.Scalar.create_from_value(session,-1)
    t_virt = [bcipy.Tensor.create_virtual(session), \
              bcipy.Tensor.create_virtual(session)]

    classifier = bcipy.Classifier.create_logistic_regression(session)
    
    # add the nodes
    bcipy.kernels.XDawnCovarianceKernel.add_xdawn_covariance_node(trial_graph, input_data, t_virt[0], X, y, 4)
    bcipy.kernels.TangentSpaceKernel.add_tangent_space_node(trial_graph, t_virt[0], t_virt[1], None, metric='riemann', tsupdate=False, sample_weight=None)
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
    t_num = 0
    sts = bcipy.BcipEnums.SUCCESS
    
    while sts == bcipy.BcipEnums.SUCCESS:
        try:
            sts = trial_graph.execute()
            if sts == bcipy.BcipEnums.SUCCESS:
                # print the value of the most recent trial
                y_bar = s_out.data
                print("Trial {}: Predicted label = {}".format(t_num+1,y_bar))
                
            else:
                print(f"Trial {t_num+1} raised error, status code: {sts}")
                break
        
        except ValueError:
            print("Trial Data completed")
            break

        t_num += 1
    
    print("Test Passed =D")

if __name__ == "__main__":
    main()
