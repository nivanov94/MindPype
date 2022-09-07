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
from classes.circle_buffer import CircleBuffer
from classes.filter import Filter
from classes.bcip_enums import BcipEnums
from classes.graph import Graph
from classes.source import BcipContinuousMat

from kernels.filter_ import FilterKernel
from kernels.extract import ExtractKernel
from kernels.resample import ResampleKernel
from kernels.enqueue import EnqueueKernel
from kernels.tangent_space import TangentSpaceKernel
from kernels.xdawn_covariances import XDawnCovarianceKernel
from kernels.classifier_ import ClassifierKernel


def main():
    # create a session
    sess = Session.create()
    offline_graph = Graph.create(sess)
    online_graph = Graph.create(sess)

    offline_trials = 252
    online_trials = 202

    Fs = 128
    trial_len = 1.5
    t_start = -0.45
    
    resample_fs = 50
    
    # create a filter
    order = 4
    bandpass = (8,35) # in Hz
    f = Filter.create_butter(sess,order,bandpass,btype='bandpass',fs=Fs,implementation='sos')

    channels = tuple([_ for _ in range(3,17)])

    # Data sources from MAT files
    offline_data_src = BcipContinuousMat.create_continuous(sess,
                                                           '../data/p300_offline.mat',
                                                           Fs*trial_len, 
                                                           relative_start=t_start*Fs,
                                                           channels=channels,
                                                           mat_data_var_name='EEG',
                                                           mat_labels_var_name='labels')



    online_data_src = BcipContinuousMat.create_continuous(sess,
                                                          '../data/p300_online.mat',
                                                          Fs*trial_len, 
                                                          relative_start=t_start*Fs,
                                                          channels=channels,
                                                          mat_data_var_name='EEG',
                                                          mat_labels_var_name='labels')

    # Data input tensors
    online_input_data = Tensor.create_from_handle(sess, (len(channels), int(trial_len*Fs)), online_data_src)
    offline_input_data = Tensor.create_from_handle(sess, (len(channels), int(trial_len*Fs)), offline_data_src)
    
    # offline graph data containers
    label_input = Scalar.create(sess, int)
    training_data = {'data'   : CircleBuffer.create(sess, offline_trials, Tensor(sess, (offline_input_data.shape[0],resample_fs),None,False,None)),
                     'labels' : CircleBuffer.create(sess, offline_trials, label_input)}

    # online graph data containers (i.e. graph edges)
    pred_label = Scalar.create_from_value(sess,-1) 
    t_virt = [Tensor.create_virtual(sess), # output of filter, input to resample
              Tensor.create_virtual(sess), # output of resample, input to extract
              Tensor.create_virtual(sess), # output of extract, input to xdawn
              Tensor.create_virtual(sess), # output of xdawn, input to tangent space
              Tensor.create_virtual(sess)] # output of tangent space, input to classifier
    
    classifier = Classifier.create_logistic_regression(sess)
    
    # extraction indices
    start_time = 0.25
    end_time = 1.25
    extract_indices = [":", # all channels
                       [_ for _ in range(int(start_time*resample_fs),int(end_time*resample_fs))] # central 1s
                      ]
    
    # offline graph nodes
    FilterKernel.add_filter_node(offline_graph, offline_input_data, f, t_virt[0])
    ResampleKernel.add_resample_node(offline_graph, t_virt[0], resample_fs/Fs, t_virt[1])
    ExtractKernel.add_extract_node(offline_graph, t_virt[1], extract_indices, t_virt[2])
    EnqueueKernel.add_enqueue_node(offline_graph, t_virt[2], training_data['data'])
    EnqueueKernel.add_enqueue_node(offline_graph, label_input, training_data['labels'])

    # online graph nodes
    FilterKernel.add_filter_node(online_graph, online_input_data, f, t_virt[0])
    ResampleKernel.add_resample_node(online_graph, t_virt[0], resample_fs/Fs, t_virt[1])
    ExtractKernel.add_extract_node(online_graph, t_virt[1], extract_indices, t_virt[2])
    XDawnCovarianceKernel.add_xdawn_covariance_node(online_graph, t_virt[2], 
                                                    t_virt[3], training_data['data'], 
                                                    training_data['labels'])
    TangentSpaceKernel.add_tangent_space_node(online_graph, t_virt[3], t_virt[4])
    ClassifierKernel.add_classifier_node(online_graph, t_virt[4], classifier, pred_label)

    # verify the session (i.e. schedule the nodes)
    for g in (offline_graph, online_graph):
        verify_sts = g.verify()

        if verify_sts != BcipEnums.SUCCESS:
            print("Test Failed D=")
            return verify_sts
    

    # Run the offline trials
    for t_num in range(offline_trials):
        # set the label input scalar
        print(f"Running trial {t_num+1} of {offline_trials}...")
        label_input.data = offline_data_src.get_next_label()
        offline_graph.execute()


    # initialize the classifiers (i.e., train the classifier)
    init_sts = online_graph.initialize()

    if init_sts != BcipEnums.SUCCESS:
        print("Test Failed D=")
        return init_sts
    
    # Run the online trials
    sts = BcipEnums.SUCCESS
    online_trials = 202
    
    for t_num in range(online_trials):
        print(f"Running trial {t_num+1} of {online_trials}...")
        sts = online_graph.execute()
        if sts == BcipEnums.SUCCESS:
            # print the value of the most recent trial
            y_bar = pred_label.data
            print(f"\tTrial {t_num+1}: Predicted label = {y_bar}")
        else:
            print(f"Trial {t_num+1} raised error, status code: {sts}")
            break

    print("Test Passed =D")

if __name__ == "__main__":
    main()
