# -*- coding: utf-8 -*-
"""
Created on Tues July 26 16:12:30 2022

@author: aaronlio
"""

# Create a simple graph for testing
from bcipy import bcipy
import numpy as np
import pylsl
import re
import time

def main():
    # create a session
    sess = bcipy.Session.create()

    # Create a graph that will be used to collect training data and labels
    training_graph = bcipy.Graph.create(sess)

    # Create a graph that will be used to run online trials
    online_graph = bcipy.Graph.create(sess)
    
    # Constants
    training_trials = 6          

    Fs = 128    
    resample_fs = 50
    
    # create a filter
    order = 4
    bandpass = (1,25) # in Hz
    f = bcipy.Filter.create_butter(sess,order,bandpass,btype='bandpass',fs=Fs,implementation='sos')

    # Channels to use
    channels = tuple([_ for _ in range(3,17)])

    # Data sources from LSL
    LSL_data_src = bcipy.source.InputLSLStream.create_marker_coupled_data_stream(sess, "type='EEG'", channels, relative_start=-0.4, marker_fmt='(^SPACE pressed$)|(^RSHIFT pressed$)')
    LSL_data_out = bcipy.source.OutputLSLStream.create_outlet(sess, 'outlet', 'type="Markers"', 1, channel_format='float32')
    
    # Data input tensors connected to LSL data sources
    online_input_data = bcipy.Tensor.create_from_handle(sess, (len(channels), 700), LSL_data_src)
    training_input_data = bcipy.Tensor.create_from_handle(sess, (len(channels), 700), LSL_data_src)

    # Data output tensors connected to LSL data sources
    online_output_data = bcipy.Tensor.create_for_volatile_output(sess, (1,2), LSL_data_out)

    # Initialization data circle buffers; the training graph will enqueue the training data to these buffers with each trial
    training_data = {'data'   : bcipy.CircleBuffer.create(sess, training_trials, bcipy.Tensor.create(sess, (len(channels), 700))),
                     'labels' : bcipy.CircleBuffer.create(sess, training_trials, bcipy.Scalar.create(sess, int))}
            
    # output classifier label
    pred_label = bcipy.Scalar.create(sess, int)

    # virtual tensors to connect the nodes in the online graph
    t_virt = [bcipy.Tensor.create_virtual(sess), # output of filter, input to resample
              bcipy.Tensor.create_virtual(sess), # output of resample, input to extract
              bcipy.Tensor.create_virtual(sess), # output of extract, input to xdawn
              bcipy.Tensor.create_virtual(sess), # output of xdawn, input to tangent space
              bcipy.Tensor.create_virtual(sess)] # output of tangent space, input to classifier
    
    classifier = bcipy.Classifier.create_logistic_regression(sess)
    
    # extraction indices
    start_time = 0.2
    end_time = 1.2
    extract_indices = [":", # all channels
                       [_ for _ in range(int(start_time*resample_fs),int(end_time*resample_fs))] # central 1s
                      ]
    

    # add the enqueue node to the training graph, will automatically enqueue the data from the lsl
    bcipy.kernels.EnqueueKernel.add_enqueue_node(training_graph, training_input_data, training_data['data'])
    
    # online graph nodes 
    bcipy.kernels.FiltFiltKernel.add_filtfilt_node(online_graph, online_input_data, f, t_virt[0])
    bcipy.kernels.ResampleKernel.add_resample_node(online_graph, t_virt[0], resample_fs/Fs, t_virt[1])
    bcipy.kernels.ExtractKernel.add_extract_node(online_graph, t_virt[1], extract_indices, t_virt[2])
    bcipy.kernels.XDawnCovarianceKernel.add_xdawn_covariance_node(online_graph, t_virt[2], 
                                                                t_virt[3], training_data['data'], 
                                                                training_data['labels'])
    bcipy.kernels.TangentSpaceKernel.add_tangent_space_node(online_graph, t_virt[3], t_virt[4])
    bcipy.kernels.ClassifierKernel.add_classifier_node(online_graph, t_virt[4], classifier, pred_label, online_output_data)

    # verify the training graph (i.e. schedule the nodes)
    sts = training_graph.verify()
    if sts != bcipy.BcipEnums.SUCCESS:
        print("Training verification failed")
        return sts

    print("Training graph verified successfully")

    # verify the online graph (i.e. schedule the nodes)
    verify_sts = online_graph.verify()

    if verify_sts != bcipy.BcipEnums.SUCCESS:
        print("Test Failed D=")
        return verify_sts
    
    # initialize the training graph
    sts = training_graph.initialize()
    if sts != bcipy.BcipEnums.SUCCESS:
        print("training graph init failure")
        return
    
    # Execute the training graph
    for t_num in range(training_trials):
       
        sts = training_graph.execute()

        # Get the most recent marker and add the equivalent label to the training labels circle buffer.
        last_marker = LSL_data_src.last_marker()
        training_data['labels'].enqueue(bcipy.Scalar.create_from_value(sess, 1) if last_marker == 'SPACE pressed' else bcipy.Scalar.create_from_value(sess, 0))
        
        if sts == bcipy.BcipEnums.SUCCESS:
            print(f"Training Trial {t_num+1} Complete")
        else:
            print("Training graph error...")
            return
             
    # initialize the online graph with the collected data
    sts = online_graph.initialize(training_data['data'], training_data['labels'])
    if sts != bcipy.BcipEnums.SUCCESS:
        print("online graph init failure")
        return
    
    # Execute the online graph
    t_num = 0
    while True:

        sts = online_graph.execute()
                
        if sts == bcipy.BcipEnums.SUCCESS:
            # print the value of the most recent trial
            y_bar = online_output_data.data
            print(f"\tTrial {t_num+1}: Max Probability = {y_bar}")
        else:
            print(f"Trial {t_num+1} raised error, status code: {sts}")
            return
        t_num+=1


if __name__ == "__main__":
    main()
