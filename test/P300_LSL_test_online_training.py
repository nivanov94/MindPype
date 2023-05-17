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


def main():
    # create a session
    sess = bcipy.Session.create()

    training_graph = bcipy.Graph.create(sess)

    online_graph = bcipy.Graph.create(sess)
    training_trials = 500

    Fs = 500    
    resample_fs = 50
    
    # create a filter
    order = 4
    bandpass = (1,25) # in Hz
    f = bcipy.Filter.create_butter(sess,order,bandpass,btype='bandpass',fs=Fs,implementation='sos')

    channels = tuple([_ for _ in range(0,32)])

    # Data sources from LSL
    LSL_data_src = bcipy.source.InputLSLStream.create_marker_coupled_data_stream(sess, "type='EEG'", channels, relative_start=-0.2, marker_fmt='flash$|target$')
    
    marker_streams = pylsl.resolve_bypred("type='Markers'")
    marker_inlet2 = pylsl.StreamInlet(marker_streams[0]) # for now, just take the first available marker stream
    marker_inlet2.open_stream()
    marker_pattern = re.compile('flash$|target$')


    LSL_data_out = bcipy.source.OutputLSLStream.create_outlet(sess, "type='label'", channels, 32, Fs, 'float32')
    

    online_input_data = bcipy.Tensor.create_from_handle(sess, (len(channels), 500), LSL_data_src)
    training_input_data = bcipy.Tensor.create_from_data(sess, (len(channels), 500), LSL_data_src)


    online_output_data = bcipy.Tensor.create_for_volatile_output(sess, (1,2), LSL_data_out)

    # Data input tensors
    label_conv = {'flash': 0, 'target': 1}
    training_data = {'data'   : bcipy.CircleBuffer.create(sess, 60, bcipy.Tensor.create(sess, (len(channels), resample_fs))),
                     'labels' : bcipy.CircleBuffer.create(sess, 60, bcipy.Scalar.create(sess, int))}

    # online graph data containers (i.e. graph edges)
    pred_probs = bcipy.Tensor.create_from_data(sess, (1,2), np.zeros((1,2))) 
    pred_label = bcipy.Scalar.create(sess, int)
    true_label = bcipy.Scalar.create(sess, int)

    t_virt = [bcipy.Tensor.create_virtual(sess), # output of filter, input to resample
              bcipy.Tensor.create_virtual(sess), # output of resample, input to extract
              bcipy.Tensor.create_virtual(sess), # output of extract, input to xdawn
              bcipy.Tensor.create_virtual(sess), # output of xdawn, input to tangent space
              bcipy.Tensor.create_virtual(sess)] # output of tangent space, input to classifier
    
    classifier = bcipy.Classifier.create_logistic_regression(sess)
    
    # extraction indices - TODO Ask Jason about filter-epoch execution order during online
    start_time = 0.25
    end_time = 1.25
    extract_indices = [":", # all channels
                       [_ for _ in range(int(start_time*resample_fs),int(end_time*resample_fs))] # central 1s
                      ]
    
    # TODO add offline graph (separate file)

    # training graph nodes
    bcipy.kernels.FilterKernel.add_filter_node(training_graph, training_input_data, f, t_virt[0])
    bcipy.kernels.ResampleKernel.add_resample_node(training_graph, t_virt[0], resample_fs/Fs, t_virt[1])
    bcipy.kernels.ExtractKernel.add_extract_node(training_graph, t_virt[1], extract_indices, t_virt[2])
    bcipy.kernels.XDawnCovarianceKernel.add_xdawn_covariance_node(training_graph, t_virt[2], t_virt[3],
                                                                  training_data['data'], training_data['labels'])
    bcipy.kernels.TangentSpaceKernel.add_tangent_space_node(training_graph, t_virt[3], t_virt[4])

    # Enqueue training data and labels to appropriate circle buffers
    # Training data is the result of the tangent space transform
    # Training label is the result of the classifier
    bcipy.kernels.EnqueueKernel.add_enqueue_node(training_graph, t_virt[4], training_data['data'])
    bcipy.kernels.EnqueueKernel.add_enqueue_node(training_graph, true_label, training_data['labels'])


    # online graph nodes 
    bcipy.kernels.FilterKernel.add_filter_node(online_graph, online_input_data, f, t_virt[0])
    bcipy.kernels.ResampleKernel.add_resample_node(online_graph, t_virt[0], resample_fs/Fs, t_virt[1])
    bcipy.kernels.ExtractKernel.add_extract_node(online_graph, t_virt[1], extract_indices, t_virt[2])
    bcipy.kernels.XDawnCovarianceKernel.add_xdawn_covariance_node(online_graph, t_virt[2], 
                                                                t_virt[3], training_data['data'], 
                                                                training_data['labels'])
    bcipy.kernels.TangentSpaceKernel.add_tangent_space_node(online_graph, t_virt[3], t_virt[4])
    bcipy.kernels.ClassifierKernel.add_classifier_node(online_graph, t_virt[4], classifier, pred_label, online_output_data)

    # verify the session (i.e. schedule the nodes)

    verify_sts = online_graph.verify()

    if verify_sts != bcipy.BcipEnums.SUCCESS:
        print("Test Failed D=")
        return verify_sts
    

    # initialize the classifiers (i.e., train the classifier)
    init_sts = online_graph.initialize()

    if init_sts != bcipy.BcipEnums.SUCCESS:
        print("Test Failed D=")
        return init_sts
    
    # Run the online trials
    sts = bcipy.BcipEnums.SUCCESS
    online_trials = 100
    
    # TODO add LSL output and loop to wait for marker

    for t_num in range(training_trials):
        inlet_marker, _ = marker_inlet2.pull_sample(timeout=0.1)

        if inlet_marker != None and marker_pattern.search(inlet_marker[0]) != None:
            true_label.data = label_conv[marker_pattern.search(inlet_marker[0]).group(0)]
            sts = training_graph.execute()
            if sts == bcipy.BcipEnums.SUCCESS:
                print(f"\Training Trial{t_num+1} Complete")
            else:
                break

    for t_num in range(online_trials):
        sts = online_graph.execute()
        if sts == bcipy.BcipEnums.SUCCESS:
            # print the value of the most recent trial
            y_bar = pred_probs.data
            print(f"\tTrial {t_num+1}: Max Probability = {max(y_bar)}")
        else:
            print(f"Trial {t_num+1} raised error, status code: {sts}")
            break

    print("Test Passed =D")

if __name__ == "__main__":
    main()
