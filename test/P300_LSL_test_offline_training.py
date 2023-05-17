# -*- coding: utf-8 -*-
"""
Created on Tues July 26 16:12:30 2022

@author: aaronlio
"""

# Create a simple graph for testing
from bcipy import bcipy
import numpy as np
import pylsl


def main():
    # create a session
    sess = bcipy.Session.create()
    online_graph = bcipy.Graph.create(sess)
    offline_trials = 500
    online_trials = 500

    Fs = 500
    trial_len = 1.0
    t_start = -0.2
    tasks = ('flash', 'target')
    resample_fs = 50


    
    # create a filter
    order = 4
    bandpass = (1,25) # in Hz
    f = bcipy.Filter.create_butter(sess,order,bandpass,btype='bandpass',fs=Fs,implementation='sos')

    channels = tuple([_ for _ in range(0,32)])

    # Data sources from LSL
    LSL_data_src = bcipy.source.InputLSLStream.create_marker_coupled_data_stream(sess, "type='EEG'", channels, relative_start=-0.2, marker_fmt='flash$|target$')
    LSL_data_out = bcipy.source.OutputLSLStream.create_outlet(sess, name='Outputs', type='type="probabilities"', channel_count=32, 
                                                              channel_format='float32')
    # TODO Change to XDF - DONE
    # training data sources from mat file
    offline_data_src = bcipy.source.BcipXDF.create_epoched(sess,
            ['C:/Users/lioa/Documents/Mindset P300 Code for Aaron/sub-P001_ses-S001_task-vP300+2x2_run-003.xdf'], 
            tasks, channels=channels, relative_start=0, Ns = Fs*trial_len) 

    online_input_data = bcipy.Tensor.create_from_handle(sess, (len(channels), 500), LSL_data_src)
    offline_input_data = bcipy.Tensor.create_from_data(sess, (len(channels), 500), offline_data_src)

    online_output_data = bcipy.Tensor.create_for_volatile_output(sess, (2,1), LSL_data_out)

    # Data input tensors
    label_input = bcipy.Scalar.create(sess, int)
    training_data = {'data'   : bcipy.CircleBuffer.create(sess, offline_trials, bcipy.Tensor(sess, (offline_input_data.shape[0],resample_fs),None,False,None)),
                     'labels' : bcipy.CircleBuffer.create(sess, offline_trials, label_input)}

    # online graph data containers (i.e. graph edges)
    pred_label = bcipy.Scalar.create_from_value(sess,-1) 

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
    t_num = 0

    while True:
        while LSL_data_src.marker_inlet.samples_available() == 0:
            print("waiting for marker...")   
        
        sts = online_graph.execute()
        
        if sts == bcipy.BcipEnums.SUCCESS:
            # print the value of the most recent trial
            label = pred_label.data
            print(f"\tTrial {t_num+1}: Predicted Label = {label}")
        else:
            print(f"Trial {t_num+1} raised error, status code: {sts}")
            break
        t_num += 1
    
    print("Test Passed =D")

if __name__ == "__main__":
    main()
