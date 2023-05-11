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


def main():
    # create a session
    sess = bcipy.Session.create()
    offline_graph = bcipy.Graph.create(sess)
    online_graph = bcipy.Graph.create(sess)

    offline_trials = 252
    online_trials = 202

    Fs = 500
    trial_len = 1.5
    t_start = -0.45
    
    resample_fs = 50
    
    # create a filter
    order = 4
    bandpass = (8,35) # in Hz
    f = bcipy.Filter.create_butter(sess,order,bandpass,btype='bandpass',fs=Fs,implementation='sos')

    channels = tuple([_ for _ in range(0,32)])

    # Data sources from MAT files
    LSL_data_src = bcipy.source.V2LSLStream.create_marker_coupled_data_stream(sess, "type='EEG'", channels, relative_start=-0.2, marker_fmt='flash$|target$')

    offline_data_src = bcipy.source.BcipContinuousMat.create_continuous(sess,
            '../data/p300_offline.mat', Fs*trial_len, relative_start=0, channels=channels,
            mat_data_var_name='EEG', mat_labels_var_name='labels')

    online_input_data = bcipy.Tensor.create_from_handle(sess, (len(channels), 500), LSL_data_src)
    offline_input_data = bcipy.Tensor.create_from_data(sess, (len(channels), 500), offline_data_src)

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
    
    # extraction indices
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
    bcipy.kernels.ClassifierKernel.add_classifier_node(online_graph, t_virt[4], classifier, pred_label)

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
    
    for t_num in range(online_trials):
        sts = online_graph.execute()
        if sts == bcipy.BcipEnums.SUCCESS:
            # print the value of the most recent trial
            y_bar = pred_label.data
            print(f"\tTrial {t_num+1}: Predicted label = {y_bar}")
        else:
            print(f"Trial {t_num+1} raised error, status code: {sts}")
            break

    print("Test Passed =D")

if __name__ == "__main__":
    main()
