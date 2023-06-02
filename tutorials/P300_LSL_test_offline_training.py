# -*- coding: utf-8 -*-
"""
Created on Tues July 26 16:12:30 2022
@author: aaronlio
"""

# Create a simple graph for testing
from bcipy import bcipy
import numpy as np
import pylsl
import json


def main(file):
    # create a session
    sess = bcipy.Session.create()
    online_graph = bcipy.Graph.create(sess)
    training_graph = bcipy.Graph.create(sess)

    # Cosntants
    Fs = 128
    trial_len = 1.4
    tasks = ('non-target', 'target')
    resample_fs = 50

    
    # create a filter
    order = 4
    bandpass = (1,25) # in Hz
    f = bcipy.Filter.create_butter(sess,order,bandpass,btype='bandpass',fs=Fs,implementation='sos')

    channels = tuple([_ for _ in range(3,17)])

    # Data sources from LSL
    print("Session Starting...")
    LSL_data_src = bcipy.source.InputLSLStream.create_marker_coupled_data_stream(sess, "type='EEG'", channels, relative_start=-0.4, marker_fmt='.*flash', marker_pred="type='Marker'")
    
    # training data sources from XDF file
    offline_data_src = bcipy.source.BcipXDF.create_continuous(sess,
            file, 
            tasks, channels=channels, relative_start=-.4, Ns = np.ceil(Fs*trial_len)) 

    # Create input tensors
    online_input_data = bcipy.Tensor.create_from_handle(sess, (len(channels), 180), LSL_data_src)
    offline_input_data = bcipy.Tensor.create_from_handle(sess, (len(channels), 180), offline_data_src)

    # Create circle buffer to store true labels
    labels = bcipy.CircleBuffer.create(sess, len(offline_data_src.trial_data['Markers']['time_series']), bcipy.Scalar.create(sess, int))

    ls = offline_data_src.trial_data['Markers']['time_series']
    target_pos = None
    task_series_list = []
    # Convert string markers to integer labels
    for i in range(len(ls)):
        scalar = bcipy.Scalar.create(sess, int)
        if list(json.loads(ls[i][0]).keys())[0] == 'target':
            target_pos = list(json.loads(ls[i][0]).values())[0]
        elif list(json.loads(ls[i][0]).keys())[0] == 'flash' and target_pos != None:
            if list(json.loads(ls[i][0]).values())[0][0] == target_pos:
                scalar.data = 1
                task_series_list.append(1)
            else:
                scalar.data = 0
                task_series_list.append(0)
            
            labels.enqueue(scalar)

    # Remove all markers that are not 'flash'
    i = 0
    l = len(ls)
    same = False
    out_of_range = False
    offline_data_src.trial_data['Markers']['time_series'] = np.squeeze(offline_data_src.trial_data['Markers']['time_series'])
    while i < l:
        if list(json.loads(offline_data_src.trial_data['Markers']['time_series'][i]).keys())[0] != 'flash':
            offline_data_src.trial_data['Markers']['time_series'] = np.delete(offline_data_src.trial_data['Markers']['time_series'], [i], axis=0)
            offline_data_src.trial_data['Markers']['time_stamps'] = np.delete(offline_data_src.trial_data['Markers']['time_stamps'], [i], axis=0)
            l -= 1
        else:
            i += 1

    # Convert flash markers to target/non-target labels
    for i in range(len(offline_data_src.trial_data['Markers']['time_series'])):    
        if task_series_list[i] == 1:
            offline_data_src.trial_data['Markers']['time_series'][i] = json.dumps({'target': 1})
        elif task_series_list[i] == 0:
            offline_data_src.trial_data['Markers']['time_series'][i] = json.dumps({'non-target': 0})



    # Data input tensors
    training_data = {'data'   : bcipy.CircleBuffer.create(sess, len(task_series_list), bcipy.Tensor(sess, (offline_input_data.shape[0],resample_fs),None,False,None)),
                     'labels' : labels}

    # online graph data containers (i.e. graph edges)
    pred_probs = bcipy.Tensor.create_from_data(sess, (1,2), np.zeros((1,2))) 
    pred_label = bcipy.Scalar.create_from_value(sess,-1) 

    t_virt = [bcipy.Tensor.create_virtual(sess), # output of filter, input to resample
              bcipy.Tensor.create_virtual(sess), # output of resample, input to extract
              bcipy.Tensor.create_virtual(sess), # output of extract, input to xdawn
              bcipy.Tensor.create_virtual(sess), # output of xdawn, input to tangent space
              bcipy.Tensor.create_virtual(sess)] # output of tangent space, input to classifier

    classifier = bcipy.Classifier.create_logistic_regression(sess)

    start_time = 0
    end_time = 1
    extract_indices = [":", # all channels
                       [_ for _ in range(int(start_time*resample_fs),int(end_time*resample_fs))] # central 1s
                      ]

    bcipy.kernels.FilterKernel.add_filter_node(training_graph, offline_input_data, f, t_virt[0])
    bcipy.kernels.ResampleKernel.add_resample_node(training_graph, t_virt[0], resample_fs/Fs, t_virt[1])
    bcipy.kernels.ExtractKernel.add_extract_node(training_graph, t_virt[1], extract_indices, t_virt[2])

    # Enqueue training data and labels to appropriate circle buffers
    # Training data is the result of the tangent space transform
    # Training label is the result of the classifier
    bcipy.kernels.EnqueueKernel.add_enqueue_node(training_graph, t_virt[2], training_data['data'])

    # online graph nodes 
    bcipy.kernels.FilterKernel.add_filter_node(online_graph, online_input_data, f, t_virt[0])
    bcipy.kernels.ResampleKernel.add_resample_node(online_graph, t_virt[0], resample_fs/Fs, t_virt[1])
    bcipy.kernels.ExtractKernel.add_extract_node(online_graph, t_virt[1], extract_indices, t_virt[2])
    bcipy.kernels.XDawnCovarianceKernel.add_xdawn_covariance_node(online_graph, t_virt[2], 
                                                    t_virt[3], training_data['data'], 
                                                    training_data['labels'])
    bcipy.kernels.TangentSpaceKernel.add_tangent_space_node(online_graph, t_virt[3], t_virt[4])
    bcipy.kernels.ClassifierKernel.add_classifier_node(online_graph, t_virt[4], classifier, pred_label, pred_probs)

    # verify the session (i.e. schedule the nodes)


     # verify the session (i.e. schedule the nodes)
    sts = training_graph.verify()
    if sts != bcipy.BcipEnums.SUCCESS:
        print("Training verification failed")
        return sts

    task_dict = {0: 'non-target', 1: 'target'}

    for t_num in range(len(task_series_list)):
        sts = training_graph.execute(task_dict[task_series_list[t_num]])
        #print(offline_input_data.data[0,0:10])
        if sts != bcipy.BcipEnums.SUCCESS:
            return

    verify_sts = online_graph.verify()

    if verify_sts != bcipy.BcipEnums.SUCCESS:
        print("Test Failed D=")
        return verify_sts


    # initialize the classifiers (i.e., train the classifier)
    init_sts = online_graph.initialize()

    if init_sts != bcipy.BcipEnums.SUCCESS:
        print("Init Failed D=")
        return init_sts

    # Run the online trials
    sts = bcipy.BcipEnums.SUCCESS
    online_trials = 1000

    # TODO add LSL output and loop to wait for marker

    for t_num in range(online_trials):
        sts = online_graph.execute()
        if sts == bcipy.BcipEnums.SUCCESS:
            # print the value of the most recent trial
            y_bar = pred_probs.data[0]
            print(f"\tTrial {t_num+1}: Probabilities = {y_bar}")
        else:
            print(f"Trial {t_num+1} raised error, status code: {sts}")
            return

    print("Test Passed =D")

if __name__ == "__main__":
    files = ["C:/Users/lioa/Documents/Mindset_Data/data/sub-P003/sourcedata/sub-P003_ses-S001_task-vP300+2x2_run-001.xdf",
            "C:/Users/lioa/Documents/Mindset_Data/data/sub-P003/sourcedata/sub-P003_ses-S001_task-vP300+2x2_run-002.xdf",
            "C:/Users/lioa/Documents/Mindset_Data/data/sub-P003/sourcedata/sub-P003_ses-S001_task-vP300+2x2_run-003.xdf",
            "C:/Users/lioa/Documents/Mindset_Data/data/sub-P003/sourcedata/sub-P003_ses-S001_task-vP300+2x2_run-004.xdf",
            "C:/Users/lioa/Documents/Mindset_Data/data/sub-P003/sourcedata/sub-P003_ses-S001_task-vP300+2x2_run-005.xdf",
            "C:/Users/lioa/Documents/Mindset_Data/data/sub-P003/sourcedata/sub-P003_ses-S001_task-vP300+2x2_run-006.xdf"]
    #files = "C:/Users/lioa/Documents/Mindset_Data/data/sub-P003/sourcedata/sub-P003_ses-S001_task-vP300+2x2_run-006.xdf"
    main(files)