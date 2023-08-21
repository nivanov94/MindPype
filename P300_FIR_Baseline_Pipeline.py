# -*- coding: utf-8 -*-
"""
Created on Tues July 26 16:12:30 2022
@author: aaronlio
"""

# Create a simple graph for testing
#from bcipy import bcipy
import bcipy
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt



def main():
    # create a session
    sess = bcipy.Session.create()
    online_graph = bcipy.Graph.create(sess)
    training_graph = bcipy.Graph.create(sess)
    preprocessing_graph = bcipy.Graph.create(sess)

    # Cosntants
    Fs = 128
    trial_len = 1.4
    tasks = ('non-target', 'target')
    resample_fs = 50

    
    # create a filter
    order = 4
    bandpass = [1,25] # in Hz
    f = bcipy.Filter.create_fir(sess, fs=Fs, low_freq=1, high_freq=25, method='fir', fir_design='firwin', phase='minimum')
    f1 = bcipy.Filter.create_butter(sess, order, bandpass, 'bandpass', implementation='sos', fs=Fs)
    channels = tuple([_ for _ in range(3,17)])

    # Data sources from LSL
    print("Session Starting...")
    #LSL_data_src = bcipy.source.InputLSLStream.create_marker_coupled_data_stream(sess, "type='EEG'", channels, relative_start=-0.4, marker_fmt='.*flash', marker_pred="type='Marker'")
    
    # training data sources from XDF file
    #offline_data_src = bcipy.source.BcipXDF.create_class_separated(sess,
    #        file, 
    #        ['flash'], channels=channels, relative_start=-.4, Ns = np.ceil(Fs*trial_len)) 

    #offline_data_src.trial_data['EEG']['time_series']['flash'] = offline_data_src.trial_data['EEG']['time_series']['flash']
    #xdf_tensor = bcipy.containers.Tensor.create_from_data(sess, offline_data_src.trial_data['EEG']['time_series']['flash'].shape, offline_data_src.trial_data['EEG']['time_series']['flash'])
    xdf_tensor = bcipy.Tensor.create_from_data(sess, (100, len(channels), 180), np.random.rand(100, len(channels), 180))
    

    # Create input tensors
    #online_input_data = bcipy.Tensor.create_from_handle(sess, (len(channels), 180), LSL_data_src)
    online_input_data = bcipy.Tensor.create(sess, (len(channels), 180))
    #offline_input_data = bcipy.Tensor.create_from_handle(sess, (len(channels), 180), offline_data_src)

    # Create circle buffer to store true labels
    labels = bcipy.Tensor.create_from_data(sess, (100,), np.random.randint(0,2,100))

    #ls = offline_data_src.trial_data['Markers']['time_series']
    #target_pos = None
    #task_series_list = []
    ## Convert string markers to integer labels
    #for i in range(len(ls)):
    #    scalar = bcipy.Scalar.create(sess, int)
    #    if list(json.loads(ls[i][0]).keys())[0] == 'target':
    #        target_pos = list(json.loads(ls[i][0]).values())[0]
    #    elif list(json.loads(ls[i][0]).keys())[0] == 'flash' and target_pos != None:
    #        if list(json.loads(ls[i][0]).values())[0][0] == target_pos:
    #            scalar.data = 1
    #            task_series_list.append(1)
    #        else:
    #            scalar.data = 0
    #            task_series_list.append(0)
    #        
    #        labels.enqueue(scalar)

    ## Remove all markers that are not 'flash'
    #i = 0
    #l = len(ls)
    
    #offline_data_src.trial_data['Markers']['time_series'] = np.squeeze(offline_data_src.trial_data['Markers']['time_series'])
    #while i < l:
    #    if list(json.loads(offline_data_src.trial_data['Markers']['time_series'][i]).keys())[0] != 'flash':
    #        offline_data_src.trial_data['Markers']['time_series'] = np.delete(offline_data_src.trial_data['Markers']['time_series'], [i], axis=0)
    #        offline_data_src.trial_data['Markers']['time_stamps'] = np.delete(offline_data_src.trial_data['Markers']['time_stamps'], [i], axis=0)
    #        l -= 1
    #    else:
    #        i += 1

    ## Convert flash markers to target/non-target labels
    #for i in range(len(offline_data_src.trial_data['Markers']['time_series'])):    
    #    if task_series_list[i] == 1:
    #        offline_data_src.trial_data['Markers']['time_series'][i] = json.dumps({'target': 1})
    #    elif task_series_list[i] == 0:
    #        offline_data_src.trial_data['Markers']['time_series'][i] = json.dumps({'non-target': 0})

    train_virt2 = bcipy.Tensor.create_virtual(sess) # output of filter, input to resample
    processed_xdf_tensor = bcipy.Tensor.create_virtual(sess)
    

    # Data input tensors
    training_data = {'data'   : processed_xdf_tensor,
                     'labels' : labels}

    # online graph data containers (i.e. graph edges)
    pred_probs = bcipy.Tensor.create_virtual(sess) # output of classifier, input to label
    pred_label = bcipy.Tensor.create_virtual(sess) 

    train_virt = bcipy.Tensor.create_virtual(sess) # output of filter, input to resample
    t_virt = [bcipy.Tensor.create_virtual(sess), # output of filter, input to resample
              bcipy.Tensor.create_virtual(sess), # output of resample, input to extract
              bcipy.Tensor.create_virtual(sess), # output of extract, input to xdawn
              bcipy.Tensor.create_virtual(sess), # output of xdawn, input to tangent space
              bcipy.Tensor.create_virtual(sess)] # output of tangent space, input to classifier
    
    extra_tensor = bcipy.Tensor.create_virtual(sess)

    classifier = bcipy.Classifier.create_logistic_regression(sess)

    start_time = 0.2
    end_time = 1.2
    
    extract_indices = [":", # all channels
                       [_ for _ in range(int(start_time*resample_fs),int(end_time*resample_fs))]] # central 1s
    
    extract_indices1 = [":", # all epochs
                        ":", # all channels
                       [_ for _ in range(int(start_time*resample_fs),int(end_time*resample_fs))] # central 1s
                      ]
    
    extract_indices2 = [":", # all epochs
                        ":", # all channels
                       [_ for _ in range(int(-.2*resample_fs),int(0*resample_fs))]
                      ]
    

    preprocessing_tensors = [bcipy.Tensor.create_virtual(sess), # output of baseline extraction,
                              bcipy.Tensor.create_virtual(sess)]

    

    #bcipy.kernels.FiltFiltKernel.add_filtfilt_node(preprocessing_graph, xdf_tensor, f1, extra_tensor, axis = 2)
    bcipy.kernels.FilterKernel.add_filter_node(preprocessing_graph, xdf_tensor, f, extra_tensor, axis = 2)
    bcipy.kernels.ExtractKernel.add_extract_node(preprocessing_graph, extra_tensor, extract_indices2, preprocessing_tensors[0])
    bcipy.kernels.MeanKernel.add_mean_node(preprocessing_graph, preprocessing_tensors[0],  outA = preprocessing_tensors[1], axis=2, keepdims=True)

    preprocessing_graph.verify()
    preprocessing_graph.execute()
    
    mean = preprocessing_tensors[1].data # mean of baseline
    repeated_mean = np.repeat(mean, 180, axis=2) # repeat mean for each epoch

    mean_tensor = bcipy.Tensor.create_from_data(sess, repeated_mean.shape, repeated_mean)

    bcipy.kernels.SubtractionKernel.add_subtraction_node(training_graph, extra_tensor, mean_tensor, train_virt)
    bcipy.kernels.ResampleKernel.add_resample_node(training_graph, train_virt, resample_fs/Fs, train_virt2, axis = 2)
    bcipy.kernels.ExtractKernel.add_extract_node(training_graph, train_virt2, extract_indices1, processed_xdf_tensor)

    # Enqueue training data and labels to appropriate circle buffers
    # Training data is the result of the tangent space transform
    # Training label is the result of the classifier
    #bcipy.kernels.EnqueueKernel.add_enqueue_node(training_graph, t_virt[2], training_data['data'])

    # online graph nodes 
    #bcipy.kernels.FiltFiltKernel.add_filtfilt_node(online_graph, online_input_data, f1, t_virt[0], axis=1)
    bcipy.kernels.FilterKernel.add_filter_node(online_graph, online_input_data, f, t_virt[0], axis=1)
    bcipy.kernels.ResampleKernel.add_resample_node(online_graph, t_virt[0], resample_fs/Fs, t_virt[1], axis=1)
    bcipy.kernels.ExtractKernel.add_extract_node(online_graph, t_virt[1], extract_indices, t_virt[2])
    bcipy.kernels.XDawnCovarianceKernel.add_xdawn_covariance_node(online_graph, t_virt[2], 
                                                    t_virt[3], training_data['data'], 
                                                    training_data['labels'])
    bcipy.kernels.TangentSpaceKernel.add_tangent_space_node(online_graph, t_virt[3], t_virt[4])
    bcipy.kernels.ClassifierKernel.add_classifier_node(online_graph, t_virt[4], classifier, pred_label, pred_probs)

    # verify the session (i.e. schedule the nodes)


     # verify the session (i.e. schedule the nodes)
    training_graph.verify()
    training_graph.execute()

    online_graph.verify()

    # initialize the classifiers (i.e., train the classifier)
    online_graph.initialize()
    print(pred_probs.data)

    # Run the online trials
    online_trials = 10

    for t_num in range(online_trials):
        online_input_data.assign_random_data()
        online_graph.execute()
        # print the value of the most recent trial
        y_bar = pred_probs.data
        print(f"\t{datetime.utcnow()}: Probabilities = {y_bar}")
  


    print("Test Passed =D")

if __name__ == "__main__":
    main()