# -*- coding: utf-8 -*-
"""
Created on Tues July 26 16:12:30 2022
@author: aaronlio
"""

# Create a simple graph for testing
from bcipy import bcipy
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt



def main(file):
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
    LSL_data_src = bcipy.source.InputLSLStream.create_marker_coupled_data_stream(sess, "type='EEG'", channels, relative_start=-0.4, marker_fmt='.*flash', marker_pred="type='Marker'")
    
    # training data sources from XDF file
    offline_data_src = bcipy.source.BcipXDF.create_class_separated(sess,
            file, 
            ['flash'], channels=channels, relative_start=-.4, Ns = np.ceil(Fs*trial_len)) 

    #offline_data_src.trial_data['EEG']['time_series']['flash'] = offline_data_src.trial_data['EEG']['time_series']['flash']
    xdf_tensor = bcipy.containers.Tensor.create_from_data(sess, offline_data_src.trial_data['EEG']['time_series']['flash'].shape, offline_data_src.trial_data['EEG']['time_series']['flash'])
    

    # Create input tensors
    online_input_data = bcipy.Tensor.create_from_handle(sess, (len(channels), 180), LSL_data_src)
    #offline_input_data = bcipy.Tensor.create_from_handle(sess, (len(channels), 180), offline_data_src)

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

    start_time = -0.2
    end_time = 0.8
    
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

    if preprocessing_graph.verify() != bcipy.BcipEnums.SUCCESS:
        return bcipy.BcipEnums.FAILURE

    sts = preprocessing_graph.execute()
    
    if sts != bcipy.BcipEnums.SUCCESS:
        return sts

    # PLOT RAW DATA
    #target_average = np.zeros((xdf_tensor.shape[1],xdf_tensor.shape[2]-6))
    #nontarget_average = np.zeros((xdf_tensor.shape[1],xdf_tensor.shape[2]-6))
    ##print(train_virt.data.shape, running_average.shape)
    #for epoch in range(xdf_tensor.data.shape[0]):
    #    if task_series_list[epoch] == 1:
    #        target_average =  target_average + xdf_tensor.data[epoch,:,5:-1]
    #    else:
    #        nontarget_average = nontarget_average + xdf_tensor.data[epoch,:,5:-1]
    #
    #target_average = target_average/sum(task_series_list)
    #nontarget_average = nontarget_average/(len(task_series_list)-sum(task_series_list))
    #   
    ## Plot the filtered data from the first channel along time
#
    #x_axis = np.linspace(0, xdf_tensor.data.shape[2]/Fs, xdf_tensor.data.shape[2]-6)
#
#
    #for i in range(xdf_tensor.shape[1]-8):
    #    fig = plt.subplot(2,3,i+1)
    #    fig.plot(x_axis, nontarget_average[i,:])
    #    fig.plot(x_axis, target_average[i,:])
    #    #fig.plot(x_axis, target_average[i,:] - nontarget_average[i,:])
    #    fig.axes.set_xlabel('Time (s)')
    #    fig.axes.set_ylabel('Amplitude')
    #    fig.axes.set_xticks([0, .3, .5, 1.0])
    #
    #fig.legend(['nontarget', 'target', 'difference'])
    #
    #plt.show()

    target_average = np.zeros((extra_tensor.shape[1],extra_tensor.shape[2]-6))
    nontarget_average = np.zeros((extra_tensor.shape[1],extra_tensor.shape[2]-6))
    #print(train_virt.data.shape, running_average.shape)
    for epoch in range(extra_tensor.data.shape[0]):
        if task_series_list[epoch] == 1:
            target_average =  target_average + extra_tensor.data[epoch,:,5:-1]
        else:
            nontarget_average = nontarget_average + extra_tensor.data[epoch,:,5:-1]
    
    target_average = target_average/sum(task_series_list)
    nontarget_average = nontarget_average/(len(task_series_list)-sum(task_series_list))
       
    # Plot the filtered data from the first channel along time

    x_axis = np.linspace(0, extra_tensor.data.shape[2]/Fs, extra_tensor.data.shape[2]-6)


    for i in range(extra_tensor.shape[1]-8):
        fig = plt.subplot(2,3,i+1)
        fig.plot(x_axis, nontarget_average[i,:])
        fig.plot(x_axis, target_average[i,:])
        #fig.plot(x_axis, target_average[i,:] - nontarget_average[i,:])
        fig.axes.set_xlabel('Time (s)')
        fig.axes.set_ylabel('Amplitude')
        fig.axes.set_xticks([0, .3, .5, 1.0])
    
    fig.legend(['nontarget', 'target', 'difference'])
    
    plt.show()


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
    sts = training_graph.verify()
    if sts != bcipy.BcipEnums.SUCCESS:
        print("Training verification failed")
        return sts
    
    training_graph_sts = training_graph.execute()
    # Graph input offline input data
    #for i in range(xdf_tensor.shape[1]-8):
    #    fig = plt.subplot(2,3,i+1)
    #    fig.plot(xdf_tensor.data[0,i,:])
#
    #plt.show()

    #target_average = np.zeros((train_virt.shape[1],train_virt.shape[2]-6))
    #nontarget_average = np.zeros((train_virt.shape[1],train_virt.shape[2]-6))
    ##print(train_virt.data.shape, running_average.shape)
    #for epoch in range(train_virt.data.shape[0]):
    #    if task_series_list[epoch] == 1:
    #        target_average =  target_average + train_virt.data[epoch,:,5:-1]
    #    else:
    #        nontarget_average = nontarget_average + train_virt.data[epoch,:,5:-1]
    #
    #target_average = target_average/sum(task_series_list)
    #nontarget_average = nontarget_average/(len(task_series_list)-sum(task_series_list))
    #   
    ## Plot the filtered data from the first channel along time
#
    #x_axis = np.linspace(0, train_virt.data.shape[2]/Fs, train_virt.data.shape[2]-6)
#
#
    #for i in range(train_virt.shape[1]-8):
    #    fig = plt.subplot(2,3,i+1)
    #    fig.plot(x_axis, nontarget_average[i,:])
    #    fig.plot(x_axis, target_average[i,:])
    #    #fig.plot(x_axis, target_average[i,:] - nontarget_average[i,:])
    #    fig.axes.set_xlabel('Time (s)')
    #    fig.axes.set_ylabel('Amplitude')
    #    fig.axes.set_xticks([0, .3, .5, 1.0])
    #
    #fig.legend(['nontarget', 'target', 'difference'])
    #
    #plt.show()

#
    #target_average = np.zeros((processed_xdf_tensor.shape[1],processed_xdf_tensor.shape[2]-6))
    #nontarget_average = np.zeros((processed_xdf_tensor.shape[1],processed_xdf_tensor.shape[2]-6))
    ##print(train_virt.data.shape, running_average.shape)
    #for epoch in range(processed_xdf_tensor.data.shape[0]):
    #    if task_series_list[epoch] == 1:
    #        target_average =  target_average + processed_xdf_tensor.data[epoch,:,5:-1]
    #    else:
    #        nontarget_average = nontarget_average + processed_xdf_tensor.data[epoch,:,5:-1]
    #
    #target_average = target_average/sum(task_series_list)
    #nontarget_average = nontarget_average/(len(task_series_list)-sum(task_series_list))
    #   
    ## Plot the filtered data from the first channel along time
#
    #x_axis = np.linspace(0, processed_xdf_tensor.data.shape[2]/Fs, processed_xdf_tensor.data.shape[2]-6)
#
#
    #for i in range(processed_xdf_tensor.shape[1]-8):
    #    fig = plt.subplot(2,3,i+1)
    #    fig.plot(x_axis, nontarget_average[i,:])
    #    fig.plot(x_axis, target_average[i,:])
    #    #fig.plot(x_axis, target_average[i,:] - nontarget_average[i,:])
    #    fig.axes.set_xlabel('Time (s)')
    #    fig.axes.set_ylabel('Amplitude')
    #
    #fig.legend(['nontarget', 'target', 'difference'])
    #
    #plt.show()

    if training_graph_sts != bcipy.BcipEnums.SUCCESS:
        return
    


    verify_sts = online_graph.verify()

    if verify_sts != bcipy.BcipEnums.SUCCESS:
        print("Test Failed D=")
        return verify_sts


    # initialize the classifiers (i.e., train the classifier)
    init_sts = online_graph.initialize()
    print(pred_probs.data)
    if init_sts != bcipy.BcipEnums.SUCCESS:
        print("Init Failed D=")
        return init_sts

    # Run the online trials
    sts = bcipy.BcipEnums.SUCCESS
    online_trials = 1000


    for t_num in range(online_trials):
        sts = online_graph.execute()
        #x_dawn_filters = online_graph._nodes[3]._kernel._xdawn_estimator.Xd_.filters_   #
        #plt.imshow(x_dawn_filters)
        #plt.show()

        if sts == bcipy.BcipEnums.SUCCESS:
            # print the value of the most recent trial
            y_bar = pred_probs.data
            label = pred_label.data
            print(f"\t{datetime.utcnow()}: Probabilities = {y_bar}")
        else:
            print(f"Trial {t_num+1} raised error, status code: {sts}")
  


    print("Test Passed =D")

if __name__ == "__main__":
    #files = ["C:/Users/lioa/Documents/Mindset_Data/data/sub-P003/sourcedata/sub-P003_ses-S002_task-vP300+2x2_run-001.xdf",
    #         "C:/Users/lioa/Documents/Mindset_Data/data/sub-P003/sourcedata/sub-P003_ses-S002_task-vP300+2x2_run-002.xdf"]
    files = [r'c:\Users\lioa\Documents\Mindset_Data\data\sub-P004\sourcedata\sub-P004_ses-S001_task-vP300+2x2_run-001.xdf']
    main(files)