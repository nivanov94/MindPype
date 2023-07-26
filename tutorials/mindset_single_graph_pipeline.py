# -*- coding: utf-8 -*-
"""
Created on Tues July 26 16:12:30 2022
@author: aaronlio
"""

# Create a simple graph for testing
import bcipy.bcipy as bcipy
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt

def main(file):
    # create a session
    sess = bcipy.Session.create()
    online_graph = bcipy.Graph.create(sess)

    # Cosntants
    Fs = 128
    trial_len = 1.4
    resample_fs = 50

    # create a filter
    f = bcipy.Filter.create_fir(sess, fs=Fs, low_freq=1, high_freq=25, method='fir', fir_design='firwin', phase='minimum')
    channels = tuple([_ for _ in range(3,17)])

    # Data sources from LSL
    LSL_data_src = bcipy.source.InputLSLStream.create_marker_coupled_data_stream(sess, "type='EEG'", channels, relative_start=-0.4, marker_fmt='.*flash', marker_pred="type='Marker'") # type: ignore
    
    # training data sources from XDF file
    offline_data_src = bcipy.source.BcipXDF.create_class_separated(sess, file, ['flash'], channels=channels, relative_start=-.4, Ns = np.ceil(Fs*trial_len)) 

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

    # online graph data containers (i.e. graph edges)
    pred_probs = bcipy.Tensor.create_virtual(sess) # output of classifier, input to label
    pred_label = bcipy.Tensor.create_virtual(sess) 

    t_virt = [bcipy.Tensor.create_virtual(sess), # output of filter, input to resample
              bcipy.Tensor.create_virtual(sess), # output of resample, input to extract
              bcipy.Tensor.create_virtual(sess), # output of extract, input to xdawn
              bcipy.Tensor.create_virtual(sess), # output of xdawn, input to tangent space
              bcipy.Tensor.create_virtual(sess),  # output of tangent space, input to classifier
              bcipy.Tensor.create_virtual(sess)]
    
    start_time = -0.2
    end_time = 0.8
        
    extract_indices = [":", [_ for _ in range(int(start_time*resample_fs),int(end_time*resample_fs))]]# All epochs, all channels, start_time to end_time

    classifier = bcipy.Classifier.create_logistic_regression(sess)
    #bcipy.kernels.FiltFiltKernel.add_filtfilt_node(online_graph, online_input_data, f1, t_virt[0], axis=1)
    bcipy.kernels.PadKernel.add_pad_node(online_graph, online_input_data, t_virt[0], pad_width=((0,0), (20, 20)), constant_values=0)
    bcipy.kernels.FilterKernel.add_filter_node(online_graph, t_virt[0], f, t_virt[1], axis=1)
    bcipy.kernels.ResampleKernel.add_resample_node(online_graph, t_virt[1], resample_fs/Fs, t_virt[2], axis=1)
    bcipy.kernels.ExtractKernel.add_extract_node(online_graph, t_virt[2], extract_indices, t_virt[3])
    bcipy.kernels.XDawnCovarianceKernel.add_xdawn_covariance_node(online_graph, t_virt[3], t_virt[4])
    bcipy.kernels.TangentSpaceKernel.add_tangent_space_node(online_graph, t_virt[4], t_virt[5])
    bcipy.kernels.ClassifierKernel.add_classifier_node(online_graph, t_virt[5], classifier , pred_label, pred_probs)

    if online_graph.verify() != bcipy.BcipEnums.SUCCESS:
        print("Test Failed D=")
        return bcipy.BcipEnums.INVALID_GRAPH

    # initialize the classifiers (i.e., train the classifier)
    if online_graph.initialize(xdf_tensor, labels) != bcipy.BcipEnums.SUCCESS:
        print("Init Failed D=")
        return bcipy.BcipEnums.INITIALIZATION_FAILURE

    # Run the online trials
    for t_num in range(1000):
        sts = online_graph.execute()

        if sts == bcipy.BcipEnums.SUCCESS:
            # print the value of the most recent trial
            print(f"\t{datetime.utcnow()}: Probabilities = {pred_probs.data}")
        else:
            print(f"Trial {t_num+1} raised error, status code: {sts}")
  
    print("Test Passed =D")

if __name__ == "__main__":
    #files = ["C:/Users/lioa/Documents/Mindset_Data/data/sub-P003/sourcedata/sub-P003_ses-S002_task-vP300+2x2_run-001.xdf",
    #         "C:/Users/lioa/Documents/Mindset_Data/data/sub-P003/sourcedata/sub-P003_ses-S002_task-vP300+2x2_run-002.xdf"]
    files = [r'c:\Users\lioa\Documents\Mindset_Data\data\sub-P004\sourcedata\sub-P004_ses-S001_task-vP300+2x2_run-001.xdf']
    main(files)