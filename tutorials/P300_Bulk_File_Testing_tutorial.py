# -*- coding: utf-8 -*-
"""
Created on Tues July 26 16:12:30 2022
@author: aaronlio

This file is used to test the P300 graph on offline training and testing data.
This can be used to test the accuracy of the P300 graph following changes to any of its components.

Change the file paths, sampling frequency (or Ns per trial), and channels to test the graph on different datasets.
"""

# Create a simple graph for testing
import bcipy.bcipy as bcipy
import numpy as np
from datetime import datetime
import json, pickle
import matplotlib.pyplot as plt
from copy import deepcopy



def plot_func(data, name, tnum, Fs):
    x = np.arange(0, 1.4, 1/Fs)
    fig = plt.figure()
    plt.plot(data.data[0,:])
    plt.savefig(f'./images/{name}_{tnum}.png')
    plt.close(fig)


def plot_func2(data, name, tnum, Fs):
    x = np.arange(0, 1.4, 1/Fs)
    for i in range(data.shape[0]):
        fig = plt.figure()
        plt.plot(data.data[i,:])
        plt.savefig(f'./images/{name}_{tnum}_{i}.png')
        plt.close(fig)
    

def main(file):
    # Create a session and graph   
    sess = bcipy.Session.create()
    online_graph = bcipy.Graph.create(sess)

    # Cosntants
    Fs = 500
    channels = tuple([_ for _ in range(32)])
    resample_fs = 50

    # create a filter
    f = bcipy.Filter.create_fir(sess, fs=Fs, low_freq=1, high_freq=25, method='fir', fir_design='firwin', phase='minimum')
    

    # Data sources from LSL
    offline_data_src = bcipy.source.BcipXDF.create_class_separated(sess, r"c:\Users\lioa\Downloads\sub-P001_ses-S001_task-vP300+2x2_run-039.xdf", ['flash'], channels=channels, relative_start=-0.2, Ns = np.ceil(Fs))

    # training data sources from XDF file
    online_data_src = bcipy.source.BcipXDF.create_class_separated(sess, r"c:\Users\lioa\Downloads\sub-P001_ses-S001_task-vP300+2x2_run-039.xdf", ['flash'], channels=channels, relative_start=-0.2, Ns = np.ceil(Fs)) 

    #offline_data_src.trial_data['EEG']['time_series']['flash'] = offline_data_src.trial_data['EEG']['time_series']['flash']
    online_xdf_tensor = bcipy.containers.Tensor.create_from_data(sess, online_data_src.trial_data['EEG']['time_series']['flash'].shape, online_data_src.trial_data['EEG']['time_series']['flash'])
    offline_xdf_tensor = bcipy.containers.Tensor.create_from_data(sess, offline_data_src.trial_data['EEG']['time_series']['flash'].shape, offline_data_src.trial_data['EEG']['time_series']['flash'])
    
    print(offline_xdf_tensor.shape, online_xdf_tensor.shape)

    # Create circle buffer to store true init_labels
    init_labels = bcipy.CircleBuffer.create(sess, len(offline_data_src.trial_data['Markers']['time_series']), bcipy.Scalar.create(sess, int))

    ls = offline_data_src.trial_data['Markers']['time_series']
    target_pos = None
    task_series_list = []
    # Convert string markers to integer init_labels
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
            
            init_labels.enqueue(scalar)

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

    # Convert flash markers to target/non-target init_labels
    for i in range(len(offline_data_src.trial_data['Markers']['time_series'])):    
        if task_series_list[i] == 1:
            offline_data_src.trial_data['Markers']['time_series'][i] = json.dumps({'target': 1})
        elif task_series_list[i] == 0:
            offline_data_src.trial_data['Markers']['time_series'][i] = json.dumps({'non-target': 0})

    online_labels = bcipy.CircleBuffer.create(sess, len(online_data_src.trial_data['Markers']['time_series']), bcipy.Scalar.create(sess, int))

    ls = online_data_src.trial_data['Markers']['time_series']
    target_pos = None
    task_series_list = []
    # Convert string markers to integer init_labels
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
            
            online_labels.enqueue(scalar)

    # Remove all markers that are not 'flash'
    i = 0
    l = len(ls)
    
    online_data_src.trial_data['Markers']['time_series'] = np.squeeze(online_data_src.trial_data['Markers']['time_series'])
    while i < l:
        if list(json.loads(online_data_src.trial_data['Markers']['time_series'][i]).keys())[0] != 'flash':
            online_data_src.trial_data['Markers']['time_series'] = np.delete(online_data_src.trial_data['Markers']['time_series'], [i], axis=0)
            online_data_src.trial_data['Markers']['time_stamps'] = np.delete(online_data_src.trial_data['Markers']['time_stamps'], [i], axis=0)
            l -= 1
        else:
            i += 1

    # Convert flash markers to target/non-target init_labels
    for i in range(len(online_data_src.trial_data['Markers']['time_series'])):    
        if task_series_list[i] == 1:
            online_data_src.trial_data['Markers']['time_series'][i] = json.dumps({'target': 1})
        elif task_series_list[i] == 0:
            online_data_src.trial_data['Markers']['time_series'][i] = json.dumps({'non-target': 0})


    # online graph data containers (i.e. graph edges)
    pred_probs = bcipy.Tensor.create_virtual(sess) # output of classifier, input to label
    pred_label = bcipy.Tensor.create_virtual(sess) 

    t_virt = [bcipy.Tensor.create_virtual(sess), # output of filter, input to resample
              bcipy.Tensor.create_virtual(sess), # output of resample, input to extract
              bcipy.Tensor.create_virtual(sess), # output of extract, input to xdawn
              bcipy.Tensor.create_virtual(sess), # output of xdawn, input to tangent space
              bcipy.Tensor.create_virtual(sess),  # output of tangent space, input to classifier
              bcipy.Tensor.create_virtual(sess),
              bcipy.Tensor.create_virtual(sess),
              bcipy.Tensor.create_virtual(sess)]

    start_time = 0.0
    end_time = 1
    extract_indices = [":", ":", [_ for _ in range(int(start_time*Fs + len(f.coeffs['fir'])),int(np.ceil(end_time*Fs + len(f.coeffs['fir']))))]]# All epochs, all channels, start_time to end_time
    print(min(extract_indices[2]), max(extract_indices[2]))

    classifier = bcipy.Classifier.create_logistic_regression(sess)
   
    node_1 = bcipy.kernels.PadKernel.add_pad_node(online_graph, online_xdf_tensor, t_virt[0], pad_width=((0,0), (0,0), (len(f.coeffs['fir']), len(f.coeffs['fir']))), mode='edge')
    node_2 = bcipy.kernels.FilterKernel.add_filter_node(online_graph, t_virt[0], f, t_virt[1], axis=2)

    node_3 = bcipy.kernels.ExtractKernel.add_extract_node(online_graph, t_virt[1], extract_indices, t_virt[2])
    node_4 = bcipy.kernels.BaselineCorrectionKernel.add_baseline_node(online_graph, t_virt[2], t_virt[4], baseline_period=[0*Fs, 0.2*Fs])

    node_6 = bcipy.kernels.ResampleKernel.add_resample_node(online_graph, t_virt[4], resample_fs/Fs, t_virt[5], axis=2)
    node_7 = bcipy.kernels.XDawnCovarianceKernel.add_xdawn_covariance_node(online_graph, t_virt[5], t_virt[6], num_filters=4, estimator="lwf", xdawn_estimator="lwf")
    node_8 = bcipy.kernels.TangentSpaceKernel.add_tangent_space_node(online_graph, t_virt[6], t_virt[7], metric="riemann")
    node_9 = bcipy.kernels.ClassifierKernel.add_classifier_node(online_graph, t_virt[7], classifier , pred_label, pred_probs)

    if online_graph.verify() != bcipy.BcipEnums.SUCCESS:
        print("Test Failed D=")
        return bcipy.BcipEnums.INVALID_GRAPH

    # initialize the classifiers (i.e., train the classifier)
    if online_graph.initialize(offline_xdf_tensor, init_labels) != bcipy.BcipEnums.SUCCESS:
        
        print("Init Failed D=")
        return bcipy.BcipEnums.INITIALIZATION_FAILURE
    

    init_probs = node_9._kernel.init_outputs[1].data
    
    #pickle.dump(init_probs, open("init_probs.pkl", "wb"))
    
    sts = online_graph.execute()

    if sts == bcipy.BcipEnums.SUCCESS:
        # print the value of the most recent trial
        print(f"Probabilities = {pred_probs.data}")
    else:
        print(f"fAIL")

    
    probs = pred_probs.data
    pickle.dump(probs, open("probs.pkl", "wb"))

    # Compute initialization classification accuracy
    correct = 0
    init_correct = 0
    
    init_labels = init_labels.to_tensor().data
    online_labels = online_labels.to_tensor().data
    
    for i in range(len(probs)):
        if np.argmax(probs[i]) == online_labels[i]:
            correct += 1
        if np.argmax(init_probs[i]) == init_labels[i]:
            init_correct += 1

    train_C_accuracy = init_correct/len(init_probs)
    print(train_C_accuracy)

    online_C_A = correct/len(probs)
    print(online_C_A)
    
if __name__ == "__main__":
    #files = ["C:/Users/lioa/Documents/Mindset_Data/data/sub-P003/sourcedata/sub-P003_ses-S002_task-vP300+2x2_run-001.xdf",
    #         "C:/Users/lioa/Documents/Mindset_Data/data/sub-P003/sourcedata/sub-P003_ses-S002_task-vP300+2x2_run-002.xdf"]
    files = [r'c:\Users\lioa\Documents\Mindset_Data\data\sub-P004\sourcedata\sub-P004_ses-S001_task-vP300+2x2_run-001.xdf']
    #files = ["C:/Users/student_admin.PRISMLAB/Documents/Mindset_Data/data/sub-P002/sourcedata/sub-P002_ses-S001_task-vP300+2x2_run-001.xdf",
    #         "C:/Users/student_admin.PRISMLAB/Documents/Mindset_Data/data/sub-P002/sourcedata/sub-P002_ses-S001_task-vP300+2x2_run-002.xdf",
    #         "C:/Users/student_admin.PRISMLAB/Documents/Mindset_Data/data/sub-P002/sourcedata/sub-P002_ses-S001_task-vP300+2x2_run-003.xdf",
    #         "C:/Users/student_admin.PRISMLAB/Documents/Mindset_Data/data/sub-P002/sourcedata/sub-P002_ses-S001_task-vP300+2x2_run-004.xdf"]
    main(files)

