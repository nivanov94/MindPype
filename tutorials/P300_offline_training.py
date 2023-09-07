# -*- coding: utf-8 -*-
"""
Created on Tues July 26 16:12:30 2022
@author: aaronlio
"""

# Create a simple graph for testing
import mindpype as mp
import numpy as np
import json, pickle
from copy import deepcopy

def preprocess_labels(offline_data_src, labels):
    ls = offline_data_src.trial_data['Markers']['time_series']
    target_pos = None
    task_series_list = []
    # Convert string markers to integer labels
    for i in range(len(ls)):
        scalar = mp.Scalar.create(sess, int)
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


def main(file):
    # create a session
    sess = mp.Session.create()
    online_graph = mp.Graph.create(sess)

    # Cosntants
    Fs = 128
    resample_fs = 50

    # create a filter
    f = mp.Filter.create_fir(sess, fs=Fs, low_freq=1, high_freq=25, method='fir', fir_design='firwin', phase='minimum')
    channels = tuple([_ for _ in range(3,17)])

    # Data sources from LSL
    LSL_data_src = mp.source.InputLSLStream.create_marker_coupled_data_stream(sess, 
                                                                              "type='EEG'", 
                                                                              channels, 
                                                                              relative_start=-0.2, 
                                                                              marker_fmt='.*flash', 
                                                                              marker_pred="type='Marker'")
    
    # training data sources from XDF file
    offline_data_src = mp.source.BcipXDF.create_class_separated(sess, file, ['flash'], channels=channels, relative_start=-0.2, Ns = np.ceil(Fs)) 

    xdf_tensor = mp.containers.Tensor.create_from_data(sess, 
                                                       offline_data_src.trial_data['EEG']['time_series']['flash'].shape,
                                                       offline_data_src.trial_data['EEG']['time_series']['flash'])
    
    init_d = pickle.load(open(r'/path/to/init_data', 'rb'))[0]

    xdf_tensor = mp.containers.Tensor.create_from_data(sess, init_d.shape, init_d)

    # Create input tensors
    online_input_data = mp.Tensor.create_from_handle(sess, (len(channels), Fs), LSL_data_src)

    # Create circle buffer to store true labels
    labels = mp.CircleBuffer.create(sess, len(offline_data_src.trial_data['Markers']['time_series']), mp.Scalar.create(sess, int))


    preprocess_labels(offline_data_src, labels)

    # online graph data containers (i.e. graph edges)
    pred_probs = mp.Tensor.create_virtual(sess, shape=(1,2)) # output of classifier
    pred_label = mp.Tensor.create_virtual(sess, shape=(1,)) 

    t_virt = [mp.Tensor.create_virtual(sess), # output of filter, input to resample
              mp.Tensor.create_virtual(sess), # output of resample, input to extract
              mp.Tensor.create_virtual(sess), # output of extract, input to xdawn
              mp.Tensor.create_virtual(sess), # output of xdawn, input to tangent space
              mp.Tensor.create_virtual(sess),  # output of tangent space, input to classifier
              mp.Tensor.create_virtual(sess),
              mp.Tensor.create_virtual(sess),
              mp.Tensor.create_virtual(sess)]
    
    classifier = mp.Classifier.create_logistic_regression(sess)
    mp.kernels.FilterKernel.add_filter_node(online_graph, online_input_data, f, t_virt[1], axis=1)
    mp.kernels.BaselineCorrectionKernel.add_baseline_node(online_graph, t_virt[1], t_virt[4], baseline_period=[0*Fs, 0.2*Fs])
    mp.kernels.ResampleKernel.add_resample_node(online_graph, t_virt[4], resample_fs/Fs, t_virt[5], axis=1)
    mp.kernels.XDawnCovarianceKernel.add_xdawn_covariance_node(online_graph, t_virt[5], t_virt[6], num_filters=4, estimator="lwf", xdawn_estimator="lwf")
    mp.kernels.TangentSpaceKernel.add_tangent_space_node(online_graph, t_virt[6], t_virt[7], metric="riemann")
    mp.kernels.ClassifierKernel.add_classifier_node(online_graph, t_virt[7], classifier , pred_label, pred_probs)

    online_graph.set_default_init_data(xdf_tensor, labels)

    online_graph.verify()

    # initialize the classifiers (i.e., train the classifier)
    online_graph.initialize()
        
    # Run the online trials
    for t_num in range(100):
        try:
            online_graph.execute()
            # print the value of the most recent trial
            print(f"trial {t_num+1} Probabilities = {pred_probs.data}")
        except:
            print(f"Trial {t_num+1} raised error")
    

if __name__ == "__main__":
    files = ['path/to/data/files/']
    main(files)

