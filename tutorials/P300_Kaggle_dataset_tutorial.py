# -*- coding: utf-8 -*-
"""
Created on Tues July 26 16:12:30 2022
@author: aaronlio

This is a test script for the P300 dataset from Kaggle. The goal is to test the initialization accuracy of the classifier
on a large scale dataset.
"""

# Create a simple graph for testing
import bcipy.bcipy as bcipy
import numpy as np
from datetime import datetime
import json, pickle
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
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
    

def main():
    # Create a session and graph   
    sess = bcipy.Session.create()
    online_graph = bcipy.Graph.create(sess)

    # Cosntants
    Fs = 250
    trial_len = 1.4
    resample_fs = 50

    # create a filter
    f = bcipy.Filter.create_fir(sess, fs=Fs, low_freq=1, high_freq=25, method='fir', fir_design='firwin', phase='minimum')
    channels = tuple([_ for _ in range(0,8)])

    # Load pickled kaggle data
    data_dict = pickle.load(open(r"C:\Users\lioa\Documents\mindset_testing\kaggle_data.p", "rb"))

    offline_data = data_dict['train'].get_data()
    online_data = data_dict['test'].get_data()
    offline_labels = data_dict['train_targets']
    online_labels = data_dict['test_targets']
    print(online_labels.shape)
    print(online_data.shape)
    print(offline_labels.shape)
    print(offline_data.shape)
    #offline_data_src.trial_data['EEG']['time_series']['flash'] = offline_data_src.trial_data['EEG']['time_series']['flash']
    online_tensor = bcipy.containers.Tensor.create_from_data(sess, online_data.shape, online_data)
    offline_tensor = bcipy.containers.Tensor.create_from_data(sess, offline_data.shape, offline_data)
    online_labels = bcipy.containers.Tensor.create_from_data(sess, online_labels.shape, online_labels)
    offline_labels = bcipy.containers.Tensor.create_from_data(sess, offline_labels.shape, offline_labels)
    

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

    start_time = 0
    end_time = 1.0
    extract_indices = [":", ":", [_ for _ in range(int(start_time*Fs + len(f.coeffs['fir'])),int(np.ceil(end_time*Fs + len(f.coeffs['fir']))))]]# All epochs, all channels, start_time to end_time
    
    classifier = bcipy.Classifier.create_logistic_regression(sess)
   
    node_1 = bcipy.kernels.PadKernel.add_pad_node(online_graph, online_tensor, t_virt[0], pad_width=((0,0), (0,0), (len(f.coeffs['fir']), len(f.coeffs['fir']))), mode='edge')
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
    if online_graph.initialize(offline_tensor, offline_labels) != bcipy.BcipEnums.SUCCESS:
        
        print("Init Failed D=")
        return bcipy.BcipEnums.INITIALIZATION_FAILURE
    

    init_probs = node_9._kernel.init_outputs[1].data
    
    pickle.dump(init_probs, open("init_probs.pkl", "wb"))
    
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
    num_ones_online = 0
    num_ones_init = 0

    pred_probs_online = np.argmax(probs, axis=1)
    pred_probs_init = np.argmax(init_probs, axis=1)

    for i in range(len(probs)-1):
        if np.argmax(probs[i]) == online_labels.data[i]:
            correct += 1

        if online_labels.data[i] == 1:
            num_ones_online += 1

    for i in range(len(init_probs)-1):
        if np.argmax(init_probs[i]) == offline_labels.data[i]:
            init_correct += 1
        if offline_labels.data[i] == 1:
            num_ones_init += 1

    
    train_C_accuracy = init_correct/len(init_probs)
    print(train_C_accuracy, num_ones_init/len(init_probs))

    print(f1_score(online_labels.data, pred_probs_online))
    print(f1_score(offline_labels.data, pred_probs_init))

    online_C_A = correct/len(probs)
    print(online_C_A, num_ones_online/len(probs))
    
if __name__ == "__main__":
    main()

