# -*- coding: utf-8 -*-
"""
Created on Tues July 26 16:12:30 2022

This is a test script for the P300 dataset from Kaggle. The goal is to test the initialization accuracy of the classifier
on a large scale dataset.
"""

# Create a simple graph for testing
import mindpype as mp
import numpy as np
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from copy import deepcopy


def main():
    # Create a session and graph
    sess = mp.Session.create()
    online_graph = mp.Graph.create(sess)

    # Cosntants
    Fs = 250
    resample_fs = 50

    # create a filter
    f = mp.Filter.create_fir(sess, fs=Fs, low_freq=1, high_freq=25, method='fir', fir_design='firwin', phase='minimum')

    # Load pickled kaggle data
    data_dict = pickle.load(open(r"tutorials/P300_Kaggle_dataset.pickle", "rb"))

    offline_data = data_dict['train'].get_data()
    online_data = data_dict['test'].get_data()
    offline_labels = data_dict['train_targets']
    online_labels = data_dict['test_targets']
    online_tensor = mp.Tensor.create_from_data(sess, online_data)
    offline_tensor = mp.Tensor.create_from_data(sess, offline_data)
    online_labels = mp.Tensor.create_from_data(sess, online_labels)
    offline_labels = mp.Tensor.create_from_data(sess, offline_labels)


    # online graph data containers (i.e. graph edges)
    Ntrials = online_data.shape[0]
    pred_probs = mp.Tensor.create(sess, shape=(Ntrials, 2)) # output of classifier, input to label
    pred_label = mp.Tensor.create(sess, shape=(Ntrials,))

    t_virt = [mp.Tensor.create_virtual(sess), # output of filter, input to resample
              mp.Tensor.create_virtual(sess), # output of resample, input to extract
              mp.Tensor.create_virtual(sess), # output of extract, input to xdawn
              mp.Tensor.create_virtual(sess), # output of xdawn, input to tangent space
              mp.Tensor.create_virtual(sess),  # output of tangent space, input to classifier
              mp.Tensor.create_virtual(sess),
              mp.Tensor.create_virtual(sess),
              mp.Tensor.create_virtual(sess)]

    start_time = 0
    end_time = 1.0
    extract_indices = [slice(None),
                       slice(None),
                       slice(int(start_time*Fs + len(f.coeffs['fir'])), int(np.ceil(end_time*Fs + len(f.coeffs['fir']))))
    ]# All epochs, all channels, start_time to end_time

    classifier = mp.Classifier.create_logistic_regression(sess)

    mp.kernels.PadKernel.add_to_graph(online_graph, online_tensor, t_virt[0], pad_width=((0,0), (0,0), (len(f.coeffs['fir']), len(f.coeffs['fir']))), mode='edge')
    mp.kernels.FilterKernel.add_to_graph(online_graph, t_virt[0], f, t_virt[1], axis=2)

    mp.kernels.ExtractKernel.add_to_graph(online_graph, t_virt[1], extract_indices, t_virt[2])
    mp.kernels.BaselineCorrectionKernel.add_to_graph(online_graph, t_virt[2], t_virt[4], baseline_period=[0*Fs, 0.2*Fs])

    mp.kernels.ResampleKernel.add_to_graph(online_graph, t_virt[4], resample_fs/Fs, t_virt[5], axis=2)
    mp.kernels.XDawnCovarianceKernel.add_to_graph(online_graph, t_virt[5], t_virt[6], num_filters=4, estimator="lwf", xdawn_estimator="lwf")
    mp.kernels.TangentSpaceKernel.add_to_graph(online_graph, t_virt[6], t_virt[7], metric="riemann")
    node_9 = mp.kernels.ClassifierKernel.add_to_graph(online_graph, t_virt[7], classifier , pred_label, pred_probs)

    online_graph.set_default_init_data(offline_tensor, offline_labels)

    online_graph.verify()
    # initialize the classifiers (i.e., train the classifier)
    online_graph.initialize()

    init_probs = node_9._kernel.init_outputs[1].data

    online_graph.execute()

    print(f"Probabilities = {pred_probs.data}")

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

