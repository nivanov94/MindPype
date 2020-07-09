# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 09:59:48 2020

@author: Nick
"""


import numpy as np
import json

from sklearn import metrics as skmetrics

def dataset_channel_map(dataset,channels):
    """
    Convert channel names to their numeric values within
    each dataset
    
    returns a set of numbers representing the channel
    indices within the dataset
    """
    if dataset == 'BCICompIV-2a':
        channel_map = {
            'Fz' : 0,
            'F3' : 1,
            'F4' : 5,
            'T7' : 6,
            'C3' : 7,
            'Cz' : 9,
            'C4' : 11,
            'T8' : 12,
            'P3' : 13,
            'P4' : 17,
            'Pz' : 19,
            'C5' : -1,
            'C6' : -1}
    elif dataset == 'Kaya':
        channel_map = {
            'Fz' : 18,
            'F3' : 2,
            'F4' : 3,
            'T7' : 14,
            'C3' : 4,
            'Cz' : 19,
            'C4' : 5,
            'T8' : 15,
            'P3' : 6,
            'P4' : 7,
            'Pz' : 20,
            'C5' : -1,
            'C6' : -1}
    elif dataset == 'HighGamma':
        channel_map = {
            'Fz' : 1,
            'F3' : 0,
            'F4' : 2,
            'T7' : 5,
            'C3' : 7,
            'Cz' : 9,
            'C4' : 11,
            'T8' : 13,
            'P3' : 17,
            'P4' : 18,
            'Pz' : 16,
            'C5' : 6,
            'C6' : 12}
    elif dataset == 'Cho':
        channel_map = {
            'Fz' : 37,
            'F3' : 4,
            'F4' : 39,
            'T7' : 14,
            'C3' : 12,
            'Cz' : 47,
            'C4' : 49,
            'T8' : 51,
            'P3' : 20,
            'P4' : 57,
            'Pz' : 30,
            'C5' : 13,
            'C6' : 50}
        
    channel_nums = set()
    for c in channels:
        channel_nums.add(channel_map[c])
    
    return channel_nums


def dataset_channel_num2str(dataset,channels):
    """
    Convert numeric channel names to their string values within
    each dataset
    
    returns a set of strings representing the channel
    indices within the dataset
    """
    if dataset == 'BCICompIV-2a':
        channel_map = {
            0  : 'Fz',
            1  : 'F3',
            5  : 'F4',
            6  : 'T7',
            7  : 'C3',
            9  : 'Cz',
            11 : 'C4',
            12 : 'T8',
            13 : 'P3',
            17 : 'P4',
            19 : 'Pz'}
    elif dataset == 'Kaya':
        channel_map = {
            18 : 'Fz',
            2  : 'F3',
            3  : 'F4',
            14 : 'T7',
            4  : 'C3',
            19 : 'Cz',
            5  : 'C4',
            15 : 'T4',
            6  : 'P3',
            7  : 'P4',
            20 : 'Pz'}
    elif dataset == 'HighGamma':
        channel_map = {
            1  : 'Fz',
            0  : 'F3',
            2  : 'F4',
            5  : 'T7',
            7  : 'C3',
            9  : 'Cz',
            11 : 'C4',
            13 : 'T8',
            17 : 'P3',
            18 : 'P4',
            16 : 'Pz',
            6  : 'C5',
            12 : 'C6'}
    elif dataset == 'Cho':
        channel_map = {
            37 : 'Fz',
            4  : 'F3',
            39 : 'F4',
            14 : 'T7',
            12 : 'C3',
            47 : 'Cz',
            49 : 'C4',
            51 : 'T8',
            20 : 'P3',
            57 : 'P4',
            30 : 'Pz',
            13 : 'C5',
            50 : 'C6'}
        
    channel_names = set()
    for c in channels:
        channel_names.add(channel_map[c])
    
    return channel_names

def reverse_confusion_mat(conf_mat):
    Nc = conf_mat.shape[0]
    Nt = np.sum(conf_mat)
    
    y = np.zeros((Nt,))
    y_bar = np.zeros((Nt,))
    
    true_sum = 0
    pred_sum = 0
    for i in range(Nc):
        Ni = np.sum(conf_mat[i,:])
        y[true_sum:true_sum+Ni] = i
        true_sum += Ni
        
        for j in range(Nc):
            Nj = conf_mat[i,j]
            y_bar[pred_sum:pred_sum+Nj] = j
            pred_sum += Nj
        
    return y.astype(int), y_bar.astype(int)

class ClsfStatAnalyzer:
    """
    Objects that extract classifier stats from json files
    and performs analyses
    """
    
    def __init__(self,filename):
        self.filename = filename
        
        
    def extract_test_stats(self,stat,class_sets,channels):
        """
        Extract and store the training and test accuracies
        for each set of hyperparameters

        Parameters
        ----------
        stat - str of metric to extract. One of 'accuracy'
        
        class_sets - set of class groups of interest
        channels - set of channels of interest

        Returns
        -------
        numpy array of requested results

        """
        
        # read the file contents
        with open(self.filename) as fp:
            file_content = json.load(fp)
            
        # save only the metric_data portion and the data set
        dataset = file_content['dataset']
        participant = file_content['number']
        file_content = file_content['metric_data']
        
        
        # save the hyperparameter sets and classifier results
        stats = []
        info = []
        hyp_sets = []
        for s in file_content:
            # for each set, extract the useful information
            hyp_set = s['hyp_set']
            
            set_channels = set(hyp_set.pop('channels'))
            target_channels = dataset_channel_map(dataset,channels)
            if len(set_channels ^ target_channels) != 0:
                # channels do not match target, ignore hyp set
                continue
            
            set_classes = tuple(hyp_set.pop('classes'))
            if set_classes not in class_sets:
                # classes not within class sets of interest, continue
                continue
            
            # save the info
            set_info = {
                'participant' : participant,
                'dataset'     : dataset,
                'channels'    : channels,
                'classes'     : set_classes}
            
            test_cf = np.asarray(s['Train-Test']['Test'])
            yte,yte_bar = reverse_confusion_mat(test_cf)
            test_res = skmetrics.classification_report(yte,
                                                       yte_bar,
                                                       output_dict=True)

            stats.append(test_res['accuracy'])
            info.append(set_info)
            hyp_sets.append(hyp_set)
        
        stats = np.asarray(stats)
        return stats,hyp_sets,info
    
    def extract_test_data(self,target_channel_set):
        # read the file contents
        with open(self.filename) as fp:
            file_content = json.load(fp)
            
        # save only the metric_data portion and the data set
        dataset = file_content['dataset']
        participant = file_content['number']
        file_content = file_content['metric_data']
        
        ret_data = {}
        
        for s in file_content:
            # for each set, extract the useful information
            hyp_set = s['hyp_set']
            
            # extract channels
            set_channels = dataset_channel_num2str(dataset,hyp_set['channels'])
            if 'C6' in set_channels:
                channel_set = 1
            else:
                channel_set = 0
                
            if channel_set != target_channel_set:
                continue
            
            # extract class set
            set_classes = hyp_set['classes']
            num_classes = len(set_classes)
            set_classes = "-".join([str(c) for c in set_classes])
                
            set_key = (set_classes,num_classes,channel_set)
            set_data = []
            
            # get stats
            train_cf = np.asarray(s['Train-Test']['Train-confusion-mat'])
            ytr,ytr_bar = reverse_confusion_mat(train_cf)
            train_res = skmetrics.classification_report(ytr,
                                                        ytr_bar,
                                                        output_dict=True)
            test_cf = np.asarray(s['Train-Test']['Test-confusion-mat'])
            yte,yte_bar = reverse_confusion_mat(test_cf)
            test_res = skmetrics.classification_report(yte,
                                                       yte_bar,
                                                       output_dict=True)
            
            set_stats = ['NA'] * 6
            
            # overall accuracy
            set_stats[0] = train_res['accuracy']
            set_stats[1] = test_res['accuracy']
            set_stats[2] = s['Train-Test']['Test-cross-entropy']
            
            # class specific metrics
            metrics = ('precision','recall','f1-score')
            # for i_c in range(num_classes):
            #     for i_m in range(len(metrics)):
            #         i = 2 + i_c * len(metrics) + i_m
            #         set_stats[i] = test_res[str(i_c)][metrics[i_m]]
            
            # weighted average metrics
            for i_m in range(len(metrics)):
                i = 3 + i_m
                set_stats[i] = test_res['weighted avg'][metrics[i_m]]
            
            
            set_data.extend(set_stats)
            ret_data[set_key] = tuple(set_data)
        
        return ret_data

    def extract_id(self):
        # read the file contents
        with open(self.filename) as fp:
            file_content = json.load(fp)
        print(self.filename)
        return file_content['number'], file_content['dataset']
        