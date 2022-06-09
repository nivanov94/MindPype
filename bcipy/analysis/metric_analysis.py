# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:58:28 2020

@author: Nick
"""

import numpy as np
import json

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


def _ext_metric_summary(metrics,window_desc,collapse=False):
    metrics = np.squeeze(np.asarray(metrics),axis=1)
    if collapse:
        metrics = np.sum(metrics,axis=1)
    
    if len(metrics.shape) == 1:
        if window_desc == 'first':
            return metrics[0]
        elif window_desc == 'last':
            return metrics[-1]
        elif window_desc == 'mean':
            return np.mean(metrics)
        elif window_desc == 'best':
            return np.max(metrics)
        else:
            return np.min(metrics)
    else:
        if window_desc == 'first':
            return metrics[0,:]
        elif window_desc == 'last':
            return metrics[-1,:]
        elif window_desc == 'best':
            metric_avgs = np.mean(metrics,axis=1)
            highest_avg = np.argmax(metric_avgs)
            return metrics[highest_avg,:]
        else:
            metric_avgs = np.mean(metrics,axis=1)
            lowest_avg = np.argmin(metric_avgs)
            return metrics[lowest_avg,:]

class MetricAnalyzer:
    
    def __init__(self,filename):
        self.filename = filename
        
    def extract_eval_metric_stats(self,metric_name,window_desc,class_sets,channels):
        """
        Extract and store the training and test accuracies
        for each set of hyperparameters

        Parameters
        ----------
        metric_name - str of metric to extract. One of 'distinct',
                 'interspread', 'consistency-sum', 'consistency-ind'
        
        window_desc - str of window of which to extract metric. One of
                      'first', 'last', 'best', 'worst'
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
        participant = file_content['participant']
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
            
            
            # extract the stat
            if metric_name == 'distinct':
                key = 'Eval-Distinct'
            elif metric_name == 'interspread':
                key = 'Eval-InterSpread'
            else:
                key = 'Eval-Consist'
                
            metrics = s[key]
            value = _ext_metric_summary(metrics,
                                        window_desc,
                                        metric_name == 'consistency-sum')
            
            if np.sum(value) == 0:
                # invalid
                continue 
            
            stats.append(value)
            info.append(set_info)
            hyp_sets.append(hyp_set)
        
        stats = np.asarray(stats)
        
        return stats, hyp_sets, info
    
    def extract_data(self,freq,interspread_ref,eval_set_sz,
                     win_sz,win_type,step_sz):
        
        # read the file contents
        with open(self.filename) as fp:
            file_content = json.load(fp)
            
        # save only the metric_data portion and the data set
        dataset = file_content['dataset']
        participant = file_content['participant']
        file_content = file_content['metric_data']
        
        ret_data = {}
        
        for s in file_content:
            # for each set, extract the useful information
            hyp_set = s['hyp_set']
            
            target_set = ((hyp_set['freq_bands'][0] == freq) and
                (hyp_set['inter_spread_ref'] == interspread_ref) and
                (hyp_set['win_sz'] == win_sz) and
                (hyp_set['win_type'] == win_type) and 
                (hyp_set['step_sz'] == step_sz) and
                (hyp_set['eval_set_sz'] == eval_set_sz))
            
            if target_set:
                
                # extract channels
                set_channels = dataset_channel_num2str(dataset,hyp_set['channels'])
                if 'C6' in set_channels:
                    channel_set = 1
                else:
                    channel_set = 0
                
                # extract class set
                set_classes = hyp_set['classes']
                num_classes = len(set_classes)
                set_classes = "-".join([str(c) for c in set_classes])
                
                set_key = (set_classes,num_classes,channel_set)
                set_data = [s['Eval-Trials']]
                
                metric_keys = ['Eval-Distinct',
                               'Eval-InterSpread',
                               'Eval-Consist']
                
                metric_windows = ['best','worst','first','last','mean']
                
                for metric_key in metric_keys:
                    metrics = s[metric_key]
                    
                    for win in metric_windows:
                        metric_data = _ext_metric_summary(metrics,
                                                          win,
                                                          metric_key != 'Eval-Distinct')
                        set_data.append(metric_data)
                
                # iso_consist_metrics = ['NA'] * 3 * len(metric_windows)
                # for i_w in range(len(metric_windows)):
                #     win = metric_windows[i_w]
                #     metrics = s['Eval-Consist']
                #     consist_data = _ext_metric_summary(metrics,
                #                                        win)
                    
                #     for i_d in range(consist_data.shape[0]):
                #         iso_consist_metrics[i_d*len(metric_windows) + i_w] = consist_data[i_d]
                    
                # set_data.extend(iso_consist_metrics)
                ret_data[set_key] = tuple(set_data)
                
        return ret_data
                
                
    def extract_id(self):
        # read the file contents
        with open(self.filename) as fp:
            file_content = json.load(fp)
        
        return file_content['participant'], file_content['dataset']