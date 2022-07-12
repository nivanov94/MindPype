# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 20:12:41 2020

@author: Nick
"""


import json
import csv
from glob import glob
import numpy as np

import time
timestr = time.strftime("%Y%m%d-%H%M")


def load_config(filename):
    """
    Load configurations for the test from a JSON file
    """
    
    with open(filename,"r") as f:
        cfg = json.load(f)
        
    return cfg

def sample_hyperparams(available_set,sample_set,used_set):
    """
    Update the three sets for hyperparameter combinations
    """
    i = 0
    done = False
    
    while not done:
        
        # move sample from index i to used_set
        used_set[i].append(sample_set[i])
        
        # check if sample can be replaced from available set
        if len(available_set[i]) > 0:
            # replace from the available set, done
            sample_set[i] = available_set[i].pop()
            done = True
        else:
            # move all items from the used set back into the available set
            available_set[i] = used_set[i]
            used_set[i] = []
            
            # move one item from the available set into the sample set
            sample_set[i] = available_set[i].pop()
            
            # increment i to move to the next hyperparam
            i += 1
    

def extract_metric_hyperparams(cfg):
    """
    Create sets of hyperparameters from
    the configuration file
    """
    
    add_hyp_set = lambda hs, names, vals : hs.append(dict(zip(names,vals)))
        
    
    hyp_sets = []
    
    hyperparams = ['num_classes','ref',
                   'win_sz','metric_win_type','clsf_win_type','step_sz']
    
    available_set = [list(cfg[h]) for h in hyperparams]
    used_set = [[] for _ in hyperparams]
    
    # initialize first sample set
    sample_set = [h_set.pop() for h_set in available_set]
    
    # save sample set
    add_hyp_set(hyp_sets,hyperparams,sample_set)
    
    available_params = sum([len(h_set) for h_set in available_set])

    while available_params > 0:
        
        # update available and used sets
        sample_hyperparams(available_set,sample_set,used_set)
        
        # save the current sample set
        add_hyp_set(hyp_sets,hyperparams,sample_set)
        
        # calculate remaining available parameters
        available_params = sum([len(h_set) for h_set in available_set])
    
    return hyp_sets


def entropy(s1,s2):
    tot = s1 + s2
    p1 = s1 / tot
    p2 = s2 / tot
    return -p1 * np.log2(p1) - p2 * np.log2(p2)

def mutual_information(Sxy):
    Pxy = Sxy / np.sum(Sxy)
    Px = np.sum(Sxy,axis=1) / np.sum(Sxy)
    Py = np.sum(Sxy,axis=0) / np.sum(Sxy)
    
    MI = 0
    for i_x in range(2):
        for i_y in range(2):
          MI += Pxy[i_x,i_y] * np.log2(Pxy[i_x,i_y] / (Px[i_x] * Py[i_y]))
    
    return MI
    

def run_mutual_info_analysis(config_file):
    """
    Run all metric extractions
    on participant and save results to 
    CSV file.
    """
    cfg = load_config(config_file)
    ## Metrics
    # extract metric hyperparameter set
    metric_hyperparams = extract_metric_hyperparams(cfg)
    
    outfile = cfg['output_file'].format(timestr)
    
    results = []
    
    i = 1
    total_sets = len(metric_hyperparams)
    for hyp_set in metric_hyperparams:
        print("\t|\Extracting metrics with hyperparams:",hyp_set)
        print("\t|\tHyperparameter set {} of {}".format(i,total_sets))
        i += 1
        
        # get all the files for analysis
        clsf_files = []
        for clsf_file_template in cfg['clsf_file_template']:
            dset_clsf_files = glob(clsf_file_template.format(
                hyp_set['win_sz'],
                hyp_set['step_sz'],
                hyp_set['clsf_win_type']))
            
            clsf_files.extend(dset_clsf_files)
            
        metric_files = []
        for metric_file_template in cfg['metric_file_template']:
            dset_metric_files = glob(metric_file_template.format(
                hyp_set['ref'],
                hyp_set['win_sz'],
                hyp_set['step_sz'],
                hyp_set['metric_win_type']))
            
            metric_files.extend(dset_metric_files)
            
        consist_cm = np.zeros((2,2))
        intermean_cm = np.zeros((2,2))
        # open each pair of files and extract relevant metrics
        for (m_file,c_file) in zip(metric_files,clsf_files):
            mfp = open(m_file,'r')
            cfp = open(c_file,'r')
            
            mfp_csv = csv.reader(mfp)
            cfp_csv = csv.reader(cfp)
            
            header = True
            for (m_line,c_line) in zip(mfp_csv,cfp_csv):
                if header:
                    d_pw_cols = [m_line.index(d) for d in ('delta interspread1',
                                                          'delta_interspread2',
                                                          'delta interspread3')]
                    d_con_cols = [m_line.index(d) for d in ('delta consist1',
                                                           'delta consist2',
                                                           'delta consist3')]
                    
                    d_recall_cols = [c_line.index(d) for d in ('delta-l1-recall',
                                                              'delta-l2-recall',
                                                              'delta-l3-recall')]
                    
                    d_f1_cols = [c_line.index(d) for d in ('delta-l1-f1',
                                                           'delta-l2-f1',
                                                           'delta-l3-f1')]
                    
                    block_f1_cols = [c_line.index(d) for d in ('block-l1-f1',
                                                           'block-l2-f1',
                                                           'block-l3-f1')]
                    
                    sess_f1_cols = [c_line.index(d) for d in ('session-l1-f1',
                                                           'session-l2-f1',
                                                           'session-l3-f1')]
                    
                    header = False
                else:
                    if m_line[0] != c_line[0] or m_line[1] != c_line[1]:
                        raise Exception('file mismatch')
                        
                    if int(m_line[0]) == 1:
                        continue
                    
                    # extract data
                    
                    # ground truth metic changes
                    d_pw_vals = [float(m_line[i]) >= 0 for i in d_pw_cols]
                    d_con_vals = [float(m_line[i]) >= 0 for i in d_con_cols]
                    
                    # classifier derived predictions
                    d_con_preds = [float(c_line[i]) for i in d_recall_cols]
                    
                    # use mean of f1 scores for pw prediction
                    pairs = ((0,1),(0,2),(1,2))
                    d_pw_preds = []
                    for pair in pairs:
                        sum_block_f1 = (float(c_line[block_f1_cols[pair[0]]]) + 
                                        float(c_line[block_f1_cols[pair[1]]]))
                        sum_sess_f1 = (float(c_line[sess_f1_cols[pair[0]]]) + 
                                       float(c_line[sess_f1_cols[pair[1]]]))
                        d_pw_preds.append(sum_block_f1 >= sum_sess_f1)
                    
                    d_pw_vals = np.asarray(d_pw_vals).astype(int)
                    d_con_vals = np.asarray(d_con_vals).astype(int)
                    d_pw_preds = np.asarray(d_pw_preds).astype(int)
                    d_con_preds = np.asarray(d_con_preds).astype(int)
                    
                    for i in range(3):
                        consist_cm[d_con_vals[i],d_con_preds] += 1
                        intermean_cm[d_pw_vals[i],d_pw_preds[i]] += 1
                    
            mfp.close()
            cfp.close()
        
        # calculate MI
        
        # start with entropy of each var
        delta_pw_entropy = entropy(np.sum(intermean_cm[0,:]),np.sum(intermean_cm[1,:]))
        delta_con_entropy = entropy(np.sum(consist_cm[0,:]),np.sum(consist_cm[1,:]))
        
        delta_f1_entropy = entropy(np.sum(intermean_cm[:,0]),np.sum(intermean_cm[:,1]))
        delta_recall_entropy = entropy(np.sum(consist_cm[:,0]),np.sum(consist_cm[:,1]))
        
        # calculate the mutual information
        pw_mi = mutual_information(intermean_cm)
        con_mi = mutual_information(consist_cm)
        
        hyp_set_results = {'hyp_set' : hyp_set,
                           'delta_pw_entropy' : delta_pw_entropy,
                           'delta_con_entropy' : delta_con_entropy,
                           'delta_f1_entropy' : delta_f1_entropy,
                           'delta_recall_entropy' : delta_recall_entropy,
                           'pw_mi' : pw_mi,
                           'con_mi' : con_mi,
                           'consist_cm' : consist_cm.tolist(),
                           'pw_cm' : intermean_cm.tolist()}
        results.append(hyp_set_results)
    
    with open(outfile,'w') as ofp:
        json.dump(results,ofp,indent=2)
    


if __name__ == "__main__":
    config_file = "D:\BCI\BCI_Capture\data\MI_datasets\mutual_info_analysis_cfg.json"
    run_mutual_info_analysis(config_file)