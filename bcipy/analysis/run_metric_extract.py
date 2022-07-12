# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:20:59 2020

@author: Nick
"""


from data_ext import ParticipantMetricExtractor

import json

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
    
    hyperparams = ['channels','freq_bands','classes','inter_spread_ref',
                   'win_sz','win_type','step_sz']
    
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

def run_participant(participant,cfg):
    """
    Run all metric extractions
    on participant and save results to 
    CSV file.
    """
    
    ## Metrics
    # extract metric hyperparameter set
    metric_hyperparams = extract_metric_hyperparams(cfg)
    
    outfile = (cfg['output_file'] + "{}-ref{}-ws{}-ss{}-wt{}"
               + '-' + timestr + '.csv')
    
    i = 1
    total_sets = len(metric_hyperparams)
    for hyp_set in metric_hyperparams:
        print("\t|\Extracting metrics with hyperparams:",hyp_set)
        print("\t|\tHyperparameter set {} of {}".format(i,total_sets))
        i += 1
        # extract data
        classes = hyp_set['classes']    # tuple of class labels
        channels = hyp_set['channels']  # tuple of channel numbers
        fbands = hyp_set['freq_bands']  # tuple of frequency band IDs
        ref = hyp_set['inter_spread_ref']
        win_sz = hyp_set['win_sz']
        win_type = hyp_set['win_type']
        step_sz = hyp_set['step_sz']
        
        participant_metric_file = cfg['metric_file_tmp'].format(participant)
        
        PME = ParticipantMetricExtractor(participant_metric_file,
                                         list(classes),
                                         list(channels),
                                         fbands[0],
                                         ref,
                                         win_sz,
                                         win_type,
                                         step_sz)
        
        
        
        PME.generate_csv(outfile)
        
def metric_extraction(config_file):
    cfg = load_config(config_file)
    
    # extract participant data
    participants = cfg['participants']
    
    
    for participant in participants:
        
        print("Performing analysis for participant {}...".format(participant))
        run_participant(participant,cfg)


if __name__ == "__main__":
    config_file = "D:\BCI\BCI_Capture\data\MI_datasets\kaya_metric_extracting_cfg.json"
    
    metric_extraction(config_file)