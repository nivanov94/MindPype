# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:48:26 2020

@author: Nick
"""

from correlation_analysis import MetricClsfCorrelAnalyzer

import json
import glob

import time
timestr = time.strftime("%Y%m%d-%H%M")
    

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
    

def extract_analysis_hyperparams(cfg):
    """
    Create sets of hyperparameters from
    the configuration file
    """
    
    add_hyp_set = lambda hs, names, vals : hs.append(dict(zip(names,vals)))
        
    
    hyp_sets = []
    
    hyperparams = ['metric_fbands','metric_references','eval_set_sz',
                   'win_sz','win_type','step_sz','channel_set']
    
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



def load_config(filename):
    """
    Load configurations for the test from a JSON file
    """
    
    with open(filename,"r") as f:
        cfg = json.load(f)
        
    return cfg

def run_analysis(cfg):
    """
    Run all metric and classifier tests
    on participant and save results to 
    JSON file.
    """
    
    analysis_hyperparams = extract_analysis_hyperparams(cfg)
    
    #outfile = (cfg['output_file'] + '-' + timestr + '.json')
    
    # get all the metric and clsf files
    metric_files = []
    for d in cfg['metric_files']:
        tmplate, ts = d
        metric_files.extend(glob.glob(tmplate.format(ts)))
    
    clsf_files = []
    for d in cfg['clsf_files']:
        tmplate, ts = d
        clsf_files.extend(glob.glob(tmplate.format(ts)))
    
    # create analyzer object
    Nf = len(clsf_files)
    file_pairs = [(metric_files[i],clsf_files[i]) for i in range(Nf)]
    correl_analyzer = MetricClsfCorrelAnalyzer(file_pairs)
    
    
    #with open(outfile,'w+') as dest_file:
    #    json.dump({'Analysis' : 'Correl'},dest_file)
    
    i = 1
    total_sets = len(analysis_hyperparams)
    for hyp_set in analysis_hyperparams:
        print("\t|\tCalculating correlation with hyperparams:",hyp_set)
        print("\t|\tHyperparameter set {} of {}".format(i,total_sets))
        i += 1

        # set hyperparameters
        metric_fband = hyp_set['metric_fbands']
        metric_references = hyp_set['metric_references']
        win_sz = hyp_set['win_sz']
        step_sz = hyp_set['step_sz']
        win_type = hyp_set['win_type']
        channel_set = hyp_set['channel_set']
        eval_set_sz = hyp_set['eval_set_sz']
        
        csv_name = (cfg['output_file'] +
                    "evalsz-{}-band-{}-ref-{}-wintype-{}-winsz-{}-stepsz-{}".format(eval_set_sz,
                                                                                    metric_fband,
                                                                                    metric_references,
                                                                                    win_type,
                                                                                    win_sz,
                                                                                    step_sz) +
                    "-" + timestr + ".csv")
        
        correl_analyzer.generate_csv(csv_name,
                                     metric_fband,
                                     metric_references,
                                     eval_set_sz,
                                     win_sz,
                                     win_type,
                                     step_sz,
                                     channel_set)
        
        # # run analysis method
        # R, L,pairs = correl_analyzer.metric_acc_correl(metric_name,
        #                                                win_type,
        #                                                clsf_stat,
        #                                                class_sets,
        #                                                channels)
        
        
        # # write analysis results
        # correl_data = {'hyp_set' : hyp_set,
        #                'R'       : R.tolist(),
        #                'L'       : L.tolist(),
        #                'pairs'   : pairs}
        
        # write_correlation_data(outfile,correl_data)


def metric_analysis(config_file):
    cfg = load_config(config_file)
        
    run_analysis(cfg)


if __name__ == "__main__":
    config_file = "D:\BCI\BCI_Capture\data\MI_datasets\metric_accuracy_correl_cfg.json"
    
    metric_analysis(config_file)