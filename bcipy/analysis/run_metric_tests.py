# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 13:41:41 2020

@author: Nick
"""


from metric_calc import Distinct
from data_ext import ParticipantDataExtractor

import json

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
    

def extract_metric_hyperparams(cfg):
    """
    Create sets of hyperparameters from
    the configuration file
    """
    
    add_hyp_set = lambda hs, names, vals : hs.append(dict(zip(names,vals)))
        
    
    hyp_sets = []
    
    hyperparams = ['channels','freq_bands','classes','train_set_sz',
                   'test_set_sz','eval_set_sz','inter_spread_ref',
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
    
    

def write_metric_data(filename,metric_data):
    """
    Write data to JSON file
    """
    with open(filename,'r') as destf:
        # read existing json file
        data = json.load(destf)

    with open(filename,'w') as destf:
        # add new data
        if 'metric_data' in data:
            data['metric_data'].append(metric_data)
        else:
            data['metric_data'] = [metric_data]
        
        # write to the file
        json.dump(data,destf,indent=2)

def load_config(filename):
    """
    Load configurations for the test from a JSON file
    """
    
    with open(filename,"r") as f:
        cfg = json.load(f)
        
    return cfg

def run_participant(participant,cfg):
    """
    Run all metric and classifier tests
    on participant and save results to 
    JSON file.
    """
    
    ## Metrics
    # extract metric hyperparameter set
    metric_hyperparams = extract_metric_hyperparams(cfg)
    
    outfile = (cfg['output_file'] + "P" + str(participant['number'])
               + '-' + timestr + '.json')
    
    with open(outfile,'w+') as dest_file:
        json.dump({'participant' : participant['number'],
                   'dataset'     : cfg['dataset']},
                  dest_file)
    
    artifact_file = cfg['artifacts_file']
    
    i = 1
    total_sets = len(metric_hyperparams)
    for hyp_set in metric_hyperparams:
        print("\t|\tCalculating metrics with hyperparams:",hyp_set)
        print("\t|\tHyperparameter set {} of {}".format(i,total_sets))
        i += 1
        # extract data
        classes = hyp_set['classes']    # tuple of class labels
        channels = hyp_set['channels']  # tuple of channel numbers
        fbands = hyp_set['freq_bands']  # tuple of frequency band IDs
        tr_set_sz = hyp_set['train_set_sz']
        te_set_sz = hyp_set['test_set_sz']
        ev_set_sz = hyp_set['eval_set_sz']
        
        trial_data_file = participant['trial_data_file']
        cov_data_file = participant['cov_data_file']
        
        PDE = ParticipantDataExtractor(trial_data_file,
                                       cov_data_file,
                                       artifact_file,
                                       str(participant['number']),
                                       classes,
                                       channels,
                                       fbands,
                                       (ev_set_sz,tr_set_sz,te_set_sz))
        
        # get the covariance data matrices
        Xev,Xtr,Xte,yev,ytr,yte = PDE.extract_cov_data()
        
        # calculate metrics
        print("\t|\t|\tCalculating Distinct metrics...")
        metric_components = cfg['metrics']
        ref = hyp_set['inter_spread_ref']
        win_sz = hyp_set['win_sz']
        win_type = hyp_set['win_type'][0]
        step_sz = hyp_set['step_sz']
        if win_type == 'exponential':
            decay = hyp_set['win_type'][1]
        metric = Distinct(metric_components,ref,win_sz,win_type,step_sz,len(classes),decay)
            
        # calcualte evaluation set metrics
        if yev.shape[0] > 0:
            print("\t|\t|\t|\tEval Set...")
            eval_metrics = metric.calculate(Xev,yev)
        
        if ytr.shape[0] > 0:
            print("\t|\t|\t|\tTrain Set...")
            train_metrics = metric.calculate(Xtr,ytr)
        
        if yte.shape[0] > 0:
            print("\t|\t|\t|\tTest Set...")
            test_metrics = metric.calculate(Xte,yte)
        
        # add the newest data
        metric_data = {'hyp_set' : hyp_set,
                       'Eval-Trials' : yev.shape[0],
                       'Train-Trials' : ytr.shape[0],
                       'Test-Trials' : yte.shape[0]}
        
        if 'Distinct' in metric_components:
            if yev.shape[0] > 0:
                metric_data['Eval-Distinct'] = eval_metrics['Distinct'].tolist()
            
            if ytr.shape[0] > 0:
                metric_data['Train-Distinct'] = train_metrics['Distinct'].tolist()
            
            if yte.shape[0] > 0:
                metric_data['Test-Distinct'] = test_metrics['Distinct'].tolist()
        
        if 'InterSpread' in metric_components:
            if yev.shape[0] > 0:
                metric_data['Eval-InterSpread'] = eval_metrics['InterSpread'].tolist()
            
            if ytr.shape[0] > 0:
                metric_data['Train-InterSpread'] = train_metrics['InterSpread'].tolist()
            
            if yte.shape[0] > 0:
                metric_data['Test-InterSpread'] = test_metrics['InterSpread'].tolist()
        
        if 'Consist' in metric_components:
            if yev.shape[0] > 0:
                metric_data['Eval-Consist'] = eval_metrics['Consist'].tolist()
            
            if ytr.shape[0] > 0:
                metric_data['Train-Consist'] = train_metrics['Consist'].tolist()
            
            if yte.shape[0] > 0:
                metric_data['Test-Consist'] = test_metrics['Consist'].tolist()
        
        write_metric_data(outfile,metric_data)


def metric_analysis(config_file):
    cfg = load_config(config_file)
    
    # extract participant data
    participants = cfg['participants']
    trial_data_template =  cfg['trial_data_dir'] + cfg['trial_data_files']
    cov_data_template =  cfg['cov_data_dir'] + cfg['cov_data_files']
    output_file_template = cfg['output_file']
    
    for participant in participants:
        partic_data = {"trial_data_file" : trial_data_template.format(participant),
                       "cov_data_file"   : cov_data_template.format(participant),
                       "number"          : participant}
        
        print("Performing analysis for participant {}...".format(participant))
        run_participant(partic_data,cfg)


if __name__ == "__main__":
    config_file = "D:\BCI\BCI_Capture\data\MI_datasets\high_gamma_metric_testing_cfg.json"
    
    metric_analysis(config_file)