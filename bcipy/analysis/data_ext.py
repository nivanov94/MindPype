# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 11:16:57 2020

@author: Nick
"""


import numpy as np
from scipy.io import loadmat
import json
import csv
from os import path

class ParticipantDataExtractor:
    """
    A BCI study participant data extractor
    """
    
    def __init__(self,
                 trialdatafile,
                 covdatafile,
                 artifactfile,
                 artifact_id,
                 classes,
                 channels,
                 fbands,
                 set_sz):
        """
        Create a BCI participant data extractor object using data from a mat file
        The mat file contains trial data of different BCI control tasks
        """
        
        self.trialdatafile = trialdatafile
        self.covdatafile = covdatafile
        self.artifactfile = artifactfile
        self.artifact_id = artifact_id
        self.channels = channels
        self.fbands = fbands
        self.eval_set_sz = set_sz[0]
        self.train_set_sz = set_sz[1]
        self.test_set_sz = set_sz[2]
        self.classes = classes
        
    
    def extract_trial_data(self):
        """
        Extract the data from the trial data file

        Returns
        -------
        Eval, Train, and Test sets

        """
        data = loadmat(self.trialdatafile)
        
        if self.artifactfile is not None:
            # load artifact data
            with open(self.artifactfile,"r") as afile:
                artifact_data = json.load(afile)
                participant_artifacts = artifact_data[self.artifact_id]
        
        
        Xev = None
        Xtr = None
        Xte = None
        yev = None
        ytr = None
        yte = None
        
        for c in self.classes:
            
            # extract the trials for this class
            class_trials = data['class{}_trials'.format(c)]
            
            if self.artifactfile is not None:
                # remove artifact trials
                artifact_indices = participant_artifacts['c{}_rejects'.format(c)]
                mask = np.ones((class_trials.shape[0],),dtype=int)
                for artf_index in artifact_indices:
                    mask[artf_index] = 0
            
                class_trials = class_trials[mask==1,:,:,:]
            
            Nt, _, Ns, _  = class_trials.shape
            
            # extract wanted channels, fbands
            ixgrid = np.ix_([_ for _ in range(Nt)],self.fbands,[_ for _ in range(Ns)],self.channels)
            class_trials = class_trials[ixgrid]
            
            
            if Xte is None:
                Xte = class_trials[-self.test_set_sz:,:,:,:]
                yte = c*np.ones((self.test_set_sz,))
            else:
                Xte = np.concatenate((Xte,class_trials[-self.test_set_sz:,:,:,:]),
                                     axis=0)
                yte = np.concatenate((yte,c*np.ones(self.test_set_sz,)),
                                     axis=0)
            
            train_start_index = Nt - self.test_set_sz - self.train_set_sz
            if Xtr is None:
                Xtr = class_trials[train_start_index:(Nt-self.test_set_sz),:,:,:]
                ytr = c*np.ones((self.train_set_sz,))
            else:
                Xtr = np.concatenate((Xtr,class_trials[train_start_index:(Nt-self.test_set_sz),:,:,:]),
                                     axis=0)
                ytr = np.concatenate((yte,c*np.ones(self.train_set_sz,)),
                                     axis=0)
                
            if self.eval_set_sz == None:
                eval_set_sz = Nt - self.test_set_sz - self.train_set_sz
            
            else:
                eval_set_sz = self.eval_set_sz

            eval_start_index = train_start_index - eval_set_sz
            if eval_start_index < 0:
                eval_start_index = 0
                eval_set_sz = Nt - self.test_set_sz - self.train_set_sz
            
            if Xev is None:
                Xev = class_trials[eval_start_index:train_start_index,:,:,:]
                yev = c*np.ones((eval_set_sz,))
            else:
                Xev = np.concatenate((Xev,class_trials[eval_start_index:train_start_index,:,:,:]),
                                     axis=0)
                yev = np.concatenate((yev,c*np.ones((eval_set_sz,))),
                                     axis=0)
            
        return Xev,Xtr,Xte,yev,ytr,yte
        
    
    def extract_cov_data(self):
        """
        Extract the data from the trial data file

        Returns
        -------
        Eval, Train, and Test sets

        """
        data = loadmat(self.covdatafile)
        
        if self.artifactfile is not None:
            # load artifact data
            with open(self.artifactfile,"r") as afile:
                artifact_data = json.load(afile)
                participant_artifacts = artifact_data[self.artifact_id]
        
        Xev = None
        Xtr = None
        Xte = None
        yev = None
        ytr = None
        yte = None
        
        for c in self.classes:
            # extract the trials for this class
            class_trials = data['class{}_trials'.format(c)]
            
            if self.artifactfile is not None:
                # remove artifact trials
                artifact_indices = participant_artifacts['c{}_rejects'.format(c)]
                mask = np.ones((class_trials.shape[0],),dtype=int)
                for artf_index in artifact_indices:
                    mask[artf_index] = 0
            
                class_trials = class_trials[mask==1,:,:,:]
            
            Nt = class_trials.shape[0]
            
            # extract wanted channels, fbands
            ixgrid = np.ix_([_ for _ in range(Nt)],self.fbands,self.channels,self.channels)
            class_trials = class_trials[ixgrid]
            
            
            if Xte is None:
                Xte = class_trials[-self.test_set_sz:,:,:,:]
                yte = c*np.ones((self.test_set_sz,))
            else:
                Xte = np.concatenate((Xte,class_trials[-self.test_set_sz:,:,:,:]),
                                     axis=0)
                yte = np.concatenate((yte,c*np.ones(self.test_set_sz,)),
                                     axis=0)
            
            train_start_index = Nt - self.test_set_sz - self.train_set_sz
            if Xtr is None:
                Xtr = class_trials[train_start_index:(Nt-self.test_set_sz),:,:,:]
                ytr = c*np.ones((self.train_set_sz,))
            else:
                Xtr = np.concatenate((Xtr,class_trials[train_start_index:(Nt-self.test_set_sz),:,:,:]),
                                     axis=0)
                ytr = np.concatenate((ytr,c*np.ones(self.train_set_sz,)),
                                     axis=0)
                
            if self.eval_set_sz == None:
                eval_set_sz = Nt - self.test_set_sz - self.train_set_sz
            
            else:
                eval_set_sz = self.eval_set_sz


            eval_start_index = train_start_index - eval_set_sz
            if eval_start_index < 0:
                eval_start_index = 0
                eval_set_sz = Nt - self.test_set_sz - self.train_set_sz
                
            if Xev is None:
                Xev = class_trials[eval_start_index:train_start_index,:,:,:]
                yev = c*np.ones((eval_set_sz,))
            else:
                Xev = np.concatenate((Xev,class_trials[eval_start_index:train_start_index,:,:,:]),
                                     axis=0)
                yev = np.concatenate((yev,c*np.ones((eval_set_sz,))),
                                     axis=0)
            
        return Xev,Xtr,Xte,yev,ytr,yte
    

class ParticipantMetricExtractor:
    
    def __init__(self,
                 datafile,
                 classes,
                 channels,
                 fband,
                 interspread_ref,
                 win_sz,
                 win_type,
                 step_sz):
        
        self.datafile = datafile
        self.classes = classes
        self.channels = channels
        self.fband = fband
        self.interspread_ref = interspread_ref
        self.win_sz = win_sz
        self.win_type = win_type
        self.step_sz = step_sz
    
    def generate_csv(self,outfile):
        
        # open the datafile
        with open(self.datafile,'r') as fp:
            file_data = json.load(fp)
            
        # find the data that matches the hyperparameters
        metric_data = file_data['metric_data']
        
        # etract hyp sets meeting channel reqs
        valid_indices = [i for i in range(len(metric_data))
                             if metric_data[i]['hyp_set']['channels'] == self.channels]
        
        # filter for freq band
        valid_indices = [i for i in valid_indices 
                             if metric_data[i]['hyp_set']['freq_bands'][0] == self.fband]
        
        valid_indices = [i for i in valid_indices
                             if metric_data[i]['hyp_set']['classes'] == self.classes]
        
        valid_indices = [i for i in valid_indices
                             if metric_data[i]['hyp_set']['win_sz'] == self.win_sz]
        
        valid_indices = [i for i in valid_indices
                             if metric_data[i]['hyp_set']['step_sz'] == self.step_sz]
        
        valid_indices = [i for i in valid_indices
                             if metric_data[i]['hyp_set']['win_type'] == self.win_type]
        
        valid_indices = [i for i in valid_indices
                             if metric_data[i]['hyp_set']['inter_spread_ref'] == self.interspread_ref]
        
        if len(valid_indices) != 1:
            raise Exception("Valid indices not 1")
            
        target_data = metric_data[valid_indices[0]]
        
        if len(self.win_type) == 2:
            win_type_str = self.win_type[0] + str(self.win_type[1])
        else:
            win_type_str = self.win_type[0]
        
        output_file = outfile.format(
            file_data['participant'],self.interspread_ref,
            self.win_sz,self.step_sz,win_type_str)
        
        exists = path.isfile(output_file)
        with open(output_file, "a",newline='') as ofp:
            csvfile = csv.writer(ofp)
            
            Nc = len(self.classes)
            header = ('block-index','class-set','delta distinct','end distinct',
                      'delta interspread1','delta_interspread2', 'delta interspread3',
                      'end interspread1', 'end interspread2','end interspread3',
                      'delta consist1', 'delta consist2', 'delta consist3',
                      'end consist1','end consist2','end consist3')
            if Nc == 2:
                if self.interspread_ref == 'common':
                    Nid = 2
                else:
                    Nid = 1
            else:
                Nid = 3
            
            if not exists:
                csvfile.writerow(header)
            
            delta_distinct = 0
            delta_interspread = ['NA'] * 3
            delta_consist = ['NA'] * 3
            end_distinct = 0
            end_interspread = ['NA'] * 3
            end_consist = ['NA'] * 3
            for i in range(len(target_data['Eval-Distinct'])):
                if i == 0:
                    delta_distinct = 0
                    delta_interspread = [0] * Nid
                    delta_consist = [0] * Nc
                else:
                    delta_distinct = target_data['Eval-Distinct'][i][0] - end_distinct
                    for j in range(Nid):
                        delta_interspread[j] = target_data['Eval-InterSpread'][i][0][j] - end_interspread[j]
                        
                    for j in range(Nc):
                        delta_consist[j] = target_data['Eval-Consist'][i][0][j] - end_consist[j]
                    
                
                end_distinct = target_data['Eval-Distinct'][i][0]
                end_interspread = [target_data['Eval-InterSpread'][i][0][j] for j in range(Nid)]
                end_consist = [target_data['Eval-Consist'][i][0][j] for j in range(Nc)]
                
                row = ([i, ".".join([str(c) for c in self.classes]),delta_distinct,end_distinct]
                       + delta_interspread + end_interspread
                       + delta_consist + end_consist)
                
                if i != 0:
                    csvfile.writerow(row)

        

def _recall(confus_mat, label):
    return confus_mat[label,label] / np.sum(confus_mat[label,:])

def _precision(confus_mat, label):
    if np.sum(confus_mat[:,label]) == 0:
        return 0
    return confus_mat[label,label] / np.sum(confus_mat[:,label])

def _f1(confus_mat, label):
    """ One vs. Rest"""
    recall = _recall(confus_mat,label)
    precis = _precision(confus_mat,label)
    if (precis + recall) == 0:
        return 0
    return 2 * (precis * recall) / (precis + recall)


class ParticipantClassifierExtractor:
    
    def __init__(self,
                 datafile,
                 classes,
                 channels,
                 fband,
                 win_sz,
                 win_type,
                 step_sz):
        
        self.datafile = datafile
        self.classes = classes
        self.channels = channels
        self.fband = fband
        self.win_sz = win_sz
        self.win_type = win_type
        self.step_sz = step_sz
    
    def generate_csv(self,outfile):
        
        # open the datafile
        with open(self.datafile,'r') as fp:
            file_data = json.load(fp)
            
        # find the data that matches the hyperparameters
        metric_data = file_data['metric_data']
        
        # etract hyp sets meeting channel reqs
        valid_indices = [i for i in range(len(metric_data))
                             if metric_data[i]['hyp_set']['channels'] == self.channels]
        
        # filter for freq band
        valid_indices = [i for i in valid_indices 
                             if metric_data[i]['hyp_set']['freq_bands'] == self.fband]
        
        valid_indices = [i for i in valid_indices
                             if metric_data[i]['hyp_set']['classes'] == self.classes]
        
        valid_indices = [i for i in valid_indices
                             if metric_data[i]['hyp_set']['win_sz'] == self.win_sz]
        
        valid_indices = [i for i in valid_indices
                             if metric_data[i]['hyp_set']['step_sz'] == self.step_sz]
        
        valid_indices = [i for i in valid_indices
                             if metric_data[i]['hyp_set']['win_type'] == self.win_type]
        
        
        if len(valid_indices) != 1:
            raise Exception("Valid indices not 1")
            
        target_data = metric_data[valid_indices[0]]
        
        if len(self.win_type) == 2:
            win_type_str = self.win_type[0] + str(self.win_type[1])
        else:
            win_type_str = self.win_type[0]
        
        output_file = outfile.format(
            file_data['number'],
            self.win_sz,self.step_sz,win_type_str)
        
        exists = path.isfile(output_file)
        with open(output_file, "a",newline='') as ofp:
            csvfile = csv.writer(ofp)
            
            Nc = len(self.classes)
            header = ('block-index','class-set','block-acc','session-acc',
                      'block-t1-p1','block-t1-p2','block-t1-p3',
                      'block-t2-p1','block-t2-p2','block-t2-p3',
                      'block-t3-p1','block-t3-p2','block-t3-p3',
                      'session-t1-p1','session-t1-p2','session-t1-p3',
                      'session-t2-p1','session-t2-p2','session-t2-p3',
                      'session-t3-p1','session-t3-p2','session-t3-p3',
                      'block-l1-recall','session-l1-recall','delta-l1-recall',
                      'block-l2-recall','session-l2-recall','delta-l2-recall',
                      'block-l3-recall','session-l3-recall','delta-l3-recall',
                      'block-l1-precision','session-l1-precision','delta-l1-precision',
                      'block-l2-precision','session-l2-precision','delta-l2-precision',
                      'block-l3-precision','session-l3-precision','delta-l3-precision',
                      'block-l1-f1','session-l1-f1','delta-l1-f1',
                      'block-l2-f1','session-l2-f1','delta-l2-f1',
                      'block-l3-f1','session-l3-f1','delta-l3-f1')
            
            if not exists:
                csvfile.writerow(header)
            target_data = target_data['Online']['Test']
            session_conf_mat = np.zeros((Nc*Nc,),dtype=int)
            session_acc = 0
            
            recall = np.zeros((3*Nc,))
            precision = np.zeros((3*Nc,))
            f1 = np.zeros((3*Nc,))
            for i in range(len(target_data)):
                block_conf_mat = np.reshape(target_data[i],-1).astype(int)
                if Nc == 3:
                    block_acc = sum(block_conf_mat[i] for i in (0,4,8)) / np.sum(block_conf_mat)
                else:
                    block_acc = sum(block_conf_mat[i] for i in (0,3)) / np.sum(block_conf_mat)
                
                for c in range(Nc):
                    recall[3*c] = _recall(np.reshape(block_conf_mat,(Nc,Nc)),c)
                    precision[3*c] = _precision(np.reshape(block_conf_mat,(Nc,Nc)),c)
                    f1[3*c] = _f1(np.reshape(block_conf_mat,(Nc,Nc)),c)
                
                row = ([i+1, ".".join([str(c) for c in self.classes]),block_acc,session_acc] +
                        block_conf_mat.tolist() +
                        session_conf_mat.tolist() + 
                        recall.tolist() + precision.tolist() + f1.tolist())
                
                session_conf_mat += block_conf_mat
                if Nc == 3:
                    session_acc = sum(session_conf_mat[i] for i in (0,4,8)) / np.sum(session_conf_mat)
                else:
                    session_acc = sum(session_conf_mat[i] for i in (0,3)) / np.sum(session_conf_mat)
                
                for c in range(Nc):
                    recall[3*c+2] = recall[3*c] >= recall[3*c+1]
                    recall[3*c+1] = _recall(np.reshape(session_conf_mat,(Nc,Nc)),c)
                    precision[3*c+2] = precision[3*c] >= precision[3*c+1]
                    precision[3*c+1] = _precision(np.reshape(session_conf_mat,(Nc,Nc)),c)
                    f1[3*c+2] = f1[3*c] >= f1[3*c+1]
                    f1[3*c+1] = _f1(np.reshape(session_conf_mat,(Nc,Nc)),c)
                    
                csvfile.writerow(row)
            
            