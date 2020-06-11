# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 11:16:57 2020

@author: Nick
"""


import numpy as np
from scipy.io import loadmat

class ParticipantDataExtractor:
    """
    A BCI study participant data extractor
    """
    
    def __init__(self,trialdatafile,covdatafile,classes,channels,fbands,set_sz):
        """
        Create a BCI participant data extractor object using data from a mat file
        The mat file contains trial data of different BCI control tasks
        """
        
        self.trialdatafile = trialdatafile
        self.covdatafile = covdatafile
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
        
        Xev = None
        Xtr = None
        Xte = None
        yev = None
        ytr = None
        yte = None
        
        for c in self.classes:
            # extract the trials for this class
            class_trials = data['class{}_trials'.format(c)]
            
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

            if Xev is None:
                Xev = class_trials[:eval_set_sz,:,:,:]
                yev = c*np.ones((eval_set_sz,))
            else:
                Xev = np.concatenate((Xev,class_trials[:eval_set_sz,:,:,:]),
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
        
        Xev = None
        Xtr = None
        Xte = None
        yev = None
        ytr = None
        yte = None
        
        for c in self.classes:
            # extract the trials for this class
            class_trials = data['class{}_trials'.format(c)]
            
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

            if Xev is None:
                Xev = class_trials[:eval_set_sz,:,:,:]
                yev = c*np.ones((eval_set_sz,))
            else:
                Xev = np.concatenate((Xev,class_trials[:eval_set_sz,:,:,:]),
                                     axis=0)
                yev = np.concatenate((yev,c*np.ones((eval_set_sz,))),
                                     axis=0)
            
        return Xev,Xtr,Xte,yev,ytr,yte
    
    