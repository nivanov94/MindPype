# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 21:03:01 2020

@author: Nick
"""


"""
Metric eval calculations
"""

from pyriemann.utils.mean import mean_riemann as rmean
from pyriemann.utils.distance import distance_riemann as rdist
import numpy as np
from itertools import combinations as iter_combs

from scipy.linalg import fractional_matrix_power

def _inter_spread(means,reference):
    if reference == 'common':
        # find the mean of the means
        common_mean = rmean(means)
                        
        # sum the distance from each mean to the common mean
        inter_spread = [rdist(m,common_mean) for m in means]
                        
    elif reference == 'pairwise':
        # sum the distance between each pair of means
        inter_spread = []
        for i_m1,i_m2 in iter_combs(range(means.shape[0]),2):
            mean1 = means[i_m1,:,:]
            mean2 = means[i_m2,:,:]
            inter_spread.append(rdist(mean1,mean2))
            
    return inter_spread


class Metric:
    
    def __init__(self,win_sz,win_type,step_sz,classes,decay=0.9):
        self.win_sz = win_sz
        self.win_type = win_type
        self.step_sz = step_sz
        self.classes = classes
        self.decay = decay
        
        
    def calc_chunk_mean_and_disp(self,data,win_sz,i_s,i_f):            
        Nc = data[0].shape[-1]
        means = np.zeros((self.classes,Nc,Nc))
        disp = np.zeros((self.classes,))
        if self.win_type == 'sliding':
            # extract data
            start = i_s * self.step_sz
            stop = start + win_sz
            for i_c in range(self.classes):
                step_data = np.squeeze(data[i_c][start:stop,i_f,:,:])
                    
                # calculate the mean
                means[i_c,:,:] = rmean(step_data)
                        
                # calculate the average distance to the mean
                disp[i_c] = sum([rdist(t,means[i_c,:,:]) for t in step_data])
                disp[i_c] /= step_data.shape[0]
                        
        elif self.win_type == 'exponential':
            #extract data
            if i_s == 0:
                for i_c in range(self.classes):
                    step_data = np.squeeze(data[i_c][0:win_sz,i_f,:,:])
                
                    # calculate the mean
                    means[i_c,:,:] = rmean(step_data)
                        
                    # calculate the average distance to the mean
                    disp[i_c] = sum([rdist(t,means[i_c,:,:]) for t in step_data])
                    disp[i_c] /= step_data.shape[0]
                                
                self.prev_means = np.copy(means)
                self.prev_disp = np.copy(disp)
                        
            else:
                start = (i_s - 1) * self.step_sz + win_sz
                stop = start + self.step_sz

                for i_c in range(self.classes):
                    step_data = np.squeeze(data[i_c][start:stop,i_f,:,:])
                    
                    # calculate the mean
                    #covs = np.concatenate((np.expand_dims(self.prev_means[i_c,:,:],axis=0),
                    #                       step_data),
                    #                      axis=0)
                    ## create the weigtht vector
                    #new_sample_weights = tuple([(1-self.decay) / self.step_sz] * self.step_sz)
                    #weights = (self.decay,) + new_sample_weights
                           
                    #means[i_c,:,:] = rmean(covs,sample_weight=weights)
                    
                    # calculate the block mean
                    block_mean = rmean(step_data)
                    
                    # new mean is weighted mean between session and block mean
                    prev_mean_inv_sqrt = fractional_matrix_power(self.prev_means[i_c,:,:],-1/2)
                    prev_mean_sqrt = fractional_matrix_power(self.prev_means[i_c,:,:],1/2)
                    inner = np.matmul(prev_mean_inv_sqrt,np.matmul(block_mean,prev_mean_inv_sqrt))
                    means[i_c,:,:] = np.matmul(prev_mean_sqrt,
                                               np.matmul(fractional_matrix_power(inner,(1-self.decay)),
                                                         prev_mean_sqrt))
                    
                    # calculate the average distance to the mean
                    disp_block = sum([rdist(t,block_mean) for t in step_data])
                    disp_block /= step_data.shape[0]
                    disp[i_c] = self.decay * self.prev_disp[i_c] + (1 - self.decay) * disp_block
                    
                self.prev_means = np.copy(means)
                self.prev_disp = np.copy(disp)
        
        return means, disp

class Distinct(Metric):
    
    def __init__(self,components,reference,win_sz,
                 win_type,step_sz,classes,decay=0.9):
        super().__init__(win_sz,win_type,step_sz,classes,decay)
        self.components = components
        self.reference = reference
        
    def calculate(self,X,y):
        """
        Calculate the metrics according to appributes
        """
        data = [None] * self.classes
        labels = np.unique(y)
        
        if labels.shape[0] != self.classes:
            raise("Too many labels")
        
        for i_l in range(labels.shape[0]):
            label = labels[i_l]
            data[i_l] = X[y==label,:,:,:]
        
        # calculate the maximum number of trials available for each class
        Nt = min([x.shape[0] for x in data])
        
        if self.win_sz == None:
            win_sz = Nt
        else:
            win_sz = self.win_sz

        metrics = {}
        if Nt < win_sz:
            # not enough trials, return zeros
            for comp in self.components:
                if comp == 'Distinct':        
                    metrics['Distinct'] = np.zeros((1,X.shape[1]))
                elif comp == 'InterSpread':
                    metrics['InterSpread'] = np.zeros((1,X.shape[1],self.classes))
                elif comp == 'Consist':
                    metrics['Consist'] = np.zeros((1,X.shape[1],self.classes))
            
            return metrics
        
        metric_samples = (Nt - win_sz) // self.step_sz + 1

        for comp in self.components:
            if comp == 'Distinct':        
                metrics['Distinct'] = np.zeros((metric_samples,X.shape[1]))
            elif comp == 'InterSpread':
                if self.classes == 3:
                    dists = 3
                elif self.classes == 2:
                    if self.reference == 'common':
                        dists = 2
                    elif self.reference == 'pairwise':
                        dists = 1
                        
                metrics['InterSpread'] = np.zeros((metric_samples,X.shape[1],dists))
            elif comp == 'Consist':
                metrics['Consist'] = np.zeros((metric_samples,X.shape[1],self.classes))
        
        
        for i_s in range(metric_samples):
            for i_f in range(X.shape[1]): # frequency band
            
                means, disp = self.calc_chunk_mean_and_disp(data,win_sz,i_s,i_f)
                    
                inter_spread = _inter_spread(means, self.reference)
                intra_spread = sum(disp)
                    
                # save the metrics for this step/band
                if 'Distinct' in self.components:
                    metrics['Distinct'][i_s,i_f] = sum(inter_spread) / intra_spread
                
                if 'InterSpread' in self.components:
                    metrics['InterSpread'][i_s,i_f,:] = inter_spread
                
                if 'Consist' in self.components:
                    metrics['Consist'][i_s,i_f,:] = 1 / (1 + disp)
                    
        return metrics
