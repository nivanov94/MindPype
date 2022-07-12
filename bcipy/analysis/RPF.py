# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:57:15 2020

@author: Nick
"""

from math import exp, log, sqrt
import numpy as np
from scipy import signal, linalg, dot

from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.distance import distance_riemann

from scipy.io import loadmat
import json


def _cov(X):
    return np.cov(X,rowvar=False)

def _dist_mean(S, T, stats='arithmetic'):
    """
    Calculate the mean distance of covariance matrices within T and the 
    covariance matrix in S using arithmetic or geometric statistics.
    """
    
    
    if stats == 'arithmetic':
        mu = 1/T.shape[0] * sum([distance_riemann(S,T[i,:,:]) 
                                      for i in range(T.shape[0])])
    else:
        mu = exp(1/T.shape[0]*sum([log(distance_riemann(S,T[i,:,:]))
                                        for i in range(T.shape[0])]))

    return mu

def _dist_std(S, T, stats='arithmetic'):
    """
    Calculate the standard deviation of the distance of covariance matrices 
    within T and the covariance matrix in S
    using arithmetic or geometric statistics.
    """

    d = [distance_riemann(S,T[i,:,:]) for i in range(T.shape[0])]
    mu = _dist_mean(S,T,stats)

    if stats == 'arithmetic':
        sigma = sqrt(1/T.shape[0] * sum([(di-mu)**2 for di in d]))
    else:
        sigma = exp(sqrt(1/T.shape[0] * sum([log(di/mu)**2 for di in d])))
        
    return sigma

def _z_score(d,mu,sigma,stats='arithmetic'):
    if stats == 'arithmetic':
        return (d - mu) / sigma
    else:
        return log(d/mu) / log(sigma)


def potato_filt(X):
    XXt = np.zeros((X.shape[0],X.shape[2],X.shape[2]))
        
    for i in range(X.shape[0]):
        
        # calculate the covariance matrices
        Xcov = _cov(X[i,:,:])
            
        # regularize the cov mat
        r = 0.01
        XXt[i,:,:] = 1/(1+r)*(Xcov + r*np.eye(Xcov.shape[0]))

    rejected_trials = []

    for i in range(5):
        X_clean = []
        i_rej = []
        # compute the mean covariance matrix
        S  = mean_covariance(XXt)
        mu = _dist_mean(S,XXt,'geometric')
        sigma = _dist_std(S,XXt,'geometric')
                            
        for j in range(XXt.shape[0]):
            if _z_score(distance_riemann(XXt[j,:,:], S),
                        mu,sigma,'geometric') < 3:
                X_clean.append(XXt[j,:,:])
            else:
                i_rej.append(j)
            
        if len(X_clean) == XXt.shape[0]:
            break
            
        XXt = np.stack(X_clean)
        
        # update list of rejected trials
        for i_r in range(len(i_rej)):
            offset = sum([1 for ii in rejected_trials if i_rej[i_r] >= ii])
            i_rej[i_r] = offset+i_rej[i_r]
    
        rejected_trials.extend(i_rej)
    return rejected_trials

def extract_intermed_values(values, min_value, max_value=None):
    interm = []
    
    for v in values:
        if v >= min_value and ((max_value is None) or v < max_value):
            interm.append(v)
    
    interm.sort()
    for i in range(len(interm)):
        interm[i] -= min_value
        
    return interm

def main():
    dataset = 'Kaya'
    Nclasses = 4
    
    artifact_trials = {}
    
    potato_params = (
        ((2,3),0),
        ((4,5),0),
        ((6,7),0),
        ((14,15),0),
        ((18,19),0),
        ((19,20),0),
        ((2,3,4,5),1),
        ((6,7,14,15),1),
        ((15,18,19,20),1)
        )

    for i_p in range(2,15):
        artifact_trials[i_p] = {}
        print("Concatenating data for participant {}".format(i_p))
        
        pdata = loadmat("HighGamma/data/raw-epoch/cropped-{}.mat".format(i_p))
        
        

        data = None
        
        for i_t in range(1,Nclasses+1):
            cdata = pdata['class{}_trials'.format(i_t)]
        
            if data is None:
                class_indices = [0]
                data = cdata
            else:
                class_indices.append(data.shape[0])
                data = np.concatenate((data,cdata),axis=0)
                
        
        rejected = set()
        for pp in potato_params:
            Xf = data[:,pp[1],:,:]
            X = Xf[:,:,pp[0]]
            rejected.update(potato_filt(X))
        
        print("\t" + str(class_indices))
        print("\t" + str(sorted(list(rejected))))

        artifact_trials[i_p]['c1_rejects'] = extract_intermed_values(rejected,
                                                                   class_indices[0],
                                                                   class_indices[1])
        artifact_trials[i_p]['c2_rejects'] = extract_intermed_values(rejected,
                                                                   class_indices[1],
                                                                   class_indices[2])
        artifact_trials[i_p]['c3_rejects'] = extract_intermed_values(rejected,
                                                                    class_indices[2],
                                                                    class_indices[3])
        artifact_trials[i_p]['c4_rejects'] = extract_intermed_values(rejected,
                                                                     class_indices[3])

    f = open("high_gamma_artifact_trials.json",'w')
    json.dump(artifact_trials,f,indent=2)
    f.close()
        
if __name__ == "__main__":
    main()