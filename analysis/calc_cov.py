# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:35:38 2020

@author: Nick
"""


import numpy as np
from glob import glob
from scipy.io import loadmat, savemat


def cov_data(X):
    """
    Calculate covariance matrices for each trial
    
    Parameters
    ----------
    X : NUMPY Array
        shape : T x S x Ch
        
        T  - Trials
        F  - Filters
        S  - Samples per trial
        Ch - Channels
        

    Returns
    -------
    Y : Numpy array 
        shape : T x F x S x Ch
        
        T  - Trials
        F  - Filter 
        Ch - Channels
        Ch - Channels

    """
    Nt, Nf, Ns, Nc = X.shape
    
    Y = np.zeros((Nt,Nf,Nc,Nc))
    for i_t in range(Nt):
        for i_f in range(Nf):
            Y[i_t,i_f,:,:] = np.cov(X[i_t,i_f,:,:],rowvar=False)
            
    return Y


if __name__ == "__main__":
    files = glob("D:/BCI/BCI_Capture/data/MI_datasets/HighGamma/data/cropped_data/*.mat")
    
    output_dir = "D:/BCI/BCI_Capture/data/MI_datasets//cov_mats_data/"
    
    for file in files:
        print("Calculating covariance matrices trials in file {}".format(file))
        data = loadmat(file)
        
        cov_mats = {}
        for i_c in range(1,5):
            key = "class" + str(i_c) + "_trials"
            if key in data:
                print("\tCalculating Cov Mats for class {} trials".format(i_c))
                cov_mats[key] = cov_data(data[key])
        
        # save the data
        out_file = file.replace("cropped","cov_mats")
        print("Saving data in file {}".format(out_file))
        savemat(out_file,cov_mats)
        cov_mats = {}