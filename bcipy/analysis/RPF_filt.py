# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:47:18 2020

@author: Nick
"""


import numpy as np
from scipy import signal
from glob import glob
from scipy.io import loadmat, savemat


def filter_data(filts,X):
    """
    Parameters
    ----------
    filts: Tuple of scipy filter object to apply to the data
    X : NUMPY Array
        shape : T x S x Ch
        
        T  - Trials
        S  - Samples per trial
        Ch - Channels
        

    Returns
    -------
    Y : Numpy array 
        shape : T x F x S x Ch
        
        T  - Trials
        F  - Filter 
        S  - Samples per trial
        Ch - Channels

    """
    Nt, Ns, Ne = X.shape
    Y = np.zeros((Nt,len(filts),Ns,Ne))
    
    for i_f in range(len(filts)):
        filt = filts[i_f]
        Y[:,i_f,:,:] = signal.sosfiltfilt(filt,X,axis=-2)
    
    return Y


if __name__ == "__main__":
    files = glob("D:/BCI/BCI_Capture/data/MI_datasets/HighGamma/data/raw-epoch/raw-epoch-*.mat")
    
    
    lp_filter = signal.butter(4,(1,20),btype='bandpass',output='sos',fs=250)
    bs_filter = signal.butter(4,(8,38),btype='bandstop',output='sos',fs=250)
    
    filters = (lp_filter,bs_filter)
    
    for file in files:
        print("Filtering data in file {}".format(file))
        data = loadmat(file)
        
        filtered_classes = {}
        for i_c in range(1,5):
            key = "class" + str(i_c) + "_trials"
            if key in data:
                print("\tFiltering class {} trials".format(i_c))
                filt_data = filter_data(filters,data[key])
                filtered_classes[key] = filt_data
                
        
        
        # save the data
        out_file = file.replace("raw-epoch-","RPF-filt-")
        print("Saving data in file {}".format(out_file))
        savemat(out_file,filtered_classes)
        