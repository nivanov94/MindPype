# -*- coding: utf-8 -*-
"""
Created on Thu May 28 12:47:48 2020

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
        shape : Cl x T x F x S x Ch
        
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
    files = glob("D:/BCI/BCI_Capture/data/MI_datasets/HighGamma/data/preprocessed_data/preprocessed-*.mat")
    
    output_dir = "D:/BCI/BCI_Capture/data/MI_datasets/HighGamma/data/filtered_data/"
    
    filt_ranges = ((4,10),
                   (7,13),
                   (10,18),
                   (14,22),
                   (18,26),
                   (22,30),
                   (26,34),
                   (30,38),
                   (8,10),
                   (10,12),
                   (12,18),
                   (18,30),
                   (8,30))
    
    filters = [signal.butter(4,fr,btype='bandpass',output='sos',fs=250) for fr in filt_ranges]
    
    for file in files:
        print("Filtering data in file {}".format(file))
        data = loadmat(file)
        
        filtered_classes = {}
        for i_c in range(1,5):
            key = "class" + str(i_c) + "_trials"
            if key in data:
                print("\tFiltering class {} trials".format(i_c))
                filt_data = filter_data(filters,data[key])
                # append a no-filter slice at the beginning
                unfiltered_data = np.expand_dims(data[key],1)
                filt_data = np.concatenate((unfiltered_data,filt_data),axis=1)
                filtered_classes[key] = filt_data
                
        
        
        # save the data
        out_file = file.replace("preprocessed","filtered")
        print("Saving data in file {}".format(out_file))
        savemat(out_file,filtered_classes)
        
