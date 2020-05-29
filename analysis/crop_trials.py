# -*- coding: utf-8 -*-
"""
Created on Thu May 28 14:49:29 2020

@author: Nick
"""


import numpy as np
from glob import glob
from scipy.io import loadmat, savemat


def crop_data(front,back,X):
    """
    Removes samples from the front and back of trials for data processing
    
    Parameters
    ----------
    front : int
    back : int
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
        S  - Samples per trial
        Ch - Channels

    """
    return X[:,:,front+1:-back,:]


if __name__ == "__main__":
    files = glob("D:/BCI/BCI_Capture/data/MI_datasets/Kayaetal/filtered_data/*.mat")
    
    output_dir = "D:/BCI/BCI_Capture/data/MI_datasets/Kayaetal/cropped_trials/"
    
    
    # amount to crop from beginning and end in seconds
    
    # BCI Comp IV-2a: Front-3, back-0.5
    # Cho et al: Front-1.5 back-1
    # HighGamma: Front-0.5, back-1
    # Kaya et al. : Front-1.5, back-1.5
    
    front_s = 1.5
    back_s = 1.5
    
    Fs = 250
    
    # amount to crop from front and back in samples
    front = round(front_s * Fs)
    back = round(back_s * Fs)
    
    for file in files:
        print("Cropping trials in file {}".format(file))
        data = loadmat(file)
        
        cropped_classes = {}
        for i_c in range(1,5):
            key = "class" + str(i_c) + "_trials"
            if key in data:
                print("\tCropping class {} trials".format(i_c))
                cropped_classes[key] = crop_data(front,back,data[key])
        
        # save the data
        out_file = file.replace("filtered","cropped")
        print("Saving data in file {}".format(out_file))
        savemat(out_file,cropped_classes)