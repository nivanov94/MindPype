import sys, os
sys.path.insert(0, os.getcwd())

from bcipy import bcipy
from scipy.io import loadmat
import numpy as np
import pylsl
import os
##for debugging
import matplotlib
import matplotlib.pyplot as plt
    
def give_cont_data(label, label_counters, continuous_data, event_duration):
    class_data = continuous_data[label]
    trial_data = class_data[:,event_duration*label_counters[label] : event_duration*label_counters[label] +event_duration]
    label_counters[label] += 1
    print(np.shape(trial_data))
    print(label_counters)
    return trial_data

def format_continuous_data(link_to_data, link_to_labels, num_classes, event_duration):
    raw_data = loadmat(link_to_data, mat_dtype = True, struct_as_record = True)
    labels = loadmat(link_to_labels, mat_dtype = True, struct_as_record = True)
    
    raw_data = np.transpose(raw_data['eegdata'])
    labels = np.array(labels['labels'])

    data = {}
    for i in range(1,num_classes+1):
        data[i] = np.array([[0]*np.size(raw_data,0)]).T

    for row in range(np.size(labels, 0)):
        data_to_add = [values[int(labels[row][1]):int(labels[row][1] + event_duration)] for values in raw_data]
        data[int(labels[row][0])] = np.hstack((data[int(labels[row][0])], data_to_add))
        #print(f"{labels[row][0]} class data is now size {np.shape(data[int(labels[row][0])])}")
    
    return data
    
def main():
    label_counters = {1:0, 2:0}
    data = format_continuous_data('data\eegdata.mat', 'data\labels.mat', 2, 800)
    trial1 = give_cont_data(1, label_counters, data, 800)
    trial2 = give_cont_data(2, label_counters, data, 800)

if __name__ == '__main__':
    main()