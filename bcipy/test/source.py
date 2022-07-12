# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:31:35 2019

sources.py - Defines classes for getting data generated outside of BCIP

Currently supported sources:
    - Lab Streaming Layer
    - mat files

@author: ivanovn
"""

# TODO: Enhance file based classes to enable bulk read (i.e. multiple trial)
# capabilities

from classes.bcip import BCIP
from classes.bcip_enums import BcipEnums
from scipy.io import loadmat
import numpy as np
import pylsl
import os
##for debugging
import matplotlib
import matplotlib.pyplot as plt

class BcipMatFile(BCIP):
    """
    Utility for extracting data from a mat file for BCIP
    
    """
    
    def __init__(self,sess,filename,path,label_varname_map, link_to_data, link_to_labels, num_classes, event_duration, dims=None):
        """
        Create a new mat file reader interface
        """
        super().__init__(BcipEnums.SRC,sess)
        p = os.path.normpath(os.path.join(os.getcwd(),path))
        f = os.path.join(p,filename)
        if not os.path.exists(p) or not os.path.exists(f) or not os.path.isfile(f):
            # TODO log error
            print("File {} not found in dir {}".format(filename,path))
            return
        
        self.filepath = f
        self.continuous_data = None
        self.link_to_data = link_to_data
        self.link_to_labels = link_to_labels
        self.num_classes = num_classes
        self.event_duration = event_duration
        self.dims = dims
        
        # check if the variable names exist in the file
        print("Setting up source from file: {}".format(self.filepath))
        d = loadmat(self.filepath)
        for varname in label_varname_map.values():
            if not varname in d:
                # TODO log error
                return
            
            if not self.dims == None:
                # check that the data has the correct number of dimensions
                data_dims = d[varname].shape
                for i in range(len(dims)):
                    min_channel = min(dims[i])
                    max_channel = max(dims[i])
                    
                    # ignore the first data dimension b/c its the trial number
                    if min_channel < 0 or min_channel >= data_dims[i+1] \
                       or max_channel < 0 or max_channel >= data_dims[i+1]:
                           # TODO log error
                           return
            
        self.label_varname_map = {}
        # copy the dictionary - converting any string keys into ints
        # a bit hacky, but makes it easier to create MAT file objs with the JSON parser
        for key in label_varname_map:
            if isinstance(key,str):
                self.label_varname_map[int(key)] = label_varname_map[key]
            else:
                self.label_varname_map[key] = label_varname_map[key]
        
        self.label_counters = {}
        
        for label in self.label_varname_map:
            self.label_counters[label] = 0
            
        
    def poll_data(self,label):
        """
        Get the data from the file for the next trial of the input label
        """
        
        data = loadmat(self.filepath)
        class_data = data[self.label_varname_map[label]]
        if self.dims == None:
            # get all the dimensions
            trial_data = class_data[label,self.label_counters[label],:,:]
        else:
            indices = np.ix_((self.label_counters[label],),
                             self.dims[0],
                             self.dims[1])

            trial_data = class_data[indices]
        
        # increment the label counter for this class
        self.label_counters[label] += 1
        
        return trial_data
    

    @classmethod
    def create(cls,sess,filename,path,label_varname_map, dims):
        """
        Factory method for API
        """
        src = cls(sess,filename,path,label_varname_map, dims)
        
        sess.add_ext_src(src)
        
        return src

class BcipContinuousMat(BCIP):
    def __init__(self,sess, num_classes, event_duration, start_index, end_index, link_to_data):
        """
        Create a new mat file reader interface
        """
        super().__init__(BcipEnums.SRC,sess)
        
        self.continuous_data = None
        self.link_to_data = link_to_data
        self.num_classes = num_classes
        self.event_duration = event_duration


        """if link_to_labels:
            raw_data = loadmat(link_to_data, mat_dtype = True, struct_as_record = True)
            raw_data = np.transpose(raw_data['eegdata'])
            self.continuous_data = raw_data
            try:
                raw_data = raw_data[:, start_index:end_index]
            except:
                print("Start and/or End index incorrect.")

            labels = loadmat(link_to_labels, mat_dtype = True, struct_as_record = True)
            labels = np.array(labels['labels'])
        
            data = {}
            for i in range(1,num_classes+1):
                data[i] = np.array([[0]*np.size(raw_data,0)]).T

            i = 0
            first_row = 0
            last_row = np.size(labels, 0)
            while i < np.size(labels, 0):
                if labels[i][1] < start_index:
                    first_row = i
                    i += 1
                elif labels[i][1] > end_index:
                    last_row = i
                    i += 1
                else:
                    i+=1

            labels = labels[first_row:last_row, :]

            for row in range(np.size(labels, 0)):
                data_to_add = [values[int(labels[row][1]):int(labels[row][1]+ event_duration)] for values in raw_data ]
                np.concatenate((data[int(labels[row][0])], data_to_add),1)

            self.class_separated_data = data
            """
        
        raw_data = loadmat(link_to_data, mat_dtype = True, struct_as_record = True)
        labels = np.array(raw_data['labels'])
        raw_data = np.transpose(raw_data['eegdata'])
        try:
            raw_data = raw_data[:, start_index:end_index]
        except:
            print("Start and/or End index incorrect.")

        self.label_counters = {}
        
        for i in range(self.num_classes):
            self.label_counters[i] = 0            

        self.labels = labels

    def poll_continuous_data(self, label):
        total_count = self.label_counters[label]
        i = 0
        cur_count = 0
        while cur_count < total_count:
            if self.labels[i, 0] == label:
                cur_count += 1
            i += 1
        
        start = self.labels[i-1, 1]
        self.label_counters[label] += 1
        end = start + self.event_duration

        return self.poll_time_indexed_data(start, end)
    
    def poll_time_indexed_data(self, start, end):
        return self.continuous_data[:,start:end]
        

    @classmethod
    def create_continuous(cls, sess, num_classes, event_duration, start_index, end_index, link_to_data):

        src = cls(sess, num_classes, event_duration, start_index, end_index, link_to_data)

        sess.add_ext_src(src)

        return src

class LSLStream(BCIP):
    """
    An object for maintaining an LSL inlet
    """

    def __init__(self,sess,prop,prop_value,Ns,labels,channels=None,
                 marker=True,marker_fmt="T{},L{},LN{}"):
        """
        Create a new LSL inlet stream object
        """
        super().__init__(BcipEnums.SRC,sess)
        
        # resolve the stream on the LSL network
        available_streams = pylsl.resolve_byprop(prop,prop_value)
        
        if len(available_streams) == 0:
            # TODO log error
            return
        
        # TODO - Warn about more than one available stream
        
        self.data_inlet = pylsl.StreamInlet(available_streams[0])
        self.Ns = Ns
        self.marker_inlet = None
        self.marker_fmt = marker_fmt
        self.trial_cnt = 0
        
        
        # TODO - check if the stream has enough input channels to match the
        # channels parameter
        self.channels = channels
        
        if marker:
            marker_streams = pylsl.resolve_stream('type','Markers')
            self.marker_inlet = pylsl.StreamInlet(marker_streams[0])
            # open the inlet
            self.marker_inlet.open_stream()
        
        # initialize the label counters
        self.label_counters = {}
        for label in labels:
            self.label_counters[label] = 0
        
    
    def poll_data(self,label):
        """
        Pull data from the inlet stream until we have Ns data points for each
        channel.
        """
        
        if not self.marker_inlet == None:
            # get the timestamp for this trial's 
            target_marker = self.marker_fmt.format(self.trial_cnt,label,
                                                   self.label_counters[label])
            
            # pull the marker sample
            marker = None
            while marker != target_marker:
                marker, t_begin = self.marker_inlet.pull_sample()
                if marker != None:
                    marker = marker[0]
        
        else:
            t_begin = 0 # i.e. all data is valid
        
        # pull the data in chunks until we get the total number of samples
        trial_data = np.array(()) # create an empty array
        
        while trial_data.shape == (0,) or trial_data.shape[0] < self.Ns:
            data, timestamps = self.data_inlet.pull_chunk()
            
            if len(timestamps) != 0:
                # convert data to numpy arrays
                data = np.asarray(data)
                timestamps = np.asarray(timestamps)
                # throw away data that comes after t_begin
                data = data[timestamps > t_begin, :]
            
                # append the latest chunk to the trial_data array
                if trial_data.shape == (0,):
                    trial_data = data
                else:
                    trial_data = np.append(trial_data,data,axis=0)
        
        # Remove any excess data pts from the end of the array and extract the
        # channels of interest
        if self.channels == None:
            channels = [i for i in range(trial_data.shape[1])] 
        else:
            channels = self.channels
        
        indices = np.ix_(tuple([i for i in range(self.Ns)]),channels)
        trial_data = trial_data[indices]
        
        # for debugging
        x = [_  for _ in range(self.Ns)]
        fig, ax = plt.subplots()
        plot_data = []
        for i in range(len(self.channels)):
            plot_data.append(x)
            plot_data.append(trial_data[:,i]+i*15)
        plot_data = tuple(plot_data)
        ax.plot(*plot_data)
        plt.show()
        
        return trial_data
    
    @classmethod
    def create_marker_coupled_data_stream(cls,sess,prop,prop_value,Ns,labels,
                                          channels=None,
                                          marker_fmt="T{},L{},LN{}"):
        """
        Create a LSLStream data object that maintains a data stream and a
        marker stream
        """
        src = cls(sess,prop,prop_value,Ns,labels,channels,True,marker_fmt)
        sess.add_ext_src(src)
        
        return src
    
    @classmethod
    def create_marker_uncoupled_data_stream(cls,sess,prop,prop_value,Ns,labels,
                                            channels=None,
                                            marker_fmt="T{},L{},LN{}"):
        """
        Create a LSLStream data object that maintains only a data stream with
        no associated marker stream
        """
        src = cls(sess,prop,prop_value,Ns,labels,channels,False)
        sess.add_ext_src(src)
        
        return src