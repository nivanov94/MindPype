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

from scipy.io import loadmat
import numpy as np
import pylsl
import os

class BcipMatFile:
    """
    Utility for extracting data from a mat file for BCIP
    
    This object
    """
    
    def __init__(self,filename,path,label_varname_map,dims=None):
        """
        Create a new mat file reader interface
        """
        p = os.path.normpath(os.path.join(os.getcwd(),path))
        f = os.path.join(p,filename)
        if not os.path.exists(p) or not os.path.exists(f) or not os.path.isfile(f):
            # TODO log error
            return
        
        self.filepath = f
        
        self.dims = dims
        
        # check if the variable names exist in the file
        d = loadmat(self.filepath)
        for varname in label_varname_map.values():
            if not varname in d:
                # TODO log error
                return
            
            if not self.dims == None:
                # check that the data has the correct number of dimensions
                data_dims = d[varname].shape
                for i in range(len(dims)): # ignore the first dimension b/c its the trial number
                    min_channel = min(dims[i])
                    max_channel = max(dims[i])
                    
                    if min_channel < 0 or min_channel < data_dims[i] \
                       or max_channel >= data_dims[i]:
                           # TODO log error
                           return
            
        self.label_varname_map = label_varname_map
        self.label_counters = {}
        
        for label in label_varname_map:
            self.label_counters[label] = 0
            
        
    def pollData(self,label):
        """
        Get the data from the file for the next trial of the input label
        """
        
        data = loadmat(self.filepath)
        class_data = data[self.label_varname_map[label]]
        if self.dims == None:
            # get all the dimensions
            trial_data = class_data[self.label_counters[label],:,:]
        else:
            indices = np.ix_((self.label_counters[label],),
                             self.dims[0],
                             self.dims[1])

            trial_data = class_data[indices]
        
        # increment the label counter for this class
        self.label_counters[label] += 1
        
        return trial_data
    
    @classmethod
    def create(cls,filename,path,label_varname_map,dims):
        """
        Factory method for API
        """
        return cls(filename,path,label_varname_map,dims)


class LSLStream:
    """
    An object for maintaining an LSL inlet
    """

    def __init__(self,stream_type,labels,Ns,channels=None,
                 marker=True,marker_fmt="T{},L{},LN{}"):
        """
        Create a new LSL inlet stream object
        """
        
        # resolve the stream on the LSL network
        available_streams = pylsl.resolve_stream('type',stream_type)
        
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
        
    
    def pollData(self,label):
        """
        Pull data from the inlet stream until we have Ns data points for each
        channel.
        """
        
        if not self.marker_inlet == None:
            # get the timestamp for this trial's 
            target_marker = self.marker_fmt.format(self.trial_cnt,label,
                                                   self.label_counter[label])
            
            # pull the marker sample
            marker = None
            while marker != target_marker:
                marker, t_begin = self.marker_inlet.pull_sample()
        
        else:
            t_begin = 0 # i.e. all data is valid
        
        # pull the data in chunks until we get the total number of samples
        trial_data = np.array(()) # create an empty array
        
        while trial_data.shape[1] < self.Ns:
            data, timestamps = self.data_inlet.pull_chunk()
            
            # throw away data that comes after t_begin
            data = data[:,timestamps > t_begin]
            
            # append the latest chunk to the trial_data array
            trial_data = np.append(trial_data,data,axis=1)
        
        # Remove any excess data pts from the end of the array and extract the
        # channels of interest
        if self.channels == None:
            channels = [i for i in range(trial_data.shape[0]-1)] # -1 b/c last row contains no data
        else:
            channels = self.channels
        
        indices = np.ix_(channels,tuple([i for i in range(self.Ns)]))
        trial_data = trial_data[indices].T # transpose to make channels columns
        
        return trial_data
    
    @classmethod
    def createMarkerCoupledDataStream(cls,stream_type,Ns,labels,
                                      channels=None,marker_fmt="T{},L{},LN{}"):
        """
        Create a LSLStream data object that maintains a data stream and a
        marker stream
        """
        return cls(stream_type,Ns,labels,channels,True,marker_fmt)
    
    @classmethod
    def createMarkerUncoupledDataStream(cls,stream_type,Ns,labels,
                                        channels=None,marker_fmt="T{},L{},LN{}"):
        """
        Create a LSLStream data object that maintains only a data stream with
        no associated marker stream
        """
        return cls(stream_type,Ns,labels,channels,False)