# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:31:35 2019

sources.py - Defines classes for getting data generated outside of BCIP

Currently supported sources:
    - Lab Streaming Layer
    - mat files

@author: Nicolas Ivanov, Aaron Lio
"""

# TODO: Enhance file based classes to enable bulk read (i.e. multiple trial)
# capabilities

from .bcip import BCIP
from .bcip_enums import BcipEnums
from scipy.io import loadmat
import numpy as np
import pylsl
import os
import re

##for debugging
#import matplotlib
#import matplotlib.pyplot as plt

class BcipMatFile(BCIP):
    """
    Utility for extracting data from a mat file for BCIP
    
    Parameters
    ----------

    Examples
    --------

    """
    
    def __init__(self,sess,filename,path,label_varname_map, dims=None):
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
        self.dims = dims
        self._file_data = None
        
        # check if the variable names exist in the file
        print("Setting up source from file: {}".format(self.filepath))
        self._file_data = loadmat(self.filepath)
        for varname in label_varname_map.values():
            if not varname in self._file_data:
                # TODO log error
                return
            
            if not self.dims == None:
                # check that the data has the correct number of dimensions
                data_dims = self._file_data[varname].shape
                for i in range(len(dims)):
                    min_channel = min(dims[i])
                    max_channel = max(dims[i])
                    
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
        Poll the data for the next trial of the input label
        """
        
        if self.dims == None:
            # get all the dimensions
            trial_data = class_data[label,self.label_counters[label],:,:]
        else:
            indices = np.ix_((self.label_counters[label],),
                             self.dims[0],
                             self.dims[1])

            trial_data = self._file_data[self.label_varname_map[label]][indices]
        
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

class BcipClassSeparatedMat(BCIP):
    """
    Utility class for extracting class separated data from a mat file for BCIP. 

    Parameters
    ----------

    sess : Session Object
        Session where the BcipClassSeparated data source will exist.

    num_classes : int
        Number of classes within the MAT data

    event_duration : int
        Number of samples during each trial. Should be equal to the number of samples divided by the number of trials, assuming no breaks between trials

    start_index : int
        Sample number at which the trial data to be used, begins. Data before the start_index sample within the MAT source will be ignored.

    end_index : int
        Sample number when the trial to be used, ends. Data after the end_index sample within the MAT source will be ignored

    relative_start : int
        Shift the beginning of each trial start by relative_start samples/

    mat_data_var_name : str
        Name of the mat data array within the .mat file.

    mat_labels_var_name : str
        Name of the labels array within the .mat file.

    link_to_data : str
        Relative path of the mat data to be stored within the created object

    link_to_labels : str
        Relative path of the labels data to be stored within the created object.

    Examples
    --------
    --> Add traceback example with keyerror when mat_data_var_name is incorrect

    Notes
    -----
    --> The imported MAT data to be stored within the object should be in the shape of Number of channels x Number of samples
    --> The MAT labels array should be in the shape of Number of trials x 2, where the first column is the start index of each trial 
        and the second column is the class label of each trial 
        --> The class label of each trial should be numeric.
    """

    def __init__(self,sess, num_classes, event_duration, start_index, end_index, relative_start, mat_data_var_name, mat_labels_var_name ,link_to_data, link_to_labels = None):
        """
        Create a new mat file reader interface
        """
        super().__init__(BcipEnums.SRC,sess)
        
        self.continuous_data = None
        self.class_separated_data = None
        self.link_to_data = link_to_data
        self.link_to_labels = link_to_labels
        self.num_classes = num_classes
        self.event_duration = event_duration
        self.mat_data_var_name = mat_data_var_name
        self.mat_labels_var_name = mat_labels_var_name
        self.relative_start = relative_start

        if link_to_labels != None:
            raw_data = loadmat(link_to_data, mat_dtype = True, struct_as_record = True)
            raw_data = raw_data[mat_data_var_name]
            self.continuous_data = raw_data
            try:
                raw_data = raw_data[:, start_index:end_index]
            except:
                print("Start and/or End index incorrect.")

            labels = loadmat(link_to_labels, mat_dtype = True, struct_as_record = True)
            labels = labels[mat_labels_var_name]
        if link_to_labels == None:
            raw_data = loadmat(link_to_data, mat_dtype = True, struct_as_record = True)
            labels = np.array(raw_data[mat_labels_var_name])
            raw_data = raw_data[mat_data_var_name]
            try:
                raw_data = raw_data[:, start_index:end_index]
            except:
                print("Start and/or End index incorrect.")

        self.label_counters = {}
        
        for i in range(self.num_classes):
            self.label_counters[i] = 0            


        i = 0
        first_row = 0
        last_row = np.size(labels, 0)
        while i < np.size(labels, 0):
            if labels[i][1] <= start_index:
                first_row = i
                i += 1
            elif labels[i][1] >= end_index:
                last_row = i
                i += 1
            else:
                i+=1


        labels = labels[first_row:last_row, :]

        labels_dict = {}

        for label in labels[:, 0]:
            labels_dict[int(label)] = []
            
            if len(labels_dict.keys()) == num_classes:
                break
 
        for label, index in labels:
            labels_dict[int(label)] = np.append(labels_dict[int(label)], index)

        self.labels_dict = labels_dict
        self.raw_data = raw_data
        self.labels = labels

    def poll_data(self, label):
        label = int(label)
        try:
            trial_indices = self.labels_dict[label]
            print(self.label_counters)
            trial_data = self.raw_data[:,int(trial_indices[self.label_counters[label]] + self.relative_start) : int(trial_indices[self.label_counters[label]] + self.event_duration)]
            self.label_counters[label] += 1
            return trial_data

        except KeyError:
            print("Label does not exist")
            return BcipEnums.EXE_FAILURE


    def format_continuous_data(self):
        raw_data = loadmat(self.link_to_data, mat_dtype = True, struct_as_record = True)
        raw_data = np.transpose(raw_data[self.mat_data_var_name])

        labels = loadmat(self.link_to_labels, mat_dtype = True, struct_as_record = True)
        labels = np.array(labels[self.mat_labels_var_name])

        data = {}
        for i in range(1,self.num_classes+1):
            data[i] = np.array([[0]*np.size(raw_data,0)]).T

        for row in range(np.size(labels, 0)):
            data_to_add = [values[int(labels[row][1]):int(labels[row][1]+ self.event_duration)] for values in raw_data ]
            np.concatenate((data[int(labels[row][0])], data_to_add),1)

        self.class_separated_data = data
        return [data, labels]

    @classmethod
    def create_class_separated(cls, sess, num_classes, event_duration, start_index, end_index, relative_start, mat_data_var_name, mat_labels_var_name ,link_to_data, link_to_labels):
        """
        Factory Method for creating class separated MAT File input source. 

        Parameters
        ----------

        sess : Session Object
            Session where the BcipClassSeparated data source will exist.

        num_classes : int
            Number of classes within the MAT data

        event_duration : int
            Number of samples during each trial. Should be equal to the number of samples divided by the number of trials, assuming no breaks between trials

        start_index : int
            Sample number at which the trial data to be used, begins. Data before the start_index sample within the MAT source will be ignored.

        end_index : int
            Sample number when the trial to be used, ends. Data after the end_index sample within the MAT source will be ignored

        relative_start : int
            Shift the beginning of each trial start by relative_start samples/

        mat_data_var_name : str
            Name of the mat data array within the .mat file.

        mat_labels_var_name : str
            Name of the labels array within the .mat file.

        link_to_data : str
            Relative path of the mat data to be stored within the created object

        link_to_labels : str
            Relative path of the labels data to be stored within the created object.

        Examples
        --------
        --> Add traceback example with keyerror when mat_data_var_name is incorrect

        Notes
        -----
        --> The imported MAT data to be stored within the object should be in the shape of Number of channels x Number of samples
        --> The MAT labels array should be in the shape of Number of trials x 2, where the first column is the start index of each trial 
            and the second column is the class label of each trial 
            --> The class label of each trial should be numeric.
        
        
        """
        src = cls(sess, num_classes, event_duration, start_index, end_index, relative_start, mat_data_var_name, mat_labels_var_name ,link_to_data, link_to_labels)

        sess.add_ext_src(src)

        return src

class BcipContinuousMat(BCIP):
    """
    Utility class for extracting continuous from a mat file for BCIP. 

    Parameters
    ----------

    sess : Session Object
        Session where the BcipClassSeparated data source will exist.

    event_duration : int
        Number of samples during each trial. Should be equal to the number of samples divided by the number of trials, assuming no breaks between trials

    start_index : int
        Sample number at which the trial data to be used, begins. Data before the start_index sample within the MAT source will be ignored.

    end_index : int
        Sample number when the trial to be used, ends. Data after the end_index sample within the MAT source will be ignored

    relative_start : int
        Shift the beginning of each trial start by relative_start samples/

    mat_data_var_name : str
        Name of the mat data array within the .mat file.

    mat_labels_var_name : str
        Name of the labels array within the .mat file.

    data_filename : str
        Relative path of the mat data to be stored within the created object

    label_filename : str
        Relative path of the labels data to be stored within the created object.

    Examples
    --------
    --> Add traceback example with keyerror when mat_data_var_name is incorrect

    Notes
    -----
    --> The imported MAT data to be stored within the object should be in the shape of Number of channels x Number of samples
    """

    def __init__(self, sess, event_duration, start_index, end_index, relative_start, channels, mat_data_var_name, mat_labels_var_name ,data_filename, label_filename = None):
        """
        Create a new mat file reader interface
        """
        super().__init__(BcipEnums.SRC,sess)
        
        self.data_filename = data_filename
        self.label_filename = label_filename
        self.event_duration = int(event_duration)
        self.mat_data_var_name = mat_data_var_name
        self.mat_labels_var_name = mat_labels_var_name
        self.relative_start = int(relative_start)

        self.data = loadmat(data_filename, mat_dtype = True, struct_as_record = True)[mat_data_var_name]
        
        if channels == None:
            self.channels = tuple([_ for _ in range(self.data.shape[0])])
        else:
            self.channels = channels
            
        self.data = self.data[self.channels,:]
        Nc, Ns = self.data.shape

        if end_index < 0:
            end_index = Ns + end_index + 1

        # extract the segment defined by the start and end indices
        try:
            self.data = self.data[:, start_index:end_index]
        except:
            print("Start and/or End index invalid.")
            # TODO error log, should probably raise an error here too
            return

        if label_filename != None:
            # labels should be 2D array, first column contain the label, second column contain the timestamp of the label
            self.labels = loadmat(label_filename, mat_dtype = True, struct_as_record = True)[mat_labels_var_name]
        else:
            # labels assumed to be in the same file as data
            self.labels = loadmat(data_filename, mat_dtype = True, struct_as_record = True)[mat_labels_var_name]

        # remove labels that are not within the start and end indices
        self.labels = self.labels.astype(int)
        self.labels = self.labels[self.labels[:,1]<end_index, :]
        self.labels = self.labels[self.labels[:,1]>=start_index, :]

        self._trial_counter = 0

    def poll_data(self, label = None):
        try:
            start = self.labels[self._trial_counter, 1] + self.relative_start
            end = start + self.event_duration
            trial_data = self.data[:, start : end]
            self._trial_counter += 1
            return trial_data

        except IndexError:
            # TODO fix bad return type format, log error, raise exception
            print("Trial data does not exist here. Please ensure you are not polling at time samples outside the provided data's region.")
            return BcipEnums.EXE_FAILURE

    def get_next_label(self):
        return self.labels[self._trial_counter, 0]

    @classmethod
    def create_continuous(cls, sess, data_filename, event_duration = None, start_index = 0, 
                          end_index = -1, relative_start = 0, channels = None,
                          mat_data_var_name = None, 
                          mat_labels_var_name = None, label_filename = None):
        """
        Factory Method for creating continuous MAT File input source. 

        Parameters
        ---------
        sess : Session Object
            Session where the BcipClassSeparated data source will exist.

        event_duration : int
            Number of samples during each trial. Should be equal to the number of samples divided by the number of trials, assuming no breaks between trials

        relative_start : int
            Shift the beginning of each trial start by relative_start samples/
        
        start_index : int
            Sample number at which the trial data to be used, begins. Data before the start_index sample within the MAT source will be ignored.

        end_index : int
            Sample number when the trial to be used, ends. Data after the end_index sample within the MAT source will be ignored

        channels : tuple of ints
            Channel indices to sample

        mat_data_var_name : str
            Name of the mat data array within the .mat file.

        mat_labels_var_name : str
            Name of the labels array within the .mat file.

        data_filename : str
            Relative path of the mat data to be stored within the created object

        label_filename : str
            Relative path of the labels data to be stored within the created object.
        
        """

        src = cls(sess, event_duration, start_index, end_index, 
                  relative_start, channels, mat_data_var_name, 
                  mat_labels_var_name, data_filename, label_filename)

        sess.add_ext_src(src)

        return src


class LSLStream(BCIP):
    """
    An object for maintaining an LSL inlet
    """

    def __init__(self,sess,prop,prop_value,channels=None,
                 marker=True,marker_fmt=None):
        """
        Create a new LSL inlet stream object

        Parameters
        ----------
        sess : session object
            Session object where the data source will exist

        prop : str
            Name of the property to be used to resolve the LSL stream

        prop_value : str
            Property value of the target stream

        channels : tuple of ints
            Index value of channels to poll from the stream, if None all channels will be polled

        marker : bool
            true if there is an associated marker to indicate relative time where data should begin to be polled

        marker_fmt : str
            Regular expression template of the marker to be matched, if none all markers will be matched

        """
        super().__init__(BcipEnums.SRC,sess)
        
        # resolve the stream on the LSL network
        available_streams = pylsl.resolve_byprop(prop,prop_value)
        
        if len(available_streams) == 0:
            # TODO log error
            return
        
        # TODO - Warn about more than one available stream
        
        self.data_inlet = pylsl.StreamInlet(available_streams[0]) # for now, just take the first available stream that matches the property
        self.marker_inlet = None
        self.marker_pattern = None
        
        # TODO - check if the stream has enough input channels to match the
        # channels parameter
        if channels:
            self.channels = channels
        else:
            self.channels = tuple([_ for _ in range(self.data_inlet.channel_count)])
        
        if marker:
            marker_streams = pylsl.resolve_stream('type','Markers')
            self.marker_inlet = pylsl.StreamInlet(marker_streams[0]) # for now, just take the first available marker stream
            # open the inlet
            self.marker_inlet.open_stream()
        
            if marker_fmt:
                self.marker_pattern = re.compile(marker_fmt)

    
    def poll_data(self, Ns):
        """
        Pull data from the inlet stream until we have Ns data points for each
        channel.

        Parameters
        ----------
        Ns: int
            number of samples to collect
        """
        
        if self.marker_inlet != None:
            # start by getting the timestamp for this trial's marker
            t_begin = None
            while t_begin == None:
                marker, t = self.marker_inlet.pull_sample()
                if marker != None:
                    marker = marker[0] # extract the string portion of the marker
                    
                    if (self.marker_pattern == None) or self.marker_pattern.match(marker):
                        t_begin = t
                    
        else:
            t_begin = 0 # i.e. all data is valid
        
        # pull the data in chunks until we get the total number of samples
        trial_data = np.array((len(self.channels), Ns)) # allocate the array
        samples_polled = 0        

        while samples_polled < Ns:
            data, timestamps = self.data_inlet.pull_chunk()
            
            if len(timestamps) != 0:
                # convert data to numpy arrays
                data = np.asarray(data).T
                timestamps = np.asarray(timestamps)
                # throw away data that comes after t_begin
                data = data[:, timestamps > t_begin]
                chunk_sz = data.shape[1]            

                # append the latest chunk to the trial_data array
                if samples_polled + chunk_sz > Ns:
                    dest_end_index = Ns
                    src_end_index = Ns - samples_polled
                else:
                    dest_end_index = samples_polled + chunk_sz
                    src_end_index = chunk_sz

                trial_data[:,samples_polled:dest_end_index] = data[self.channels,:src_end_index]
                samples_polled += chunk_sz
        
        
        # for debugging
        #x = [_  for _ in range(self.Ns)]
        #fig, ax = plt.subplots()
        #plot_data = []
        #for i in range(len(self.channels)):
        #    plot_data.append(x)
        #    plot_data.append(trial_data[i,:]+i*15)
        #plot_data = tuple(plot_data)
        #ax.plot(*plot_data)
        #plt.show()
        
        return trial_data
    
    @classmethod
    def create_marker_coupled_data_stream(cls,sess,prop,prop_value,
                                          channels=None,
                                          marker_fmt=None):
        """
        Create a LSLStream data object that maintains a data stream and a
        marker stream

        Parameters
        ----------

        Examples
        --------
        """
        src = cls(sess,prop,prop_value,channels,True,marker_fmt)
        sess.add_ext_src(src)
        
        return src
    
    @classmethod
    def create_marker_uncoupled_data_stream(cls,sess,prop,prop_value,
                                            channels=None,
                                            marker_fmt="T{},L{},LN{}"):
        """
        Create a LSLStream data object that maintains only a data stream with
        no associated marker stream

        Parameters
        ----------

        Examples
        --------



        """
        src = cls(sess,prop,prop_value,channels,False)
        sess.add_ext_src(src)
        
        return src
