"""
Currently supported sources:
    - Lab Streaming Layer
    - mat files
    - xdf files


"""

# TODO: Enhance file based classes to enable bulk read (i.e. multiple trial)
# capabilities


from .core import BCIP, BcipEnums
from .containers import CircleBuffer, Tensor
from scipy.io import loadmat
import numpy as np
import pylsl
import pyxdf
import os
import re
import sys
import warnings
import liesl
import threading
import time

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
            
            if not dims == None:
                # check that the data has the correct number of dimensions
                data_dims = self._file_data[varname].shape
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
            
        
    def poll_data(self,Ns,label):
        """
        Poll the data for the next trial of the input label
        """
        
        class_data = self._file_data[self.label_varname_map[label]]
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
        * Relative path of the mat data to be stored within the created object

    link_to_labels : str
        * Relative path of the labels data to be stored within the created object.

    Examples
    --------
    * Add traceback example with keyerror when mat_data_var_name is incorrect

    Notes
    -----
    * The imported MAT data to be stored within the object should be in the shape of Number of channels x Number of samples
    * The MAT labels array should be in the shape of Number of trials x 2, where the first column is the start index of each trial and the second column is the class label of each trial 
        * The class label of each trial should be numeric.
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

    def poll_data(self, Ns, label):
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
    def create_class_separated(cls, sess, num_classes, event_duration, start_index, end_index, relative_start, mat_data_var_name, mat_labels_var_name, link_to_data, link_to_labels):
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
            Relative path of the mat data to be stored within the created object.


        link_to_labels : str
            Relative path of the labels data to be stored within the created object.

        Examples
        --------
        * Add traceback example with keyerror when mat_data_var_name is incorrect

        Notes
        -----
        * The imported MAT data to be stored within the object should be in the shape of Number of channels x Number of samples
        * The MAT labels array should be in the shape of Number of trials x 2, where the first column is the start index of each trial and the second column is the class label of each trial 
            * The class label of each trial should be numeric.
        
        
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

    def poll_data(self, Ns, label = None):
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


class BcipXDF(BCIP):
    """
    Utility class for extracting trial data from an XDF file for BCIP. 

    Parameters
    ----------

    sess : Session Object
        Session where the BcipXDF data source will exist.

    files : list of str
        XDF file(s) where data should be extracted from.

        
    tasks : list or tuple of strings 
        List or Tuple of strings corresponding to the tasks to be completed by the user.
        For P300-type setups, the tasks 'target' and 'non-target'/'flash' can be used.

    channels : list or tuple of int
        Values corresponding to the EEG channels used during the session

    relative_start : float, default = 0
        Value corresponding to the start of the trial relative to the marker onset.

    Ns : int, default = 1
        Number of samples to be extracted per trial. For epoched and class-separated data, this value determines the
        size of each epoch, whereas this value is used in polling for continuous data.    
        
    mode : 'continuous', 'class-separated' or 'epoched', default = 'epoched'
        Mode indicates whether the inputted data will be epoched sequentially as individual trials,
        epoched by class, or to leave the data in a continuous format

    .. warning:: 
       The task list used in the BcipXDF object MUST REFLECT the task list used in the XDF file.
       Differences will cause the program to fail.
    """

    def __init__(self, sess, files, tasks, channels, relative_start = 0, Ns = 1, mode = 'epoched'):
        """
        Create a new xdf file reader interface
        """
        super().__init__(BcipEnums.SRC,sess)
        
        if type(files) == str:
            files = [files]

        self.files = files
        self.relative_start = relative_start
        self.Ns = Ns
        self.tasks = tasks
        self.channels = channels
        self.label_counter = None
        self.mode = mode
        self._continuous_data = None
        self._continuous_labels = None

        
        trial_data = {task: [] for task in tasks}
        
        if mode == 'class-separated':
            combined_marker_streams = {'time_series': None, 'time_stamps': None}
            for filename in files:
                print(filename)
                data, header = pyxdf.load_xdf(filename)
                
                for stream in data:
    
                    if stream['info']['type'][0] == 'Marker' or stream['info']['type'][0] == 'Markers': #change to Markers after testing
                        marker_stream = stream
    
                    elif stream['info']['type'][0] == 'EEG':
                        eeg_stream = stream
                    
                sample_indices = np.full(eeg_stream['time_stamps'].shape, False) # used to extract EEG samples, pre-allocated here
                self._continuous_data = eeg_stream['time_series'][:,channels]
                self._continuous_labels = marker_stream['time_series']
                #print(eeg_stream['time_series'].shape)
                total_tasks = 0
                
                for i_m, markers in enumerate(marker_stream['time_series']):
                    marker_value = markers[0]
                    curr_task = ''
                                        
                    for task in self.tasks:
                        if task in marker_value:
                            curr_task = task
    
                            marker_time = marker_stream['time_stamps'][i_m]
                            total_tasks += 1
                            
                            # compute the correct start and end indices for the current trial
                            eeg_window_start = marker_time + relative_start
                            #eeg_window_end = marker_time + 0.8 +(5/Fs) # Added temporal buffer to limit indexing errors
        
                            sample_indices = np.array(eeg_stream['time_stamps'] >= eeg_window_start)
                            
                            sample_data = eeg_stream['time_series'][sample_indices, :][:, channels].T # Nc X len(eeg_stream)
                            trial_data[curr_task].append(sample_data[:,:][:, :int(Ns)]) #Nc x Ns
                if combined_marker_streams['time_series'] is None:
                    combined_marker_streams['time_series'] = marker_stream['time_series']
                    combined_marker_streams['time_stamps'] = marker_stream['time_stamps']
                else:
                    combined_marker_streams['time_series'] = np.concatenate((combined_marker_streams['time_series'], marker_stream['time_series']), axis=0)
                    combined_marker_streams['time_stamps'] = np.concatenate((combined_marker_streams['time_stamps'], marker_stream['time_stamps']), axis=0)
        
            for task in trial_data:
                trial_data[task] = np.stack(trial_data[task], axis=0) # Nt x Nc x Ns
                
            self.trial_data = {'EEG': {'time_series': trial_data, 'time_stamps': eeg_stream}, 'Markers': combined_marker_streams}
            self.label_counter = {task: 0 for task in tasks}

        elif mode == 'continuous':
            
            eeg_stream = None
            marker_stream = None

            for filename in files:
                data, header = pyxdf.load_xdf(filename)
                
                for stream in data:
                    if stream['info']['type'][0] == 'Marker' or stream['info']['type'][0] == 'Markers': #change to Markers after testing
                        if marker_stream:
                            marker_stream['time_series'] = np.concatenate((marker_stream['time_series'], stream['time_series']), axis=0)
                            marker_stream['time_stamps'] = np.concatenate((marker_stream['time_stamps'], stream['time_stamps']), axis=0)
                        else:
                            marker_stream = stream
    
                    elif stream['info']['type'][0] == 'EEG':
                        if eeg_stream:
                            eeg_stream['time_series'] = np.concatenate((eeg_stream['time_series'], stream['time_series'].T[channels, :][:,:]), axis=0)
                            eeg_stream['time_stamps'] = np.concatenate((eeg_stream['time_stamps'], stream['time_stamps'][channels,:]), axis=0)
                        else:
                            eeg_stream = stream
                            eeg_stream['time_series'] = stream['time_series'].T[channels, :][:,:]
                            eeg_stream['time_stamps'] = stream['time_stamps'][channels,]

            self.trial_data = {'EEG': eeg_stream, 'Markers': marker_stream} 
            #print(self.trial_data['EEG']['time_stamps'][-100:], self.trial_data['Markers']['time_stamps'][-10:])
            self.label_counter = {task: 0 for task in tasks}

        elif mode == 'epoched':
            self.epoched_counter = 0
            eeg_stream_data = None
            eeg_stream_stamps = None
            eeg_stream = None
            marker_stream = None
            Ns = int(Ns)
            epoch_num = 0
            


            for filename in files:
                # Load the data from the current xdf file
                data, header = pyxdf.load_xdf(filename)
                # Iterate through all streams in every file, add current file's data to the previously loaded data
                for stream in data:
                    if stream['info']['type'][0] == 'Marker' or stream['info']['type'][0] == 'Markers': #change to Markers after testing
                        if marker_stream:
                            marker_stream['time_series'] = np.concatenate((marker_stream['time_series'], stream['time_series']), axis=0)
                            marker_stream['time_stamps'] = np.concatenate((marker_stream['time_stamps'], stream['time_stamps']), axis=0)
                        else:
                            marker_stream = stream
                    
                    elif stream['info']['type'][0] == 'EEG':
                        if eeg_stream:
                            eeg_stream['time_series'] = np.concatenate((eeg_stream['time_series'], stream['time_series']), axis=0)
                            eeg_stream['time_stamps'] = np.concatenate((eeg_stream['time_stamps'], stream['time_stamps']), axis=0)
                        else:
                            eeg_stream = stream

            eeg_stream_data = np.zeros((len(marker_stream['time_stamps']), Ns, len(channels)))
            eeg_stream_stamps = np.zeros((len(marker_stream['time_stamps']), Ns))

            # Actual epoching operation
            while epoch_num < len(marker_stream['time_stamps']):
                # Find the marker value where the current epoch starts
                marker_time = marker_stream['time_stamps'][epoch_num]
                # Correct the starting time of the epoch based on the relative start time
                eeg_window_start = marker_time + relative_start


                # Find the index of the first sample after the marker
                first_sample_index = np.where(eeg_stream['time_stamps'] >= eeg_window_start == True)[0][0]
                # Find the index of the last sample in the window
                final_sample_index = first_sample_index + Ns 
                # Extract the data from the eeg stream
                eeg_stream_data[epoch_num, :, :] = eeg_stream['time_series'][first_sample_index:final_sample_index, :][:, channels]
                eeg_stream_stamps[epoch_num, :] = eeg_stream['time_stamps'][first_sample_index:final_sample_index]
                
                epoch_num += 1

            self.trial_data = {'EEG': {'time_stamps': eeg_stream_stamps, 'time_series': eeg_stream_data}, 'Markers': marker_stream} 
            
            

    def poll_data(self, Ns = 1, label = None):
        #TODO: implement polling
        """
        Polls the data source for new data.

        Parameters
        ----------

        label : string
            Marker to be used for polling. Number of trials previously extracted is recorded internally.
            This marker must be present in the XDF file and must be present in the list of tasks used in
            initialization.

        Ns : int, default = 1
            Number of samples to be extracted per trial. For continuous data, determines the size of
            the extracted sample. This value is disregarded for epoched and class-separated data.   
        """

        if self.mode == 'class-separated':
            # Extract sample data from epoched trial data and increment the label counter
            sample_data = self.trial_data[label][self.label_counter[label], :, :]
            self.label_counter[label] += 1

            return sample_data
        
        elif self.mode == 'epoched':
            # Extract sample data from epoched trial data and increment the label counter
            sample_data = self.trial_data['EEG']['time_series'][self.epoched_counter, :, :]
            self.epoched_counter += 1

            return sample_data
        
        elif self.mode == 'continuous':
            # Find the index of the marker in the marker stream data
            lst_of_marker_indices = []
            for i in range(len(self.trial_data['Markers']['time_series'])):
                if label in self.trial_data['Markers']['time_series'][i] or label in self.trial_data['Markers']['time_series'][i][0]:
                    lst_of_marker_indices.append(i)
            
            index = lst_of_marker_indices[self.label_counter[label]]
            
            # Extract the corresponding marker timestamp
            eeg_window_start = self.trial_data['Markers']['time_stamps'][index] + self.relative_start

            # Construct the boolean array for samples that fall after the marker timestamp
            sample_indices = np.array(self.trial_data['EEG']['time_stamps'] >= eeg_window_start)

            print(eeg_window_start, self.trial_data['EEG']['time_stamps'][sample_indices][179])
            #while np.sum(sample_indices) < Ns:
            #    eeg_window_start = eeg_window_start + (.2*self.relative_start)

                # Construct the boolean array for samples that fall after the marker timestamp
            #    sample_indices = np.array(self.trial_data['EEG']['time_stamps'] >= eeg_window_start)

            
            sample_data = self.trial_data['EEG']['time_series'][sample_indices, :][:, self.channels].T # Nc X len(eeg_stream)
            sample_data = sample_data[:, :Ns] #Nc x Ns
            self.label_counter[label] += 1
            return sample_data
    
    def load_into_tensor(self):
        """
        Loads entirity of BCIPXDF data object into a tensor.
        Returns two BCIPy Tensor objects; one containing the data and the other containing the labels.
        """
        if self.trial_data and self.mode == 'continuous':
            ret = Tensor.create_from_data(self.session, self.trial_data['EEG']['time_series'].shape, self.trial_data['EEG']['time_series'])
            ret_labels = Tensor.create_from_data(self.session, self.trial_data['Markers']['time_series'].shape, self.trial_data['Markers']['time_series'])

        elif self.trial_data and self.mode == 'epoched':
            ret = Tensor.create_from_data(self.session,self.trial_data['EEG']['time_series'].shape, self.trial_data['EEG']['time_series'])
            ret_labels = Tensor.create_from_data(self.session, self.trial_data['Markers']['time_series'].shape, self.trial_data['Markers']['time_series'])

        elif self.trial_data and self.mode == 'class-separated':
            warnings.warn('Class-separated data is not yet supported for Tensor loading. This must be performed manually. Use tensor.create_from_data() with the appropriate class-separated dataset to create the tensor', RuntimeWarning, stacklevel=5)
            #ret = Tensor.create_from_data(self.session, self.trial_data['EEG']['time_series'].shape, self.trial_data['EEG']['time_series'])
            #ret_labels = Tensor.create_from_data(self.session, self.trial_data['Markers']['time_series'].shape, self.trial_data['Markers']['time_series'])

            #RuntimeWarning('Class-separated data is not yet supported for Tensor loading. This must be performed manually.')

        return ret, ret_labels

    @classmethod
    def create_continuous(cls, sess, files, tasks, channels, relative_start = 0, Ns = 1):
        """
        Factory Method for creating continuous XDF File input source. 

        
        Parameters
        ---------
        sess : Session Object
            Session where the BcipXDF data source will exist.

        files : list of str
            XDF file(s) where data should be extracted from.
       
        tasks : list or tuple of strings 
            List or Tuple of strings corresponding to the tasks to be completed by the user.
            For P300-type setups, the tasks 'target' and 'non-target'/'flash' can be used.

        channels : list or tuple of int
            Values corresponding to the EEG channels used during the session

        relative_start : float, default = 0
            Value corresponding to the start of the trial relative to the marker onset.

        Ns : int, default = 1
            Number of samples to be extracted per trial. For epoched data, this value determines the
            size of each epoch, whereas this value is used in polling for continuous data. 

        """

        src = cls(sess, files, tasks, channels, relative_start, Ns, mode = 'continuous')

        sess.add_ext_src(src)

        return src

    @classmethod
    def create_class_separated(cls, sess, files, tasks, channels, relative_start = 0, Ns = 1):
        """
        Factory Method for creating class-separated XDF File input source. 

        Parameters
        ---------
        sess : Session Object
            Session where the BcipXDF data source will exist.

        files : list of str
            XDF file(s) where data should be extracted from.
       
        tasks : list or tuple of strings 
            List or Tuple of strings corresponding to the tasks to be completed by the user.
            For P300-type setups, the tasks 'target' and 'non-target'/'flash' can be used.

        channels : list or tuple of int
            Values corresponding to the EEG channels used during the session

        relative_start : float, default = 0
            Value corresponding to the start of the trial relative to the marker onset.

        Ns : int, default = 1
            Number of samples to be extracted per trial. For class-separated data, this value determines the
            size of each epoch, whereas this value is used in polling for continuous data.    
            
        """
    
        src = cls(sess, files, tasks, channels, relative_start, Ns, mode = 'class-separated')

        sess.add_ext_src(src)

        return src
    
    @classmethod
    def create_epoched(cls, sess, files, tasks, channels, relative_start = 0, Ns = 1):
        """
        Factory Method for creating epoched XDF File input source. 

        Parameters
        ---------
        sess : Session Object
            Session where the BcipXDF data source will exist.

        files : list of str
            XDF file(s) where data should be extracted from.
       
        tasks : list or tuple of strings 
            List or Tuple of strings corresponding to the tasks to be completed by the user.
            For P300-type setups, the tasks 'target' and 'non-target'/'flash' can be used.

        channels : list or tuple of int
            Values corresponding to the EEG channels used during the session

        relative_start : float, default = 0
            Value corresponding to the start of the trial relative to the marker onset.

        Ns : int, default = 1
            Number of samples to be extracted per trial. For class-separated data, this value determines the
            size of each epoch, whereas this value is used in polling for continuous data.    
        
        """
        src = cls(sess, files, tasks, channels, relative_start, Ns, mode = 'epoched')
        
        sess.add_ext_src(src)
        
        return src
    
    @classmethod
    def create(cls, sess, files, tasks, channels, relative_start = 0, Ns = 1, mode = 'epoched'):
        """
        Factory Method for creating epoched XDF File input source. 

        Parameters
        ---------
        sess : Session Object
            Session where the BcipXDF data source will exist.

        files : list of str
            XDF file(s) where data should be extracted from.
       
        tasks : list or tuple of strings 
            List or Tuple of strings corresponding to the tasks to be completed by the user.
            For P300-type setups, the tasks 'target' and 'non-target'/'flash' can be used.

        channels : list or tuple of int
            Values corresponding to the EEG channels used during the session

        relative_start : float, default = 0
            Value corresponding to the start of the trial relative to the marker onset.

        Ns : int, default = 1
            Number of samples to be extracted per trial. For epoched data, this value determines the
            size of each epoch, whereas this value is used in polling for continuous data.    
            
        """
    
        src = cls(sess, files, tasks, channels, relative_start, Ns, mode)

        sess.add_ext_src(src)

        return src


class LSLStream(BCIP):
    """
    An object for maintaining an LSL inlet
    """

    def __init__(self,sess,pred,channels=None,
                 marker=True,marker_fmt=None,marker_pred=None):
        """
        Create a new LSL inlet stream object
        Parameters
        ----------
        sess : session object
            Session object where the data source will exist
        pred : str
            The predicate string, e.g. "name='BioSemi'" or "type='EEG' and starts-with(name,'BioSemi') and 
            count(description/desc/channels/channel)=32"
        prop_value : str
            Property value of the target stream
        channels : tuple of ints
            Index value of channels to poll from the stream, if None all channels will be polled
        marker : bool
            true if there is an associated marker to indicate relative time where data should begin to be polled
        marker_fmt : str
            Regular expression template of the marker to be matched, if none all markers will be matched
        marker_pred : str
            The predicate string for the marker stream
        """
        super().__init__(BcipEnums.SRC,sess)
        
        # resolve the stream on the LSL network
        available_streams = pylsl.resolve_bypred(pred)
        
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
            marker_streams = pylsl.resolve_bypred(marker_pred)
            self.marker_inlet = pylsl.StreamInlet(marker_streams[0]) # for now, just take the first available marker stream
            # open the inlet
            self.marker_inlet.open_stream()
        
            if marker_fmt:
                self.marker_pattern = re.compile(marker_fmt)

    
    def poll_data(self, Ns, label):
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
        trial_data = np.zeros((len(self.channels), Ns)) # allocate the array
        samples_polled = 0        

        while samples_polled < Ns:
            data, timestamps = self.data_inlet.pull_chunk()
            timestamps = np.asarray(timestamps)

            if len(timestamps) != 0 and np.any(timestamps > t_begin):
                # convert data to numpy arrays
                data = np.asarray(data).T
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


class InputLSLStream(BCIP):
    """
    An object for maintaining an LSL inlet

    Attributes
    ----------
    data_buffer : dict 
        {'EEG': np.array, 'time_stamps': np.array}
        A dictionary containing the data and time stamps from past samples (used when trials have overlapping data)

    data_inlet : pylsl.StreamInlet
        The LSL inlet object

    marker_inlet : pylsl.StreamInlet
        The LSL inlet object for the marker stream

    marker_pattern : re.Pattern
        The regular expression pattern for the marker stream. Use "task1$|task2$|task3$" if task1, task2, and task3 are the markers

    channels : tuple of ints
        Index value of channels to poll from the stream, if None all channels will be polled.

    """

    def __init__(self,sess,pred=None,channels=None, relative_start = 0,
                 marker=True,marker_fmt=None,marker_pred=None, stream_info=None, marker_stream_info=None, active=True):
        """
        Create a new LSL inlet stream object
        Parameters
        ----------
        sess : session object
            Session object where the data source will exist
        pred : str
            The predicate string, e.g. "name='BioSemi'" or "type='EEG' and starts-with(name,'BioSemi') and 
            count(description/desc/channels/channel)=32"
        channels : tuple of ints
            Index value of channels to poll from the stream, if None all channels will be polled
        relative_start : float, default = 0
            Duration of tiem before marker from which samples should be extracted during polling.

        marker : bool
            true if there is an associated marker to indicate relative time where data should begin to be polled
        marker_fmt : Regex or list
            Regular expression template of the marker to be matched, if none all markers will be matched. Alternatively, a list of markers can be provided.
        marker_pred : str
            The predicate string for the marker stream
        stream_info : pylsl.StreamInfo
            The stream info object for the stream can be passed instead of the predicate to avoid the need to resolve the stream
        marker_stream_info : pylsl.StreamInfo
            The stream info object for the marker stream can be passed instead of the predicate to avoid the need to resolve the stream
        """
        super().__init__(BcipEnums.SRC,sess)
        
        if active:
            if not stream_info:
                # resolve the stream on the LSL network
                available_streams = pylsl.resolve_bypred(pred)
            else:
                available_streams = [stream_info]
            
            if len(available_streams) == 0:
                # TODO log error
                return
            
            # TODO - Warn about more than one available stream
            self.data_buffer = {'EEG':None,'time_stamps':None}
            self.data_inlet = pylsl.StreamInlet(available_streams[0], processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter, recover=False) # for now, just take the first available stream that matches the property
            self.data_inlet.open_stream()

            self.marker_inlet = None
            self.marker_pattern = None
            self.relative_start = relative_start
            self._already_peeked = False
            self._peeked_marker = None

            self.marker_timestamps = []
            self.first_data_timestamp = None
            self.time_correction = None
            
            # TODO - check if the stream has enough input channels to match the
            # channels parameter
            if channels:
                self.channels = channels
            else:
                self.channels = tuple([_ for _ in range(self.data_inlet.channel_count)])

            if marker:
                if not marker_stream_info:
                    # resolve the stream on the LSL network
                    marker_streams = pylsl.resolve_bypred(marker_pred)
                else:
                    marker_streams = [marker_stream_info]

                self.marker_inlet = pylsl.StreamInlet(marker_streams[0]) # for now, just take the first available marker stream
                self.peek_marker_inlet = pylsl.StreamInlet(marker_streams[0]) 
                
                # open the inlet
                self.marker_inlet.open_stream()
                self.peek_marker_inlet.open_stream()
            
            
                if marker_fmt:
                #    if isinstance(marker_fmt,list):
                #        marker_fmt = '$|^'.join(marker_fmt)
                #       marker_fmt = '^' + marker_fmt + '$' 
                    
                    self.marker_pattern = re.compile(marker_fmt)
            
            

        
    def poll_data(self, Ns, label=None):
        """
        Pull data from the inlet stream until we have Ns data points for each
        channel.
        Parameters
        ----------
        Ns: int
            number of samples to collect
        Label : None
            used for file-based polling, not used here
        """
        
        if self.marker_inlet != None:
            # start by getting the timestamp for this trial's marker
            t_begin = None
            while t_begin == None:
                marker, t = self.marker_inlet.pull_sample()
                print("marker: ", marker)
                if marker != None:
                    marker = marker[0] # extract the string portion of the marker
                    #print(marker)
                    #print(self.marker_pattern)
                    if (self.marker_pattern == None) or self.marker_pattern.match(marker):
                        t_begin = t
                        #print(t_begin)
                        self.marker_timestamps.append(t_begin)
                    
        else:
            t_begin = 0 # i.e. all data is valid
        
        t_begin += self.relative_start
        print(t_begin)
        # pull the data in chunks until we get the total number of samples
        trial_data = np.zeros((len(self.channels), Ns)) # allocate the array
        trial_timestamps = np.zeros((Ns,))
        samples_polled = 0        

        # First, pull the data required data from the buffer
        if self.data_buffer['EEG'] is not None:
            eeg_index_bool = np.array(self.data_buffer['time_stamps'] >= t_begin)

            samples_polled = np.sum(eeg_index_bool)
            #print(eeg_index_bool)
            #print(self.data_buffer['time_stamps'])
            try:
                trial_data[:,:samples_polled] = self.data_buffer['EEG'][:,:][:,eeg_index_bool]
            except ValueError:
                data = self.data_buffer['EEG'][:,:][:,eeg_index_bool]
                trial_data[:,:samples_polled] = data[:,:][:,:Ns]
            #print(trial_timestamps)
            try:
                trial_timestamps[:samples_polled] = self.data_buffer['time_stamps'][eeg_index_bool]
            except ValueError:
                stamps = self.data_buffer['time_stamps'][eeg_index_bool]
                trial_timestamps[:samples_polled] = stamps[:Ns]

        #print(f"T-BEGIN:`{t_begin}`")
        while samples_polled < Ns:
            data, timestamps = self.data_inlet.pull_chunk(timeout=0.0)
            timestamps = np.asarray(timestamps)

            if len(timestamps) > 0:
                self.time_correction = self.data_inlet.time_correction()
                timestamps += self.time_correction

                if np.any(timestamps >= t_begin):
                    
                    #print(f"Number of new trials added to trial data: {np.sum(timestamps >= t_begin)}")
                    # convert data to numpy arrays
                    data = np.asarray(data).T
                    timestamps_index_bool = timestamps >= t_begin

                    data = data[self.channels,:][:,timestamps_index_bool]
                    timestamps = timestamps[timestamps_index_bool]
                    
                    if len(data.shape) > 1:
                        chunk_sz = data.shape[1]
                        # append the latest chunk to the trial_data array
                        if samples_polled + chunk_sz > Ns:
                            dest_end_index = Ns
                            src_end_index = Ns - samples_polled
                        
                        else:
                            dest_end_index = samples_polled + chunk_sz
                            src_end_index = chunk_sz
                        
                        trial_data[:, samples_polled:dest_end_index] = data[:,:src_end_index]
                        trial_timestamps[samples_polled:dest_end_index] = timestamps[:src_end_index]

                        if dest_end_index == Ns:
                            self.data_buffer['EEG'] = np.concatenate((trial_data, data[:, src_end_index:]), axis=1)                                
                            self.data_buffer['time_stamps'] = np.concatenate((trial_timestamps, timestamps[src_end_index:]))
                        

                        samples_polled += chunk_sz 
                        
        self.first_data_timestamp = trial_timestamps[0]
        trial_data = trial_data[:, :Ns] # TODO remove?

        return trial_data
    
    def peek_marker(self):
        """
        Peek at the next marker in the marker stream
        """
        if self._already_peeked:
            return self._peeked_marker
        
        else:
            marker, t = self.peek_marker_inlet.pull_sample()
            while self.marker_pattern != None and not self.marker_pattern.match(marker[0]):
                marker, t = self.peek_marker_inlet.pull_sample()

            if marker != None:
                self._peeked_marker = marker[0]
                self._already_peeked = True
                return marker[0]
               
        return None
        

    #TODO: It is technically safe to recall self.__init__ here instead of doing this by hand (since the object reference won't be changed),
    # this method was used to avoid confusion and to make the code more readable
    def update_input_streams(self,pred=None,channels=None, relative_start = 0,
                 marker=True,marker_fmt=None,marker_pred=None, stream_info=None, marker_stream_info=None):

        """
        Update the input stream with new parameters

        Parameters
        ----------
        pred : str
            The predicate string, e.g. "name='BioSemi'" or "type='EEG' and starts-with(name,'BioSemi') and
            count(description/desc/channels/channel)=32"
        channels : tuple of ints
            Index value of channels to poll from the stream, if None all channels will be polled
        relative_start : float, default = 0 
            Duration of tiem before marker from which samples should be extracted during polling.
        marker : bool
            true if there is an associated marker to indicate relative time where data should begin to be polled
        marker_fmt : Regex or list
            Regular expression template of the marker to be matched, if none all markers will be matched. Alternatively, a list of markers can be provided.
        marker_pred : str
            The predicate string for the marker stream
        stream_info : pylsl.StreamInfo
            The stream info object for the stream can be passed instead of the predicate to avoid the need to resolve the stream
        marker_stream_info : pylsl.StreamInfo 
            The stream info object for the marker stream can be passed instead of the predicate to avoid the need to resolve the stream

        """
        
        if not stream_info:
            # resolve the stream on the LSL network
            available_streams = pylsl.resolve_bypred(pred)
        else:
            available_streams = [stream_info]
        
        if len(available_streams) == 0:
            # TODO log error
            return
        
        # TODO - Warn about more than one available stream
        self.data_buffer = {'EEG':None,'time_stamps':None}
        self.data_inlet = pylsl.StreamInlet(available_streams[0]) # for now, just take the first available stream that matches the property
        self.data_inlet.open_stream()

        self.marker_inlet = None
        self.marker_pattern = None
        self.relative_start = relative_start
        self._already_peeked = False
        self._peeked_marker = None

        self.marker_timestamps = []
        
        # TODO - check if the stream has enough input channels to match the
        # channels parameter
        if channels:
            self.channels = channels
        else:
            self.channels = tuple([_ for _ in range(self.data_inlet.channel_count)])

        if marker:
            if not marker_stream_info:
                # resolve the stream on the LSL network
                marker_streams = pylsl.resolve_bypred(marker_pred)
            else:
                marker_streams = [marker_stream_info]

            self.marker_inlet = pylsl.StreamInlet(marker_streams[0]) # for now, just take the first available marker stream
            self.peek_marker_inlet = pylsl.StreamInlet(marker_streams[0]) 
            
            # open the inlet
            self.marker_inlet.open_stream()
            self.peek_marker_inlet.open_stream()
        
            if marker_fmt:
            #    if isinstance(marker_fmt,list):
            #        marker_fmt = '$|^'.join(marker_fmt)
            #       marker_fmt = '^' + marker_fmt + '$' 
                
                self.marker_pattern = re.compile(marker_fmt)

        

    @classmethod
    def create_marker_coupled_data_stream(cls,sess,pred=None, channels = None, relative_start=0,
                                          marker_fmt=None, marker_pred="type='Markers'", stream_info=None, marker_stream_info = None, active=True):
        """
        Create a LSLStream data object that maintains a data stream and a
        marker stream

        Parameters
        -----------

        sess : session object
            Session object where the data source will exist
        pred : str
            The predicate string, e.g. "name='BioSemi'" or "type='EEG' and starts-with(name,'BioSemi') and
            count(description/desc/channels/channel)=32"
        channels : tuple or list of ints
            Index value of channels to poll from the stream, if None all channels will be polled
        marker_fmt : str
            Regular expression template of the marker to be matched, if none all markers will be matched
        marker_pred : str
            Predicate string to match the marker stream, if None all streams will be matched
        stream_info : StreamInfo object
            StreamInfo object to use for the data stream, if None a default StreamInfo object will be created
        
        Examples
        --------
        """
        src = cls(sess,pred,channels,relative_start,True,marker_fmt,marker_pred, stream_info, marker_stream_info, active)
        sess.add_ext_src(src)
        
        return src
    
    @classmethod
    def create_marker_uncoupled_data_stream(cls,sess,pred,channels = None, relative_start = 0,
                                            marker_fmt="T{},L{},LN{}"):
        """
        Create a LSLStream data object that maintains only a data stream with
        no associated marker stream
        Parameters
        ----------
        Examples
        --------
        sess : session object
            Session object where the data source will exist
        pred : str
            The predicate string, e.g. "name='BioSemi'" or "type='EEG' and starts-with(name,'BioSemi') and
            count(description/desc/channels/channel)=32"
        channels : tuple or list of ints
            Index value of channels to poll from the stream, if None all channels will be polled
        marker_fmt : str
            Regular expression template of the marker to be matched, if none all markers will be matched
        """
        src = cls(sess,pred,channels, relative_start, False)
        sess.add_ext_src(src)
        
        return src

class OutputLSLStream(BCIP):
    """
    An object for maintaining an LSL outlet

    Attributes
    ----------
    
    """

    def __init__(self, sess, stream_info, filesave=None, chunk_size = 0, max_buffer=360):
        """Establish a new stream outlet. This makes the stream discoverable.
        
        Parameters
        ----------

        stream_info : StreamInfo
            StreamInfo object to describe this stream. Stays constant over the lifetime of the outlet.
        
        chunk_size : int, default = 0
            Optionally the desired chunk granularity (in samples) for transmission. 
            If unspecified, each push operation yields one chunk. Inlets can override this setting. (default 0)
        
        max_buffered : default = 360
            The maximum amount of data to buffer (in seconds if there is a nominal sampling rate, otherwise 
            x100 in samples). The default is 6 minutes of data. Note that, for high-bandwidth data, you will want to 
            use a lower value here to avoid running out of RAM.
                        

        """
        super().__init__(BcipEnums.SRC,sess)
        self._sess = sess
        self._stream_info = stream_info
        # resolve the stream on the LSL network
        self.lsl_marker_outlet = pylsl.StreamOutlet(stream_info,chunk_size,max_buffer)
        self.liesl_session = None
        # Start LieSL recording if the user has specified a filesave
        
        warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='subprocess')
        x = threading.Thread(target=self._check_status, args=(filesave,))
        x.start()


    def _check_status(self, filesave):
        if filesave is not None:

            streamargs = [{'name': self._stream_info.name(), "type": self._stream_info.type(),
                          "channel_count": self._stream_info.channel_count(), "nominal_srate": self._stream_info.nominal_srate(),
                          "channel_format": self._stream_info.channel_format(),
                          "source_id": self._stream_info.source_id()}]
            self.liesl_session = liesl.Session(mainfolder=f"{os.path.dirname(os.path.realpath(__file__))}\labrecordings", streamargs=streamargs)
        
        with self.liesl_session(filesave):
            while True:
                time.sleep(.1)
                if not threading.main_thread().is_alive():
                    sys.stdout = open(os.devnull, "w")
                    sys.stderr = open(os.devnull, "w")
                    self.liesl_session.stop_recording()
                    sys.stdout = sys.__stdout__
                    sys.stderr = sys.__stderr__
                    return    

    def push_data(self, data, label = None):
        """
        Push data to the outlet stream.
        
        Parameters
        ----------
        
        """

        try:
            self.lsl_marker_outlet.push_sample(data)
            return BcipEnums.SUCCESS

        except (ValueError, TypeError) as ve:
            try:
                self.lsl_marker_outlet.push_sample(data[0], pylsl.local_clock())
                return BcipEnums.SUCCESS
            
            except Exception as e:
                print(e)
                print("Irreparable Error in LSL Output. No data pushed to output stream\n") 

    @classmethod
    def create_outlet_from_streaminfo(cls,sess, stream_info, filesave = None):
        """
        Factory method to create a OutletLSLStream bcipy object from a pylsl.StreamInfo object.

        Parameters
        -----------

        sess : session object
            Session object where the data source will exist
        stream_info : pylsl.StreamInfo object
            pylsl.StreamInfo object that describes the stream to be created
        
        Examples
        --------
        """
        src = cls(sess, stream_info, filesave)
        sess.add_ext_out(src)
        
        return src
    
    @classmethod
    def create_outlet(cls,sess,name='untitled', type='', channel_count = 1, nominal_srate=0.0,
                                          channel_format=1, source_id="", filesave = None):    
        """
        Factory Method to create an OutletLSLStream bcipy object from scratch.

        Parameters
        ----------

        sess : session object
            Session object where the data source will exist

        name : str, default = 'untitled'
            * Name of the stream. 
            * Describes the device (or product series) that this stream makes available.

        type  str, default = ''
            * Content type of the stream. 
            * By convention LSL uses the content types defined in the XDF file format specification where applicable. 
        
        channel_count : int, default = 1
            * Number of channels per sample. This stays constant for the lifetime of the stream. 
        
        nominal_srate : float, default = 0.0
            * The sampling rate (in Hz) as advertised by the data source.

        channel_format : int or str, default = 1
            * Format/type of each channel (ie. 'float32'). 

        source_id : str, default = ''
            * Unique identifier of the device or source of the data, if available (such as the serial number). 
            * This is critical for system robustness since it allows recipients to recover from failure even after the serving app, device or computer crashes (just by finding a stream with the same source id on the network again). 
        
        filesave : str, default = None
            If not None, the data will be saved to the given file.
        """

        stream_info = pylsl.StreamInfo(name, type, channel_count, nominal_srate, channel_format, source_id='1007988689')
        src = cls(sess, stream_info, filesave)
        sess.add_ext_out(src)
        
        return src
