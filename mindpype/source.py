"""
Currently supported sources:
    - Lab Streaming Layer
    - xdf files


"""

from .core import MPBase, MPEnums
from .containers import Tensor
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
import json
import time

class InputXDFFile(MPBase):
    """
    Utility class for extracting trial data from an XDF file for MindPype.

    Parameters
    ----------

    sess : Session Object
        Session where the MPXDF data source will exist.

    files : list of str
        XDF file(s) where data should be extracted from.

    tasks : list or tuple of strings
        List or Tuple of strings corresponding to the tasks to be completed by the user.
        For example, the tasks 'target' and 'non-target'/'flash' can be used for P300-type setups.

    channels : list or tuple of int
        Values corresponding to the stream channels used during the session

    relative_start : float, default = 0
        Value corresponding to the start of the trial relative to the marker onset.

    Ns : int, default = 1
        Number of samples to be extracted per trial. For epoched data, this value determines the
        size of each epoch, whereas this value is used in polling for continuous data.

    mode : 'continuous', 'class-separated' or 'epoched', default = 'epoched'
        Mode indicates whether the inputted data will be epoched sequentially as individual trials,
        epoched by class, or to leave the data in a continuous format

    .. warning::
       The task list used in the InputXDFFile object MUST REFLECT the task list used in the XDF file.
       Differences will cause the program to fail.

    .. note::
        There are 3 types of modes for the MPXDF object: 'continuous', 'class-separated' and 'epoched'.
        Continuous mode will leave the data in a continuous format, and will poll the data for the next Ns samples
        each time the poll_data method is called. Class-separated mode will epoch the data by class, and will poll the
        data for the next Ns samples of the specified class each time the poll_data method is called. Epoched mode will
        epoch the data sequentially, and will poll the data for the next Ns samples of the next trial
        (Ns < length of the epoch) each time the poll_data method is called.

        For P300/MI paradigms, where there are specified task names (i.e. 'target' and 'non-target'/'flash', etc.),
        class-separated mode is recommended. For other paradigms, where there are no specified task names, and data will
        be polled sequentially, either continuous or epoched mode is recommended.

        Class-separated mode will store the data in a dictionary with the following format:

        .. code-block:: python

            self.trial_data = {
                "Data":
                    {"time_series":
                        {task_name1: np.array([Nt x Nc x Ns]),
                         task_name2: np.array([Nt x Nc x Ns]),},
                     "time_stamps": np.array([Ns])}},
                "Markers": {"time_series": np.array([Ns]),
                            "time_stamps": np.array([Ns])},
            }

        Continuous mode will store the data in a dictionary with the following format:

        .. code-block:: python

            self.trial_data = {
                "Data":
                    {"time_series": np.array([Nc x Ns]),
                     "time_stamps": np.array([Ns])},
                "Markers":
                    {"time_series": np.array([Ns]),
                     "time_stamps": np.array([Ns])},

        Epoched mode will store the data in a dictionary with the following format:

        .. code-block:: python

            self.trial_data = {
                "Data":
                    {"time_series": np.array([Nt x Nc x Ns]),
                     "time_stamps": np.array([Ns])},
                "Markers":
                    {"time_series": np.array([Ns]),
                     "time_stamps": np.array([Ns])},
            }
    
    Attributes
    ----------
    files : list of str
        XDF file(s) where data should be extracted from.
        
    relative_start : float, default = 0
        Value corresponding to the start of the trial relative to the marker onset.
        
    Ns : int, default = 1
        Number of samples to be extracted per trial. For epoched data, this value determines the
        size of each epoch, whereas this value is used in polling for continuous data.

    tasks : list or tuple of strings
        List or Tuple of strings corresponding to the tasks to be completed by the user.
        For example, the tasks 'target' and 'non-target'/'flash' can be used for P300-type setups.

    channels : list or tuple of int
        Values corresponding to the stream channels used during the session

    mode : 'continuous', 'class-separated' or 'epoched', default = 'epoched'
        Mode indicates whether the inputted data will be epoched sequentially as individual trials,
        epoched by class, or to leave the data in a continuous format
        
    stream_type: str
        Type of stream (Data or Markers)

    """

    def __init__(self, sess, files, channels, tasks=None, relative_start=0, Ns=1, stype='EEG', mode="epoched", marker_stream_name=None):
        """
        Create a new xdf file reader interface
        """
        super().__init__(MPEnums.SRC, sess)

        if type(files) == str:
            files = [files]

        self.files = files
        self.relative_start = relative_start
        self.Ns = int(Ns)
        self.tasks = tasks
        self._inferred_tasks = tasks is None
        self.channels = channels
        self._label_counter = None
        self.mode = mode
        self.stream_type = stype

        # Epoched mode will store trial data in a 3D array, with the first dimension corresponding to the trial number
        # and the second and third dimensions corresponding to the channel and sample number, respectively
        # The markers will be stored in a 1D tuple, with the first dimension corresponding to the sample number.
        if mode == "epoched":
            self._data = {"Data": [], "Markers": []}

            for filename in files:
                # open file and extract data
                data, header = pyxdf.load_xdf(filename)

                # extract the marker and data streams
                marker_stream = None
                data_stream = None
                for stream in data:
                    if (stream["info"]["type"][0] == "Marker" or
                        stream["info"]["type"][0] == "Markers"):
                        if (marker_stream is None or 
                            marker_stream_name is None or 
                            stream["info"]["name"][0] == marker_stream_name
                        ):
                            marker_stream = stream

                    elif stream["info"]["type"][0] == self.stream_type:
                        data_stream = stream

                if marker_stream is None or data_stream is None:
                    raise ValueError(f"The XDF file {filename} does not contain the required streams")

                sample_indices = np.zeros(data_stream["time_stamps"].shape)  # used to extract data samples, pre-allocated here

                # filter the marker stream for the specified tasks
                marker_stream = self._filter_marker_stream(marker_stream)

                # iterate throught the markers and extract the data for each task
                for i_m, marker in enumerate(marker_stream["time_series"]):
                    # compute the correct start and end indices for the current trial
                    marker_time = marker_stream["time_stamps"][i_m]
                    data_window_start = marker_time + relative_start

                    # find the index of the first sample after the marker
                    sample_indices = np.array(data_stream["time_stamps"] >= data_window_start)
                    first_sample_ix = np.argwhere(sample_indices)[0][0]
                    sample_indices[first_sample_ix + self.Ns:] = False  # remove the samples after the end of the trial

                    # extract the data and append to the data dictionary
                    sample_data = data_stream["time_series"][np.ix_(sample_indices, channels)].T  # Nc X Ns
                    self._data["Data"].append(sample_data)
                    self._data["Markers"].append(marker)

            # convert the data to a numpy array and the markers to a tuple
            self._data["Data"] = np.stack(self._data["Data"], axis=0) # Nt x Nc x Ns
            self._data["Markers"] = tuple(self._data["Markers"])

            # create a corresponding numerical task label for each task
            self._data["numerical_labels"] = np.array([self.tasks.index(task) for task in self._data["Markers"]])


        # Continuous mode will leave the data in a continuous format, and will poll the data for the next Ns samples
        elif mode == "continuous":
            self._data = {"Data": {"time_series": None, "time_stamps": None},
                          "Markers": {"time_series": None, "time_stamps": None}}

            # First order the files by the first marker timestamp
            file_first_marker = np.zeros((len(files),))
            for i_f, filename in enumerate(files):
                data, _ = pyxdf.load_xdf(filename)

                # extract the first marker timestamp from the file
                for stream in data:
                    if (stream["info"]["type"][0] == "Marker" or
                        stream["info"]["type"][0] == "Markers"):
                        file_first_marker[i_f] = stream["time_series"][0][0]

            # Sort the files by the first marker value
            file_order = np.argsort(file_first_marker)
            files = [files[i] for i in file_order]

            data_streams = []
            marker_streams = []

            # Iterate through all files and extract the data
            for filename in files:
                data, _ = pyxdf.load_xdf(filename)

                # Iterate through all streams in every file, add current file's data to the previously loaded data
                for stream in data:
                    if (stream["info"]["type"][0] == "Marker" or
                        stream["info"]["type"][0] == "Markers"):
                        marker_stream = stream

                    # If the data stream already exists, concatenate the new data to the existing data
                    elif stream["info"]["type"][0] == self.stream_type:
                        data_stream = stream

                if marker_stream is None or data_stream is None:
                    raise ValueError(f"The XDF file {filename} does not contain the required streams")

                # Extract the data from the data stream
                data_stream["time_series"] = data_stream["time_series"][:, channels].T

                # Filter the marker stream for the specified tasks
                marker_stream = self._filter_marker_stream(marker_stream)

                # Append the data and marker streams to the list
                data_streams.append(data_stream)
                marker_streams.append(marker_stream)

            # Concatenate the data and marker streams
            self._data["Data"]["time_series"] = np.concatenate([stream["time_series"] for stream in data_streams], axis=1)
            self._data["Data"]["time_stamps"] = np.concatenate([stream["time_stamps"] for stream in data_streams])

            self._data["Markers"]["time_series"] = marker_streams[0]["time_series"]
            if len(marker_streams) > 1:
                for stream in marker_streams[1:]:
                    self._data["Markers"]["time_series"].extend(stream["time_series"])

            self._data["Markers"]["time_stamps"] = np.concatenate([stream["time_stamps"] for stream in marker_streams])
            self._data["numerical_labels"] = np.array([self._tasks.index(task) for task in self._data["Markers"]["time_series"]])

        # create a counter to keep track of the number of trials extracted when polling
        self._task_counter = {task: 0 for task in self.tasks}

    def _filter_marker_stream(self, marker_stream):
        """
        Filter the marker stream for the specified tasks.
        If no task list is provided, try to infer the tasks
        from the marker stream (currently only supported for Mindset P300 data).

        Parameters
        ----------

        marker_stream: dictionary
            Time series and time stamps for data 

        Returns
        -------
        marker_stream: dictionary
            Time series and time stamps for inferred tasks
        """
        if not self._inferred_tasks and self.tasks:
            # filter for markers that are tasks
            task_marker_mask = np.array([marker[0] in self.tasks for marker in marker_stream["time_series"]])
            marker_stream["time_series"] = [marker[0] for marker, mask in zip(marker_stream["time_series"], task_marker_mask) if mask]
            marker_stream["time_stamps"] = marker_stream["time_stamps"][task_marker_mask]
        else:
            # infer tasks from Marker stream - only works for Mindset P300 data
            warnings.warn("No task list provided. Infering tasks from the marker stream. This is only supported for Mindset P300 data.", RuntimeWarning, stacklevel=2)
            marker_stream["time_series"] = [marker[0] for marker in marker_stream["time_series"]]
            self.tasks = ['non-target', 'target']  # default tasks for Mindset P300 data

            inferred_markers = []
            inferred_marker_times = []
            current_target = None
            for i_m, marker in enumerate(marker_stream["time_series"]):
                if "target" in marker:
                    # if the marker identifies a target, store the target grid
                    current_target = json.loads(marker)["target"]
                elif current_target is not None and "flash" in marker:
                    # if the marker is a flash, check if it is a target or non-target
                    flash_positions = json.loads(marker)["flash"]
                    if current_target in flash_positions:
                        inferred_markers.append("target")
                    else:
                        inferred_markers.append("non-target")

                    # record the time of the marker
                    inferred_marker_times.append(marker_stream["time_stamps"][i_m])

            # overwrite the original marker stream with the inferred markers
            marker_stream["time_series"] = inferred_markers
            marker_stream["time_stamps"] = np.array(inferred_marker_times)

        return marker_stream

    def poll_data(self, label=None):

        """
        Polls the data source for new data.

        Parameters
        ----------

        label : string
            Marker of next trial to be polled. If None, the next trial according
            to timestamps will be polled.

        Returns
        -------
        sample_data: dictionary
        """

        if label is not None and label not in self.tasks:
            # check if the coorresponding numerical label has been provided
            if label in self._data["numerical_labels"]:
                label = self._data['Markers'][np.argwhere(self._data["numerical_labels"]==label)[0][0]]
            else:
                raise ValueError(f"Label {label} is not in the list of tasks")


        # determine the index of the next trial to be polled
        if self.mode == "epoched":
            markers = self._data["Markers"]
        else:
            markers = self._data["Markers"]["time_series"]

        if label is None:
            num_prev_polled = sum(self._task_counter.values())
            poll_index = num_prev_polled  # default, assumes that the trials have been polled in order
            for task in self.tasks:
                # find the first trial for the specified task that has not been polled
                task_min = markers.index(task, self._task_counter[task])
                if task_min < poll_index:
                    poll_index = task_min

            label = markers[poll_index]
        else:
            label_indices = [i for i, m in enumerate(markers) if m == label]
            poll_index = label_indices[self._task_counter[label]]

        if self.mode == "epoched":
            # Extract sample data from epoched trial data
            sample_data = self._data["Data"][poll_index, :, :]

        else:
            # Extract the nth marker timestamp, corresponding to the nth trial in the XDF file
            data_window_start = (
                self._data["Markers"]["time_stamps"][poll_index] + self.relative_start
            )

            # Construct the boolean array for samples that fall after the marker timestamp
            sample_indices = self._data["Data"]["time_stamps"] >= data_window_start
            first_sample_ix = np.argwhere(sample_indices)[0][0]
            sample_indices[first_sample_ix + self.Ns:] = False  # remove the samples after the end of the trial
            sample_data = self._data["Data"]["time_series"][:, sample_indices]

        # increment the task counter
        self._task_counter[label] += 1

        return sample_data

    def load_into_tensors(self, include_timestamps=False):
        """
        Loads entirity of InputXDFFile data object into a tensor.
        Returns 2-4 MindPype Tensor objects, in the following order.

            1. Tensor containing the Stream data
            2. Tensor containing the Marker data
            3. Tensor containing the Stream timestamps (if continuous data and include_timestamps is True)
            4. Tensor containing the Marker timestamps (if continuous data and include_timestamps is True)

        Parameters
        ----------
        include_timestamps : bool, default = False
            If True, the function will return the Marker timestamps as well as the data.
            Only applicable for continuous data.

        Returns
        -------
        data : Tensor
            Tensor containing the stream data

        labels : Tensor
            Tensor containing the numerical encoded markers

        data_ts : Tensor
            Tensor containing the stream timestamps

        labels_ts : Tensor
            Tensor containing the Marker timestamps
        """

        if self.mode == "epoched":
            data = Tensor.create_from_data(self.session, self._data["Data"])
            labels = Tensor.create_from_data(self.session, self._data["numerical_labels"])

            return data, labels

        elif self.mode == "continuous":
            data = Tensor.create_from_data(self.session, self._data["Data"]["time_series"])
            labels = Tensor.create_from_data(self.session, self._data["numerical_labels"])

            if include_timestamps:
                data_ts = Tensor.create_from_data(self.session, self._data["Data"]["time_stamps"])
                labels_ts = Tensor.create_from_data(self.session, self._data["Markers"]["time_stamps"])

                return data, labels, data_ts, labels_ts
            else:
                return data, labels


    @classmethod
    def create_continuous(cls, sess, files, channels, tasks=None, relative_start=0, Ns=1):
        """
        Factory Method for creating continuous XDF File input source.


        Parameters
        ---------
        sess : Session Object
            Session where the MPXDF data source will exist.

        files : list of str
            XDF file(s) where data should be extracted from.

        tasks : list or tuple of strings (default = None)
            List or Tuple of strings corresponding to the tasks to be completed by the user.
            For P300-type setups, the tasks 'target' and 'non-target'/'flash' can be used.
            If None, the tasks will be inferred from the marker stream. This is only
            supported for P300 data recorded using Mindset.

        channels : list or tuple of int
            Values corresponding to the data stream channels used during the session

        relative_start : float, default = 0
            Value corresponding to the start of the trial relative to the marker onset.

        Ns : int, default = 1
            Number of samples to be extracted per trial. For epoched data, this value determines the
            size of each epoch, whereas this value is used in polling for continuous data.

        
        Returns
        -------
        src: InputXDFFile
            Continous XDF file input source
        """

        src = cls(sess, files, channels, tasks, relative_start, Ns, mode="continuous")
        sess.add_to_session(src)

        return src


    @classmethod
    def create_epoched(
        cls, 
        sess, 
        files, 
        channels, 
        tasks=None, 
        relative_start=0, 
        Ns=1, 
        stype='EEG',
        marker_stream_name=None
    ):

        """
        Factory Method for creating epoched XDF File input source.

        Parameters
        ---------
        sess : Session Object
            Session where the MPXDF data source will exist.

        files : list of str
            XDF file(s) where data should be extracted from.

        tasks : list or tuple of strings (default = None)
            List or Tuple of strings corresponding to the tasks to be completed by the user.
            For P300-type setups, the tasks 'target' and 'non-target'/'flash' can be used.
            If None, the tasks will be inferred from the marker stream. This is only
            supported for P300 data recorded using Mindset.

        channels : list or tuple of int
            Values corresponding to the data stream channels used during the session

        relative_start : float, default = 0
            Value corresponding to the start of the trial relative to the marker onset.

        stype : str, default = EEG
            String indicating the data type

        Ns : int, default = 1
            Number of samples to be extracted per trial. For class-separated data, this value determines the
            size of each epoch, whereas this value is used in polling for continuous data.

        marker_stream : str, default = None
            Name of the marker stream to be used. If none, the first marker stream found in the XDF file will be used.

        Returns
        -------
        src: InputXDFFile
            Epoched XDF file input source

        """
        src = cls(
            sess, 
            files, 
            channels, 
            tasks, 
            relative_start, 
            Ns, 
            stype=stype, 
            mode="epoched", 
            marker_stream_name=marker_stream_name
        )
        sess.add_to_session(src)

        return src
    

class InputLSLStream(MPBase):
    """
    An object for maintaining an LSL inlet

    Attributes
    ----------
    data_buffer : dict
        {'Data': np.array, 'time_stamps': np.array}
        A dictionary containing the data and time stamps from past samples (used when trials have overlapping data)

    data_inlet : pylsl.StreamInlet
        The LSL inlet object

    marker_inlet : pylsl.StreamInlet
        The LSL inlet object for the marker stream

    marker_pattern : re.Pattern
        The regular expression pattern for the marker stream. Use "task1$|task2$|task3$" if task1, task2, and task3 are the markers

    channels : tuple of ints
        Index value of channels to poll from the stream, if None all channels will be polled.

    TODO: update attributes docstring 
    """

    MAX_NULL_READS = 1000

    def __init__(
        self,
        sess,
        pred=None,
        channels=None,
        relative_start=0,
        marker_coupled=True,
        marker_fmt=None,
        marker_pred=None,
        stream_info=None,
        marker_stream_info=None,
        active=True,
        interval=None,
        Ns=1,
        mode='single',
        n_epochs=1
    ):
        """
        Create a new LSL inlet stream object

        Parameters
        ----------

        sess : session object
            Session object where the data source will exist

        pred : str
            The predicate string, e.g. "name='BioSemi'" or "type='EEG' and starts-with(name, 'BioSemi') and
            count(description/desc/channels/channel)=32"

        channels : tuple of ints
            Index value of channels to poll from the stream, if None all channels will be polled

        relative_start : float, default = 0
            Duration of tiem before marker from which samples should be extracted during polling.

        marker_coupled : bool
            true if there is an associated marker to indicate relative time where data should begin to be polled

        marker_fmt : Regex or list
            Regular expression template of the marker to be matched, if none all markers will be matched. Alternatively, a list of markers can be provided.

        marker_pred : str
            The predicate string for the marker stream

        stream_info : pylsl.StreamInfo
            The stream info object for the stream can be passed instead of the predicate to avoid the need to resolve the stream

        marker_stream_info : pylsl.StreamInfo
            The stream info object for the marker stream can be passed instead of the predicate to avoid the need to resolve the stream

        active : bool
            True if the stream should be opened immediately, false if the stream should be opened later

        interval : float
            The minimum interval between polling the stream for new data. Only used for marker uncoupled streams.
            If None, then the stream will be polled as fast as possible.

        Ns : int, default = 1
            The number of samples to be extracted per poll.

        mode : str, default = 'single'
            Mode of the stream. Can be 'single', 'continuous', or 'epoched'. 
            If 'single', the stream will be polled for a single trial at a time.
            If 'continuous', the first trial will be couple to a marker, and subsequent trials 
            will be polled based on the interval. If 'epoched', the first trial will be coupled 
            to a marker, and then a fixed number of subsequent trials, defined by the n_epochs 
            parameter, will be polled based on the interval. The interval parameter must be 
            provided for both the epoched and continuous modes.

        n_epochs : int, default = 1
            Number of epochs to poll for in the epoched mode. Only used if mode is 'epoched'.

        .. note::
            The active parameter is used when the session is created before the LSL stream is started, or the stream is
            not available when the session is created. In that case, the stream can be updated later by calling the update_input_stream() method.
        """
        super().__init__(MPEnums.SRC, sess)
        self.marker_coupled = marker_coupled

        self._marker_inlet = None
        self.marker_pattern = None
        self.relative_start = relative_start
        self._already_peeked = False
        self._peeked_marker = None
        self._marker_buffer = {"time_series": None, "time_stamps": None} # only keeps most recent value, can expand in future if needed
        self._time_correction = None
        self._interval = interval
        self.channels = channels
        self.Ns = Ns
        self.mode = mode
        self.n_epochs = n_epochs
        self._epochs_polled = 0
        self._data_buffer = {"time_series": None, "time_stamps": None}

        if active:
            self._active = False # will be set to True when the stream is opened
            self.update_input_streams(pred, channels, marker_coupled, marker_fmt, marker_pred, stream_info, marker_stream_info, Ns)

    def poll_data(self, label=None):
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

        if not self._active:
            raise RuntimeWarning("InputLSLStream.poll_data() called on inactive stream. Please call update_input_streams() first to configure the stream object.")

        poll_marker = False
        if self.marker_coupled:
            if self.mode == 'single':
                poll_marker = True
            elif self.mode == 'epoched' and self._epochs_polled in (0, self.n_epochs):
                poll_marker = True
                # reset the epochs polled counter
                self._epochs_polled = 0

        if poll_marker:
            # start by getting the timestamp for this trial's marker
            t_begin = None
            null_reads = 0
            while t_begin is None:
                marker, t = self._marker_inlet.pull_sample(timeout=0.0)

                if marker is not None:
                    null_reads = 0  # reset the null reads counter
                    marker = marker[0]  # extract the string portion of the marker

                    if (self.marker_pattern is None) or self.marker_pattern.match(marker):
                        t_begin = t
                        self._marker_buffer["time_stamps"] = t_begin
                        self._marker_buffer["time_series"] = marker
                else:
                    null_reads += 1
                    if null_reads > self.MAX_NULL_READS:
                        raise RuntimeError(
                            f"The marker stream has not been updated in the last {self.MAX_NULL_READS} read attemps. Please check the stream."
                        )
                    time.sleep(0.1)

        else:
            # marker-uncoupled stream, determine the start time based on the interval attribute
            if self._data_buffer["time_series"] is not None:
                if self._interval is not None:
                    t_begin = self._data_buffer["time_stamps"][0] + self._interval # shift forward by interval
                elif self._data_buffer["time_stamps"].shape[0] > 1:
                    t_begin = self._data_buffer["time_stamps"][1] # shift forward by 1 sample
                else:
                    # rare situation where the buffer only contains one sample
                    # and the interval is None. Shift forward by a very small amount.
                    t_begin = self._data_buffer["time_stamps"][0] + 10**(-6) # shift forward by 1 microsecond
            else:
                t_begin = 0  # i.e. all data is valid

        t_begin += self.relative_start

        # pull the data in chunks until we get the total number of samples
        samples_polled = 0

        # First, pull the data required data from the buffer
        if self._data_buffer["time_series"] is not None:

            # Create a boolean array to index the data buffer for the required data
            valid_indices = self._data_buffer["time_stamps"] >= t_begin

            # Find the number of samples in the buffer that are valid
            samples_polled = np.sum(valid_indices)

            # discard old data
            self._data_buffer["time_series"] = self._data_buffer["time_series"][:, valid_indices]
            self._data_buffer["time_stamps"] = self._data_buffer["time_stamps"][valid_indices]

            # If the number of samples in the buffer is greater than the number of samples required, extract the required data
            if samples_polled >= self.Ns:
                # Buffer contains a backlog of data, warn that execution may be too slow for target polling rate
                warnings.warn("Buffer contains a backlog of data. Execution may be too slow for target polling rate.", RuntimeWarning, stacklevel=2)

                if self.marker_coupled:
                    # if this is a marker-coupled stream, use the oldest valid data in the buffer
                    # to ensure that the data is aligned with the marker
                    self._trial_data = self._data_buffer["time_series"][:, :self.Ns]
                    self._trial_timestamps = self._data_buffer["time_stamps"][:self.Ns]
                else:
                    # if this is a marker-uncoupled stream, use the newest valid data in the buffer
                    # to ensure that the data is as recent as possible
                    self._trial_data = self._data_buffer["time_series"][:, -self.Ns:]
                    self._trial_timestamps = self._data_buffer["time_stamps"][-self.Ns:]

            # If the number of valid samples in the buffer is less than the number of samples required, extract all the data in the buffer
            else:
                self._trial_data[:, :samples_polled] = self._data_buffer["time_series"]
                self._trial_timestamps[:samples_polled] = self._data_buffer["time_stamps"]

        # If the buffer does not contain enough data, pull data from the inlet
        null_reads = 0
        while samples_polled < self.Ns:
            data, timestamps = self._data_inlet.pull_chunk(timeout=0.0)

            if len(timestamps) > 0:
                timestamps = np.asarray(timestamps)
                null_reads = 0  # reset the null reads counter

                # apply time correction to timestamps
                self._time_correction = self._data_inlet.time_correction()
                timestamps += self._time_correction

                # check if the data is within the target time window
                if np.any(timestamps >= t_begin):
                    # convert data to numpy arrays
                    data = np.asarray(data).T # now in Nchannel x Nsamples format
                    valid_timestamps = timestamps >= t_begin

                    # discard extra channels and old data
                    data = data[np.ix_(self.channels, valid_timestamps)]
                    timestamps = timestamps[valid_timestamps]

                    # append the latest chunk to the trial_data array
                    # start by indentifying the start and end indices
                    # of the source and destination arrays
                    chunk_sz = data.shape[1]
                    if samples_polled + chunk_sz > self.Ns:
                        # more data in the chunk than required
                        dst_end_ix = self.Ns
                        src_end_ix = self.Ns - samples_polled
                    else:
                        # less data in the chunk than required
                        dst_end_ix = samples_polled + chunk_sz
                        src_end_ix = chunk_sz

                    self._trial_data[:, samples_polled:dst_end_ix] = data[:, :src_end_ix]
                    self._trial_timestamps[samples_polled:dst_end_ix] = timestamps[:src_end_ix]

                    if dst_end_ix == self.Ns:
                        # we have polled enough data, update the buffer
                        # with the latest data plus any extra data
                        # that we did not use in this trial
                        self._data_buffer["time_series"] = np.concatenate(
                            (self._trial_data, data[:, src_end_ix:]), axis=1
                        )
                        self._data_buffer["time_stamps"] = np.concatenate(
                            (self._trial_timestamps, timestamps[src_end_ix:])
                        )

                    samples_polled += chunk_sz
            else:
                null_reads += 1
                if null_reads > self.MAX_NULL_READS:
                    raise RuntimeError(
                        f"The stream has not been updated in the last {self.MAX_NULL_READS} read attemps. Please check the stream."
                    )
                time.sleep(0.1)

        if self.marker_coupled:
            # reset the maker peeked flag since we have polled new data
            self._already_peeked = False

        # if in epoched mode, increment the epochs polled counter
        if self.mode == 'epoched':
            self._epochs_polled += 1

        return self._trial_data

    def peek_marker(self):
        """
        Peek at the next marker in the marker stream

        Returns
        -------
        marker : str
            The marker string

        """

        if not self._active:
            raise RuntimeError("InputLSLStream.peek_marker() called on inactive stream. Please call update_input_streams() first to configure the stream object.")

        if self._already_peeked:
            return self._peeked_marker

        marker, t = self.peek_marker_inlet.pull_sample()
        read_attemps = 0
        while (self.marker_pattern is not None and
               not self.marker_pattern.match(marker[0])):
            marker, t = self.peek_marker_inlet.pull_sample(timeout=0.0)

            read_attemps += 1
            if read_attemps > self.MAX_NULL_READS:
                raise RuntimeError(
                    f"The marker stream has not been updated in the last {self.MAX_NULL_READS} read attemps. Please check the stream."
                )

        self._peeked_marker = marker[0]
        self._already_peeked = True
        return marker[0]

    def last_marker(self):
        """
        Get the last marker in the marker stream

        Returns
        -------
        marker : str
            The last marker string

        """
        if not self._active:
            raise RuntimeError("InputLSLStream.last_marker() called on inactive stream. Please call update_input_streams() first to configure the stream object.")

        return self._marker_buffer["time_series"]
    
    def change_mode(self, new_mode, interval=None, n_epochs=None):
        """
        Change the mode of the stream

        Parameters
        ----------
        new_mode : str
            The new mode of the stream. Can be 'single', 'continuous', or 'epoched'

        """
        if new_mode not in ('single', 'continuous', 'epoched'):
            raise ValueError(f"Invalid mode {new_mode}. Mode must be 'single', 'continuous', or 'epoched'.")

        if new_mode in ('single', 'epoched'):
            # ensure that a valid marker stream is available
            if self._marker_inlet is None:
                raise RuntimeError("Cannot change mode to 'single' or 'epoched' without a valid marker stream.")
            
        self.mode = new_mode
        if new_mode in ('continuous', 'epoched'):
            if interval is not None and interval != self._interval:
                self._interval = interval
            
        if new_mode == 'epoched':
            if n_epochs is not None and n_epochs != self.n_epochs:
                self.n_epochs = n_epochs
            self._epochs_polled = 0

        # flush the data buffer to ensure that the next trial is polled correctly
        self.flush_data_buffer()

    def flush_data_buffer(self, time_cutoff=None):
        """
        Flush the data buffer

        Parameters
        ----------
        time_cutoff : float
            The time cutoff for the data buffer. If None, all data will be flushed.
        """

        if time_cutoff is None:
            # clear the entire buffer
            self._data_buffer = {"time_series": None, "time_stamps": None}
            self._marker_buffer = {"time_series": None, "time_stamps": None}
        else:
            # reserve the data that is newer than the time cutoff
            for buf in (self._data_buffer, self._marker_buffer):
                if buf["time_series"] is not None:
                    valid_indices = buf["time_stamps"] >= time_cutoff + self.relative_start
                    buf["time_series"] = buf["time_series"][:, valid_indices]
                    buf["time_stamps"] = buf["time_stamps"][valid_indices]
        

        if time_cutoff is None:
            # flush the inlet streams
            if self._marker_inlet is not None:
                self._marker_inlet.flush()

            self._data_inlet.flush()
        elif self.mode == 'continuous':
            # poll data and discard it until the time cutoff is reached
            adj_cutoff = time_cutoff + self.relative_start
            null_reads = 0
            cutoff_reached = False
            while not cutoff_reached:
                data, timestamps = self._data_inlet.pull_chunk(timeout=0.0)

                if len(timestamps) > 0:
                    timestamps = np.asarray(timestamps)
                    null_reads = 0
                    # apply time correction to timestamps
                    self._time_correction = self._data_inlet.time_correction()
                    timestamps += self._time_correction

                    # check if the data is within the target time window
                    if np.any(timestamps >= adj_cutoff):
                        # convert data to numpy arrays
                        data = np.asarray(data).T
                        valid_timestamps = timestamps >= adj_cutoff

                        # discard extra channels and old data
                        data = data[np.ix_(self.channels, valid_timestamps)]
                        timestamps = timestamps[valid_timestamps]

                        # append the latest chunk to the trial_data array
                        # start by indentifying the start and end indices
                        # of the source and destination arrays
                        if self._data_buffer["time_series"] is None:
                            # first chunk of data, create the buffer
                            self._data_buffer["time_series"] = data
                            self._data_buffer["time_stamps"] = timestamps
                        else:
                            # append the data to the buffer
                            self._data_buffer["time_series"] = np.concatenate(
                                (self._data_buffer["time_series"], data), axis=1
                            )
                            self._data_buffer["time_stamps"] = np.concatenate(
                                (self._data_buffer["time_stamps"], timestamps)
                            )

                        cutoff_reached = True

                else:
                    null_reads += 1
                    if null_reads > self.MAX_NULL_READS:
                        raise RuntimeError(
                            f"The stream has not been updated in the last {self.MAX_NULL_READS} read attemps. Please check the stream."
                        )
                    time.sleep(0.01)
        else:
            raise RuntimeError("Cannot flush this type of stream.")


    def update_input_streams(
        self,
        pred=None,
        channels=None,
        marker_coupled=True,
        marker_fmt=None,
        marker_pred=None,
        stream_info=None,
        marker_stream_info=None,
        Ns=1
    ):
        """
        Update the input stream with new parameters

        Parameters
        ----------
        pred : str
            The predicate string, e.g. "name='BioSemi'" or "type='EEG' and starts-with(name, 'BioSemi') and
            count(description/desc/channels/channel)=32"
        channels : tuple of ints
            Index value of channels to poll from the stream, if None all channels will be polled
        marker_coupled : bool
            true if there is an associated marker to indicate relative time where data should begin to be polled
        marker_fmt : Regex or list
            Regular expression template of the marker to be matched, if none all markers will be matched. Alternatively, a list of markers can be provided.
        marker_pred : str
            The predicate string for the marker stream
        stream_info : pylsl.StreamInfo
            The stream info object for the stream can be passed instead of the predicate to avoid the need to resolve the stream
        marker_stream_info : pylsl.StreamInfo
            The stream info object for the marker stream can be passed instead of the predicate to avoid the need to resolve the stream
        Ns : int, default = 1
            The number of samples to be extracted per poll.

        """
        if self._active:
            return

        if not stream_info:
            # resolve the stream on the LSL network
            available_streams = pylsl.resolve_bypred(pred)
        else:
            available_streams = [stream_info]

        if len(available_streams) == 0:
            raise RuntimeError("No streams found matching the predicate")
        elif len(available_streams) > 1:
            warnings.warn(
                "More than one stream found matching the predicate. Using the first stream found.",
                RuntimeWarning, stacklevel=2
            )

        self._data_buffer = {"time_series": None, "time_stamps": None}
        self._data_inlet = pylsl.StreamInlet(
            available_streams[0],
            processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter,
            recover=False,
        )
        self._data_inlet.open_stream()

        if channels is not None:
            if max(channels) >= self._data_inlet.channel_count or min(channels) < 0:
                raise ValueError(
                    "The number of channels in the stream does not match the channel indices specified in the channels parameter. Please check the channels parameter and try again."
                )
            self.channels = channels
        else:
            self.channels = tuple([_ for _ in range(self._data_inlet.channel_count)])

        if marker_coupled:
            if not marker_stream_info:
                # resolve the stream on the LSL network
                marker_streams = pylsl.resolve_bypred(marker_pred)
            else:
                marker_streams = [marker_stream_info]

            if len(marker_streams) == 0:
                raise RuntimeError("No marker streams found matching the predicate")
            elif len(marker_streams) > 1:
                warnings.warn(
                    "More than one marker stream found matching the predicate. Using the first stream found.",
                    RuntimeWarning, stacklevel=2
                )

            self._marker_inlet = pylsl.StreamInlet(marker_streams[0])
            self._peek_marker_inlet = pylsl.StreamInlet(marker_streams[0])

            # open the inlet
            self._marker_inlet.open_stream()
            self._peek_marker_inlet.open_stream()

            if marker_fmt:
                self.marker_pattern = re.compile(marker_fmt)

        self.Ns = Ns

        # allocate array for trial data and timestamps
        self._trial_data = np.zeros((len(self.channels), self.Ns))
        self._trial_timestamps = np.zeros((self.Ns,))

        self._active = True

    @classmethod
    def create_marker_coupled_data_stream(
        cls,
        sess,
        pred=None,
        channels=None,
        relative_start=0,
        marker_fmt=None,
        marker_pred="type='Markers'",
        stream_info=None,
        marker_stream_info=None,
        Ns=1,
        active=True,
        mode='single',
        epoch_interval=None,
        n_epochs=1
    ):
        """
        Create a LSLStream data object that maintains a data stream and a
        marker stream

        Parameters
        -----------

        sess : session object
            Session object where the data source will exist
        pred : str
            The predicate string, e.g. "name='BioSemi'" or "type='EEG' and starts-with(name, 'BioSemi') and
            count(description/desc/channels/channel)=32"
        channels : tuple or list of ints
            Index value of channels to poll from the stream, if None all channels will be polled
        marker_fmt : str
            Regular expression template of the marker to be matched, if none all markers will be matched
        marker_pred : str
            Predicate string to match the marker stream, if None all streams will be matched
        stream_info : StreamInfo object
            StreamInfo object to use for the data stream, if None a default StreamInfo object will be created
        Ns : int, default = 1
            Number of samples to be extracted per poll.
        mode: str, default = 'single'
            Mode of the stream. Can be 'single', 'continuous', or 'epoched'. 
            If 'single', the stream will be polled for a single trial at a time.
            If 'continuous', the first trial will be couple to a marker, and subsequent trials 
            will be polled based on the interval. If 'epoched', the first trial will be coupled 
            to a marker, and then a fixed number of subsequent trials, defined by the n_epochs 
            parameter, will be polled based on the interval. The interval parameter must be 
            provided for both the epoched and continuous modes.
        epoch_interval: float, default = None
            The minimum interval between polling the stream for new data. Only used for the epoched and continuous modes.
        n_epochs: int, default = 1
            Number of epochs to be extracted per poll. Only used for the epoched mode.
        """
        src = cls(
            sess,
            pred,
            channels,
            relative_start,
            True,
            marker_fmt,
            marker_pred,
            stream_info,
            marker_stream_info,
            active,
            Ns=Ns,
            mode=mode,
            interval=epoch_interval,
            n_epochs=n_epochs
        )
        sess.add_to_session(src)

        return src

    @classmethod
    def create_marker_uncoupled_data_stream(
        cls, 
        sess, 
        pred=None, 
        channels=None, 
        relative_start=0,
        active=True, 
        interval=None, 
        Ns=1
    ):
        """
        Create a LSLStream data object that maintains only a data stream with
        no associated marker stream
        Parameters
        ----------
        sess : session object
            Session object where the data source will exist
        pred : str
            The predicate string, e.g. "name='BioSemi'" or "type='EEG' and starts-with(name, 'BioSemi') and
            count(description/desc/channels/channel)=32"
        channels : tuple or list of ints
            Index value of channels to poll from the stream, if None all channels will be polled
        active : bool
            Flag to indicate whether the stream is active or will be activated in the future
        interval : float
            The minimum interval at which the stream will be polled
        Ns : int, default = 1
            Number of samples to be extracted per poll.
        """
        src = cls(
            sess, pred, channels, relative_start, marker_coupled=False, active=active,
            interval=interval, Ns=Ns, mode='continuous')
        sess.add_to_session(src)

        return src


class OutputLSLStream(MPBase):
    """
    An object for maintaining an LSL outlet
    """

    def __init__(self, sess, stream_info, filesave=None, chunk_size=0, max_buffer=360):
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
        super().__init__(MPEnums.SRC, sess)
        self._sess = sess
        self.stream_info = stream_info
        # resolve the stream on the LSL network
        self._lsl_marker_outlet = pylsl.StreamOutlet(stream_info, chunk_size, max_buffer)
        self._liesl_session = None
        # Start LieSL recording if the user has specified a filesave

        warnings.filterwarnings(
            action="ignore", category=RuntimeWarning, module="subprocess"
        )
        output_save_thread = threading.Thread(target=self.check_status, args=(filesave,))
        output_save_thread.start()

    def check_status(self, filesave):
        """
        TODO: add description
        Parameters
        ----------
        filesave: TODO - add type
        """
        if filesave is not None:
            streamargs = [
                {
                    "name": self.stream_info.name(),
                    "type": self.stream_info.type(),
                    "channel_count": self.stream_info.channel_count(),
                    "nominal_srate": self.stream_info.nominal_srate(),
                    "channel_format": self.stream_info.channel_format(),
                    "source_id": self.stream_info.source_id(),
                }
            ]
            self._liesl_session = liesl.Session(
                mainfolder=f"{os.path.dirname(os.path.realpath(__file__))}\labrecordings",
                streamargs=streamargs,
            )

            with self._liesl_session(filesave):
                while True:
                    time.sleep(0.1)
                    if not threading.main_thread().is_alive():
                        # Suppress output from pyLiesl
                        sys.stdout = open(os.devnull, "w")
                        sys.stderr = open(os.devnull, "w")
                        self._liesl_session.stop_recording()
                        sys.stdout = sys.__stdout__
                        sys.stderr = sys.__stderr__
                        return
        else:
            warnings.warn("No file save specified. Data will not be saved to disk.")
            return

    def push_data(self, data, label=None):
        """
        Push data to the outlet stream.

        Parameters
        ----------
        data: Tensor
            Data to be pushed to the output stream
        """

        try:
            self._lsl_marker_outlet.push_sample(data, pylsl.local_clock())

        except (ValueError, TypeError) as ve:
            try:
                self._lsl_marker_outlet.push_sample(data[0], pylsl.local_clock())

            except Exception as e:
                additional_msg = "Push data - Irreparable Error in LSL Output. No data pushed to output stream"
                if sys.version_info[:2] >= (3, 11):
                    e.add_note(additional_msg)
                else:
                    pretty_msg = f"{'*'*len(additional_msg)}\n{additional_msg}\n{'*'*len(additional_msg)}"
                    print(pretty_msg)
                raise

    @classmethod
    def _create_outlet_from_streaminfo(cls, sess, stream_info, filesave=None):
        """
        Factory method to create a OutletLSLStream mindpype object from a pylsl.StreamInfo object.

        Parameters
        -----------

        sess : session object
            Session object where the data source will exist
        stream_info : pylsl.StreamInfo object
            pylsl.StreamInfo object that describes the stream to be created

        Returns
        -------
        src: OutputLSLStream
            Output LSL Stream
        """
        src = cls(sess, stream_info, filesave)
        sess.add_to_session(src)

        return src

    @classmethod
    def create_outlet(
        cls,
        sess,
        name="untitled",
        type="",
        channel_count=1,
        nominal_srate=0.0,
        channel_format=1,
        source_id="",
        filesave=None,
    ):
        """
        Factory Method to create an OutletLSLStream mindpype object from scratch.

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

        Returns
        -------

        src: OutputLSLStream
            Output LSL Stream
        """

        stream_info = pylsl.StreamInfo(
            name,
            type,
            channel_count,
            nominal_srate,
            channel_format,
            source_id="1007988689",
        )
        src = cls(sess, stream_info, filesave)
        sess.add_to_session(src)

        return src
