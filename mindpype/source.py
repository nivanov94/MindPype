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
import time


class MPXDF(MPBase):
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
        For P300-type setups, the tasks 'target' and 'non-target'/'flash' can be used.

    channels : list or tuple of int
        Values corresponding to the stream channels used during the session

    relative_start : float, default = 0
        Value corresponding to the start of the trial relative to the marker onset.

    Ns : int, default = 1
        Number of samples to be extracted per trial. For epoched and class-separated data, this value determines the
        size of each epoch, whereas this value is used in polling for continuous data.

    mode : 'continuous', 'class-separated' or 'epoched', default = 'epoched'
        Mode indicates whether the inputted data will be epoched sequentially as individual trials,
        epoched by class, or to leave the data in a continuous format

    .. warning::
       The task list used in the MPXDF object MUST REFLECT the task list used in the XDF file.
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
    """

    def __init__(self, sess, files, tasks, channels, relative_start=0, Ns=1, stype='EEG', mode="epoched"):
        """
        Create a new xdf file reader interface
        """
        super().__init__(MPEnums.SRC, sess)

        if type(files) == str:
            files = [files]

        self.files = files
        self.relative_start = relative_start
        self.Ns = Ns
        self.tasks = tasks
        self.channels = channels
        self.label_counter = None
        self.mode = mode
        self.stype = stype

        trial_data = {task: [] for task in tasks}

        # Class separated mode will epoch the data by class, and will poll the data for the
        # next Ns samples of the specified class each time the poll_data method is called.
        if mode == "class-separated":
            combined_marker_streams = {"time_series": None, "time_stamps": None}
            for filename in files:
                data, header = pyxdf.load_xdf(filename)

                for stream in data:
                    if (stream["info"]["type"][0] == "Marker" or
                        stream["info"]["type"][0] == "Markers"):
                        marker_stream = stream

                    elif stream["info"]["type"][0] == self.stype:
                        data_stream = stream

                sample_indices = np.zeros(data_stream["time_stamps"].shape)  # used to extract data samples, pre-allocated here

                total_tasks = 0

                for i_m, markers in enumerate(marker_stream["time_series"]):
                    marker_value = markers[0]
                    curr_task = ""

                    for task in self.tasks:
                        if task in marker_value:
                            curr_task = task

                            marker_time = marker_stream["time_stamps"][i_m]
                            total_tasks += 1

                            # compute the correct start and end indices for the current trial
                            data_window_start = marker_time + relative_start

                            sample_indices = np.array(data_stream["time_stamps"] >= data_window_start)

                            sample_data = data_stream["time_series"][sample_indices, :][:, channels].T  # Nc X len(data_stream)
                            trial_data[curr_task].append(sample_data[:,:int(Ns)])  # Nc x Ns

                if combined_marker_streams["time_series"] is None:
                    combined_marker_streams["time_series"] = marker_stream["time_series"]
                    combined_marker_streams["time_stamps"] = marker_stream["time_stamps"]
                else:
                    combined_marker_streams["time_series"] = np.concatenate(
                        (
                            combined_marker_streams["time_series"],
                            marker_stream["time_series"],
                        ),
                        axis=0,
                    )
                    combined_marker_streams["time_stamps"] = np.concatenate(
                        (
                            combined_marker_streams["time_stamps"],
                            marker_stream["time_stamps"],
                        ),
                        axis=0,
                    )

            for task in trial_data:
                trial_data[task] = np.stack(trial_data[task], axis=0)  # Nt x Nc x Ns

            self.trial_data = {
                "Data": {"time_series": trial_data, "time_stamps": data_stream},
                "Markers": combined_marker_streams,
            }
            self.label_counter = {task: 0 for task in tasks}


        # Continuous mode will leave the data in a continuous format, and will poll the data for the next Ns samples
        elif mode == "continuous":
            # Counter to track how many trials have been extracted previously
            self.cont_trial_num = 0
            data_stream = None
            marker_stream = None

            first_marker = []

            for filename in files:
                data, header = pyxdf.load_xdf(filename)

                # First order the files by the first marker value
                for stream in data:
                    if (stream["info"]["type"][0] == "Marker" or
                        stream["info"]["type"][0] == "Markers"):
                        first_marker.append(stream["time_series"][0][0])

            # Sort the files by the first marker value
            files = [x for _, x in sorted(zip(first_marker, files))]

            for filename in files:
                data, header = pyxdf.load_xdf(filename)

                # Iterate through all streams in every file, add current file's data to the previously loaded data
                for stream in data:
                    if (stream["info"]["type"][0] == "Marker" or
                        stream["info"]["type"][0] == "Markers"):
                        marker_stream = stream

                    # If the data stream already exists, concatenate the new data to the existing data
                    elif stream["info"]["type"][0] == self.stype:
                        data_stream = stream

            # Extract the data from the data stream
            data_stream["time_series"] = data_stream["time_series"][:, channels].T


            self.trial_data = {"Data": data_stream, "Markers": marker_stream}
            self.label_counter = {task: 0 for task in tasks}

        # Epoched mode will epoch the data sequentially, and will poll the data for the next Ns samples of the next trial
        elif mode == "epoched":
            self.epoched_counter = 0
            data_stream_data = None
            data_stream_stamps = None
            data_stream = None
            marker_stream = None
            Ns = int(Ns)
            epoch_num = 0

            self.trial_data = {"Data" : {"time_stamps": None, "time_series": None},
                               "Markers" : {"time_stamps" : None, "time_series": None}}

            for filename in files:
                # Load the data from the current xdf file
                data, header = pyxdf.load_xdf(filename)
                # Iterate through all streams in every file, add current file's data to the previously loaded data
                for stream in data:
                    if (stream["info"]["type"][0] == "Marker"
                        or stream["info"]["type"][0] == "Markers"):
                        marker_stream = stream

                    elif stream["info"]["type"][0] == self.stype:
                        data_stream = stream

                total_markers = len(marker_stream["time_stamps"])
                data_stream_data = np.zeros((total_markers, len(channels), Ns))
                data_stream_stamps = np.zeros((total_markers, Ns))
                valid_markers_tseries = [_ for _ in range(total_markers)]
                valid_markers_tstamps = np.zeros((total_markers,))

                # Actual epoching operation
                valid_epoch_num = 0
                for epoch_num in range(total_markers):
                    marker = marker_stream["time_series"][epoch_num]
                    print(marker)
                    if marker[0] not in self.tasks:
                        continue

                    # Find the marker value where the current epoch starts
                    marker_time = marker_stream["time_stamps"][epoch_num]
                    # Correct the starting time of the epoch based on the relative start time
                    data_window_start = marker_time + relative_start

                    # Find the index of the first sample after the marker
                    first_sample_index = np.where(data_stream["time_stamps"] >= data_window_start)[0][0]
                    # Find the index of the last sample in the window
                    final_sample_index = first_sample_index + Ns
                    if final_sample_index <= data_stream["time_series"].shape[0]:
                        # Extract the data from the data stream
                        data_stream_data[valid_epoch_num, :, :] = data_stream["time_series"][first_sample_index:final_sample_index,:][:, channels].T
                        data_stream_stamps[valid_epoch_num, :] = data_stream["time_stamps"][first_sample_index:final_sample_index]
                        valid_markers_tseries[valid_epoch_num] = marker_stream["time_series"][epoch_num]
                        valid_markers_tstamps[valid_epoch_num] = marker_stream["time_stamps"][epoch_num]
                        valid_epoch_num += 1

                print(f"total markers : {total_markers}, valid_epochs: {valid_epoch_num}")

                if self.trial_data["Data"]["time_series"] is None:
                    self.trial_data = {
                        "Data": {
                            "time_stamps" : data_stream_stamps[:valid_epoch_num],
                            "time_series" : data_stream_data[:valid_epoch_num],
                        },
                        "Markers": {
                            "time_stamps" : valid_markers_tstamps[:valid_epoch_num],
                            "time_series" : valid_markers_tseries[:valid_epoch_num]
                        }
                    }
                else:
                    self.trial_data["Data"]["time_stamps"] = np.concatenate((self.trial_data["Data"]["time_stamps"],
                                                                             data_stream_stamps[:valid_epoch_num]),
                                                                            axis=0)
                    self.trial_data["Data"]["time_series"] = np.concatenate((self.trial_data["Data"]["time_series"],
                                                                             data_stream_data[:valid_epoch_num]),
                                                                            axis=0)
                    self.trial_data["Markers"]["time_stamps"] = np.concatenate((self.trial_data["Markers"]["time_stamps"],
                                                                             valid_markers_tstamps[:valid_epoch_num]),
                                                                            axis=0)
                    self.trial_data["Markers"]["time_series"] = np.concatenate((self.trial_data["Markers"]["time_series"],
                                                                             valid_markers_tseries[:valid_epoch_num]),
                                                                            axis=0)




    def poll_data(self, Ns=1, label=None):

        """
        Polls the data source for new data.

        Parameters
        ----------

        Ns : int, default = 1
            Number of samples to be extracted per trial. For continuous data, determines the size of
            the extracted sample. This value is disregarded for epoched and class-separated data.

            This parameter is used and required for continuous data only.

        label : string
            Marker to be used for polling. Number of trials previously extracted is recorded internally.
            This marker must be present in the XDF file and must be present in the list of tasks used in
            initialization.

            This parameter is used and required for epoched and class-separated data only.


        """

        if self.mode == "class-separated":
            # Extract sample data from epoched trial data and increment the label counter
            sample_data = self.trial_data[label][self.label_counter[label], :, :]
            self.label_counter[label] += 1

            return sample_data

        elif self.mode == "epoched":
            # Extract sample data from epoched trial data and increment the label counter
            sample_data = self.trial_data["Data"]["time_series"][self.epoched_counter, :, :]
            self.epoched_counter += 1

            return sample_data

        elif self.mode == "continuous":

            # Extract the nth marker timestamp, corresponding to the nth trial in the XDF file
            data_window_start = (
                self.trial_data["Markers"]["time_stamps"][self.cont_trial_num] + self.relative_start
            )

            # Construct the boolean array for samples that fall after the marker timestamp
            sample_indices = self.trial_data["Data"]["time_stamps"] >= data_window_start
            sample_data = self.trial_data["Data"]["time_series"][:, sample_indices]

            sample_data = sample_data[:, :Ns]  # Nc x Ns
            self.cont_trial_num += 1

            return sample_data

    def load_into_tensor(self):
        """
        Loads entirity of MindPypeXDF data object into a tensor.
        Returns 4 MindPype Tensor objects, in the following order.

            1. Tensor containing the Stream data
            2. Tensor containing the Stream timestamps
            3. Tensor containing the Marker data
            4. Tensor containing the Marker timestamps

        Parameters
        ----------
        None

        Returns
        -------
        ret : Tensor
            Tensor containing the stream data

        ret_timestamps : Tensor
            Tensor containing the stream timestamps

        ret_labels : Tensor
            Tensor containing the Marker data

        ret_labels_timestamps : Tensor
            Tensor containing the Marker timestamps
        """
        if self.trial_data and self.mode in ("continuous", "epoched"):
            ret = Tensor.create_from_data(
                self.session,
                self.trial_data["Data"]["time_series"],
            )

            ret_timestamps = Tensor.create_from_data(
                self.session,
                self.trial_data["Data"]["time_stamps"],
            )

            ret_labels = Tensor.create_from_data(
                self.session,
                self.trial_data["Markers"]["time_series"],
            )

            ret_labels_timestamps = Tensor.create_from_data(
                self.session,
                self.trial_data["Markers"]["time_stamps"],
            )


        elif self.trial_data and self.mode == "class-separated":
            warnings.warn(
                "Class-separated data is not yet supported for Tensor loading. This must be performed manually. Use tensor.create_from_data() with the appropriate class-separated dataset to create the tensor",
                RuntimeWarning,
                stacklevel=5,
            )
            ret, ret_timestamps, ret_labels, ret_labels_timestamps = None, None, None, None

        return ret, ret_timestamps, ret_labels, ret_labels_timestamps

    @classmethod
    def create_continuous(cls, sess, files, tasks, channels, relative_start=0, Ns=1):
        """
        Factory Method for creating continuous XDF File input source.


        Parameters
        ---------
        sess : Session Object
            Session where the MPXDF data source will exist.

        files : list of str
            XDF file(s) where data should be extracted from.

        tasks : list or tuple of strings
            List or Tuple of strings corresponding to the tasks to be completed by the user.
            For P300-type setups, the tasks 'target' and 'non-target'/'flash' can be used.

        channels : list or tuple of int
            Values corresponding to the data stream channels used during the session

        relative_start : float, default = 0
            Value corresponding to the start of the trial relative to the marker onset.

        Ns : int, default = 1
            Number of samples to be extracted per trial. For epoched data, this value determines the
            size of each epoch, whereas this value is used in polling for continuous data.

        """

        src = cls(sess, files, tasks, channels, relative_start, Ns, mode="continuous")

        sess.add_ext_src(src)

        return src


    @classmethod
    def create_class_separated(
        cls, sess, files, tasks, channels, relative_start=0, stype='EEG', Ns=1):
        """
        Factory Method for creating class-separated XDF File input source.

        Parameters
        ---------
        sess : Session Object
            Session where the MPXDF data source will exist.

        files : list of str
            XDF file(s) where data should be extracted from.

        tasks : list or tuple of strings
            List or Tuple of strings corresponding to the tasks to be completed by the user.
            For P300-type setups, the tasks 'target' and 'non-target'/'flash' can be used.

        channels : list or tuple of int
            Values corresponding to the data stream channels used during the session

        relative_start : float, default = 0
            Value corresponding to the start of the trial relative to the marker onset.

        stype : str, default = EEG
            String indicating the data type

        Ns : int, default = 1
            Number of samples to be extracted per trial. For class-separated data, this value determines the
            size of each epoch, whereas this value is used in polling for continuous data.

        """

        src = cls(
            sess, files, tasks, channels, relative_start, Ns, stype=stype, mode="class-separated"
        )

        sess.add_ext_src(src)

        return src

    @classmethod
    def create_epoched(cls, sess, files, tasks, channels, relative_start=0, Ns=1, stype='EEG'):

        """
        Factory Method for creating epoched XDF File input source.

        Parameters
        ---------
        sess : Session Object
            Session where the MPXDF data source will exist.

        files : list of str
            XDF file(s) where data should be extracted from.

        tasks : list or tuple of strings
            List or Tuple of strings corresponding to the tasks to be completed by the user.
            For P300-type setups, the tasks 'target' and 'non-target'/'flash' can be used.

        channels : list or tuple of int
            Values corresponding to the data stream channels used during the session

        relative_start : float, default = 0
            Value corresponding to the start of the trial relative to the marker onset.

        stype : str, default = EEG
            String indicating the data type

        Ns : int, default = 1
            Number of samples to be extracted per trial. For class-separated data, this value determines the
            size of each epoch, whereas this value is used in polling for continuous data.

        """
        src = cls(sess, files, tasks, channels, relative_start, Ns, stype=stype, mode="epoched")

        sess.add_ext_src(src)

        return src

    @classmethod
    def create(
        cls, sess, files, tasks, channels, relative_start=0, Ns=1, mode="epoched"
    ):
        """
        Factory Method for creating epoched XDF File input source.

        Parameters
        ---------
        sess : Session Object
            Session where the MPXDF data source will exist.

        files : list of str
            XDF file(s) where data should be extracted from.

        tasks : list or tuple of strings
            List or Tuple of strings corresponding to the tasks to be completed by the user.
            For P300-type setups, the tasks 'target' and 'non-target'/'flash' can be used.

        channels : list or tuple of int
            Values corresponding to the data stream channels used during the session

        relative_start : float, default = 0
            Value corresponding to the start of the trial relative to the marker onset.

        Ns : int, default = 1
            Number of samples to be extracted per trial. For epoched data, this value determines the
            size of each epoch, whereas this value is used in polling for continuous data.

        """

        src = cls(sess, files, tasks, channels, relative_start, Ns, mode)

        sess.add_ext_src(src)

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

    """

    MAX_NULL_READS = 100

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
        interval=None
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

        .. note::
            The active parameter is used when the session is created before the LSL stream is started, or the stream is
            not available when the session is created. In that case, the stream can be updated later by calling the update_input_stream() method.
        """
        super().__init__(MPEnums.SRC, sess)
        self._active = active
        self._marker_coupled = marker_coupled

        self._marker_inlet = None
        self._marker_pattern = None
        self._relative_start = relative_start
        self._already_peeked = False
        self._peeked_marker = None
        self._marker_buffer = {"time_series": None, "time_stamps": None} # only keeps most recent value, can expand in future if needed
        self._time_correction = None
        self._interval = interval
        self._channels = channels

        if active:
            self.update_input_stream(pred, channels, marker_coupled, marker_fmt, marker_pred, stream_info, marker_stream_info)

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

        if not self.active:
            raise RuntimeWarning("InputLSLStream.poll_data() called on inactive stream. Please call update_input_streams() first to configure the stream object.")

        if self._marker_inlet is not None:
            # start by getting the timestamp for this trial's marker
            t_begin = None
            null_reads = 0
            while t_begin is None:
                marker, t = self._marker_inlet.pull_sample(timeout=0.0)

                if marker is not None:
                    null_reads = 0  # reset the null reads counter
                    marker = marker[0]  # extract the string portion of the marker

                    if (self.marker_pattern == None) or self._marker_pattern.match(marker):
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
            if self._data_buffer["Data"] is not None:
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

        t_begin += self._relative_start

        # pull the data in chunks until we get the total number of samples
        trial_data = np.zeros((len(self.channels), Ns))  # allocate the array
        trial_timestamps = np.zeros((Ns,))
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
            if samples_polled >= Ns:
                # Buffer contains a backlog of data, warn that execution may be too slow for target polling rate
                warnings.warn("Buffer contains a backlog of data. Execution may be too slow for target polling rate.", RuntimeWarning, stacklevel=2)

                if self._marker_coupled:
                    # if this is a marker-coupled stream, use the oldest valid data in the buffer
                    # to ensure that the data is aligned with the marker
                    trial_data = self._data_buffer["time_series"][:, :Ns]
                    trial_timestamps = self.data_buffer["time_stamps"][:Ns]
                else:
                    # if this is a marker-uncoupled stream, use the newest valid data in the buffer
                    # to ensure that the data is as recent as possible
                    trial_data = self._data_buffer["time_series"][:, -Ns:]
                    trial_timestamps = self.data_buffer["time_stamps"][-Ns:]

            # If the number of valid samples in the buffer is less than the number of samples required, extract all the data in the buffer
            else:
                trial_data[:, :samples_polled] = self.data_buffer["time_series"]
                trial_timestamps[:samples_polled] = self.data_buffer["time_stamps"]

        # If the buffer does not contain enough data, pull data from the inlet
        null_reads = 0
        while samples_polled < Ns:
            data, timestamps = self.data_inlet.pull_chunk(timeout=0.0)
            timestamps = np.asarray(timestamps)

            if len(timestamps) > 0:
                null_reads = 0  # reset the null reads counter

                # apply time correction to timestamps
                self._time_correction = self.data_inlet.time_correction()
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
                    if samples_polled + chunk_sz > Ns:
                        # more data in the chunk than required
                        dst_end_ix = Ns
                        src_end_ix = Ns - samples_polled
                    else:
                        # less data in the chunk than required
                        dst_end_ix = samples_polled + chunk_sz
                        src_end_ix = chunk_sz

                    trial_data[:, samples_polled:dst_end_ix] = data[:, :src_end_ix]
                    trial_timestamps[samples_polled:dst_end_ix] = timestamps[:src_end_ix]

                    if dst_end_ix == Ns:
                        # we have polled enough data, update the buffer
                        # with the latest data plus any extra data
                        # that we did not use in this trial
                        self.data_buffer["Data"] = np.concatenate(
                            (trial_data, data[:, src_end_ix:]), axis=1
                        )
                        self.data_buffer["time_stamps"] = np.concatenate(
                            (trial_timestamps, timestamps[src_end_ix:])
                        )

                    samples_polled += chunk_sz
        else:
            null_reads += 1
            if null_reads > self.MAX_NULL_READS:
                raise RuntimeError(
                    f"The stream has not been updated in the last {self.MAX_NULL_READS} read attemps. Please check the stream."
                )
            time.sleep(0.1)
            

        if self._marker_coupled:
            # reset the maker peeked flag since we have polled new data
            self._already_peeked = False

        return trial_data

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
        if not self.active:
            raise RuntimeError("InputLSLStream.last_marker() called on inactive stream. Please call update_input_streams() first to configure the stream object.")

        return self._marker_buffer["time_series"]

    def update_input_streams(
        self,
        pred=None,
        channels=None,
        marker_coupled=True,
        marker_fmt=None,
        marker_pred=None,
        stream_info=None,
        marker_stream_info=None,
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

        """
        if self.active:
            return

        if not stream_info:
            # resolve the stream on the LSL network
            available_streams = pylsl.resolve_bypred(pred)
        else:
            available_streams = [stream_info]

        if len(available_streams) == 0:
            raise RuntimeError("No streams found matching the predicate")
        else:
            warnings.warn("More than one stream found matching the predicate. Using the first stream found.", 
                          RuntimeWarning, stacklevel=2)

        self._data_buffer = {"time_series": None, "time_stamps": None}
        self._data_inlet = pylsl.StreamInlet(
            available_streams[0],
            processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter,
            recover=False,
        )
        self._data_inlet.open_stream()

        if channels:
            if max(channels) >= self._data_inlet.channel_count or min(channels) < 0:
                raise ValueError(
                    "The number of channels in the stream does not match the channel indices specified in the channels parameter. Please check the channels parameter and try again."
                )
            self._channels = channels
        else:
            self._channels = tuple([_ for _ in range(self._data_inlet.channel_count)])

        if marker_coupled:
            if not marker_stream_info:
                # resolve the stream on the LSL network
                marker_streams = pylsl.resolve_bypred(marker_pred)
            else:
                marker_streams = [marker_stream_info]

            if len(marker_streams) == 0:
                raise RuntimeError("No marker streams found matching the predicate")
            else:
                warnings.warn("More than one marker stream found matching the predicate. Using the first stream found.", 
                              RuntimeWarning, stacklevel=2)
                
            self._marker_inlet = pylsl.StreamInlet(marker_streams[0])
            self._peek_marker_inlet = pylsl.StreamInlet(marker_streams[0])

            # open the inlet
            self._marker_inlet.open_stream()
            self._peek_marker_inlet.open_stream()

            if marker_fmt:
                self._marker_pattern = re.compile(marker_fmt)

        self.active = True

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
        active=True,
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

        Examples
        --------
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
        )
        sess.add_ext_src(src)

        return src

    @classmethod
    def create_marker_uncoupled_data_stream(cls, sess, 
                                            pred=None,
                                            channels=None,
                                            relative_start=0,
                                            active=True,
                                            interval=None):
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
            The predicate string, e.g. "name='BioSemi'" or "type='EEG' and starts-with(name, 'BioSemi') and
            count(description/desc/channels/channel)=32"
        channels : tuple or list of ints
            Index value of channels to poll from the stream, if None all channels will be polled
        active : bool
            Flag to indicate whether the stream is active or will be activated in the future
        interval : float
            The minimum interval at which the stream will be polled
        """
        src = cls(sess, pred, channels, relative_start, marker=False, active=active, interval=interval)
        sess.add_ext_src(src)

        return src


class OutputLSLStream(MPBase):
    """
    An object for maintaining an LSL outlet

    Attributes
    ----------

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
        self._stream_info = stream_info
        # resolve the stream on the LSL network
        self.lsl_marker_outlet = pylsl.StreamOutlet(stream_info, chunk_size, max_buffer)
        self.liesl_session = None
        # Start LieSL recording if the user has specified a filesave

        warnings.filterwarnings(
            action="ignore", category=RuntimeWarning, module="subprocess"
        )
        output_save_thread = threading.Thread(target=self._check_status, args=(filesave,))
        output_save_thread.start()

    def _check_status(self, filesave):
        if filesave is not None:
            streamargs = [
                {
                    "name": self._stream_info.name(),
                    "type": self._stream_info.type(),
                    "channel_count": self._stream_info.channel_count(),
                    "nominal_srate": self._stream_info.nominal_srate(),
                    "channel_format": self._stream_info.channel_format(),
                    "source_id": self._stream_info.source_id(),
                }
            ]
            self.liesl_session = liesl.Session(
                mainfolder=f"{os.path.dirname(os.path.realpath(__file__))}\labrecordings",
                streamargs=streamargs,
            )

            with self.liesl_session(filesave):
                while True:
                    time.sleep(0.1)
                    if not threading.main_thread().is_alive():
                        # Suppress output from pyLiesl
                        sys.stdout = open(os.devnull, "w")
                        sys.stderr = open(os.devnull, "w")
                        self.liesl_session.stop_recording()
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

        """

        try:
            self.lsl_marker_outlet.push_sample(data, pylsl.local_clock())

        except (ValueError, TypeError) as ve:
            try:
                self.lsl_marker_outlet.push_sample(data[0], pylsl.local_clock())

            except Exception as e:
                raise type(e)(f"{str(e)}\nPush data - Irreparable Error in LSL Output. No data pushed to output stream").with_traceback(sys.exc_info()[2])

    @classmethod
    def create_outlet_from_streaminfo(cls, sess, stream_info, filesave=None):
        """
        Factory method to create a OutletLSLStream mindpype object from a pylsl.StreamInfo object.

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
        sess.add_ext_out(src)

        return src
