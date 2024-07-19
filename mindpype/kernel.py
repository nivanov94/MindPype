from .core import MPBase, MPEnums
from abc import ABC
from .containers import Tensor
import sys
import numpy as np


class Kernel(MPBase, ABC):
    """
    An abstract base class for kernels. Only used by developers loooking to extend the library.

    Parameters
    ----------
    name : str
        Name of the kernel
    init_style : MPEnums Object
        Kernel initialization style, according to MPEnum class

    Attributes
    ----------
    name : str
        Name of the kernel
    init_style : MPEnums Object
        Kernel initialization style, according to BcipEnum class
    """

    def __init__(self, name, init_style, graph):
        """
        Constructor for the Kernel class
        """
        session = graph.session
        super().__init__(MPEnums.KERNEL, session)
        self.name = name
        self.init_style = init_style
        self._num_classes = None

        self._inputs = []
        self._outputs = []

        self._init_inputs = []
        self._init_outputs = []
        self._init_input_labels = None
        self._init_output_labels = None

        # phony inputs and outputs for verification
        self._phony_inputs = {}
        self._phony_outputs = {}
        self._phony_init_inputs = {}
        self._phony_init_outputs = {}
        self._phony_init_input_labels = None
        self._phony_init_output_labels = None

        # tuple to define which inputs must be covariance matrices
        self._covariance_inputs = None

    # API Getters
    @property
    def init_input_labels(self):
        """
        Returns the labels for the initialization inputs

        Returns
        -------
        _init_input_labels : list
            The labels for the initialization inputs
        """

        return self._init_input_labels

    @property
    def init_output_labels(self):
        """
        Returns the labels for the initialization outputs

        Returns
        -------
        _init_output_labels : list
            The labels for the initialization outputs
        """

        return self._init_output_labels

    @property
    def phony_init_input_labels(self):
        """
        Returns the labels for the phony initialization inputs

        Returns
        -------
        _phony_init_input_labels : list
            The labels for the initialization inputs
        """

        return self._phony_init_input_labels

    @property
    def phony_init_output_labels(self):
        """
        Returns the labels for the phony initialization outputs

        Returns
        -------
        _phony_init_output_labels : list
            The labels for the initialization outputs
        """

        return self._phony_init_output_labels

    @property
    def inputs(self):
        """
        Returns the inputs of the kernel

        Returns
        -------
        _inputs : list
            The inputs of the kernel
        """

        return self._inputs

    @property
    def outputs(self):
        """
        Returns the outputs of the kernel

        Returns
        -------
        _outputs : list
            The outputs of the kernel
        """

        return self._outputs

    @property
    def init_inputs(self):
        """
        Returns the inputs of the kernel

        Returns
        -------
        _init_inputs : list
            The inputs of the kernel
        """

        return self._init_inputs

    @property
    def init_outputs(self):
        """
        Returns the outputs of the kernel

        Returns
        -------
        _init_outputs : list
            The outputs of the kernel
        """

        return self._init_outputs

    @property
    def phony_inputs(self):
        """
        Returns the phony inputs of the kernel

        Returns
        -------
        _phony_inputs : list
            The phony inputs of the kernel
        """

        return self._phony_inputs

    @property
    def phony_outputs(self):
        """
        Returns the phony outputs of the kernel

        Returns
        -------
        _phony_outputs : list
            The phony outputs of the kernel
        """

        return self._phony_outputs

    @property
    def phony_init_inputs(self):
        """
        Returns the phony inputs of the kernel

        Returns
        -------
        _phony_init_inputs : list
            The phony inputs of the kernel
        """

        return self._phony_init_inputs

    @property
    def phony_init_outputs(self):
        """
        Returns the phony outputs of the kernel

        Returns
        -------
        _phony_init_outputs : list
            The phony outputs of the kernel
        """

        return self._phony_init_outputs

    @init_input_labels.setter
    def init_input_labels(self, labels):
        """
        Sets the labels for the initialization inputs

        Parameters
        ----------
        labels : list
            The list of labels to be set
        """
        self._init_input_labels = labels

    @init_output_labels.setter
    def init_output_labels(self, labels):
        """
        Sets the labels for the initialization outputs

        Parameters
        ----------
        labels : list
            The list of labels to be set
        """
        self._init_output_labels = labels

    @inputs.setter
    def inputs(self, inputs):
        """
        Sets the inputs of the kernel

        Parameters
        ----------
        inputs : list
            The list of inputs to be set
        """
        self._inputs = inputs

        # update init_inputs so the length matches inputs
        if len(self._init_inputs) != len(self._inputs):
            if len(self._init_inputs) == 0:
                self._init_inputs = [None] * len(self._inputs)
            else:
                self._init_inputs.extend(
                    [None] * (len(self._inputs) - len(self._init_inputs)))

    @outputs.setter
    def outputs(self, outputs):
        """
        Sets the outputs of the kernel

        Parameters
        ----------
        outputs : list
            The list of outputs to be set
        """
        self._outputs = outputs

        # update init_outputs so the length matches outputs
        if len(self._init_outputs) != len(self._outputs):
            if len(self._init_outputs) == 0:
                self._init_outputs = [None] * len(self._outputs)
            else:
                self._init_outputs.extend(
                    [None] * (len(self._outputs) - len(self._init_outputs)))

    @init_inputs.setter
    def init_inputs(self, inputs):
        """
        Sets the initialization inputs of the kernel

        Parameters
        ----------
        inputs : list
            The list of inputs to be set
        """
        self._init_inputs = inputs

    @init_outputs.setter
    def init_outputs(self, outputs):
        """
        Sets the initialization outputs of the kernel

        Parameters
        ----------
        outputs : list
            The list of outputs to be set
        """
        self._init_outputs = outputs

    @phony_init_input_labels.setter
    def phony_init_input_labels(self, labels):
        """
        Sets the labels for the phony initialization inputs

        Parameters
        ----------
        labels : list
            The list of labels to be set
        """
        self._phony_init_input_labels = labels

    # INPUT and OUTPUT getter methods
    def get_input(self, index):
        """
        Returns the input at the specified index

        Parameters
        ----------
        index : int
            The index of the input to be returned

        Returns
        -------
        _inputs[index] : Object
            The input at the specified index
        """
        return self._inputs[index]

    def get_output(self, index):
        """
        Returns the output at the specified index

        Parameters
        ----------
        index : int
            The index of the output to be returned

        Returns
        -------
        _outputs[index] : Object
            The output at the specified index
        """
        return self._outputs[index]

    def get_init_input(self, index):
        """
        Returns the input at the specified index

        Parameters
        ----------
        index : int
            The index of the input to be returned

        Returns
        -------
        _init_inputs[index] : Object
            The input at the specified index
        """
        return self._init_inputs[index]

    def get_init_output(self, index):
        """
        Returns the output at the specified index

        Parameters
        ----------
        index : int
            The index of the output to be returned

        Returns
        -------
        _init_outputs[index] : Object
            The output at the specified index
        """
        return self._init_outputs[index]

    def add_phony_input(self, ph_input, index):
        """
        Adds a phony input to the kernel

        Parameters
        ----------
        ph_input : Object
            The input to be added
        index : int
            The index at which the input is to be added
        """
        self._phony_inputs[index] = ph_input

    def add_phony_output(self, ph_output, index):
        """
        Adds a phony output to the kernel

        Parameters
        ----------
        ph_output : Object
            The output to be added
        index : int
            The index at which the output is to be added
        """
        self._phony_outputs[index] = ph_output

    def add_phony_init_input(self, ph_input, index):
        """
        Adds a phony initialization input to the kernel

        Parameters
        ----------
        ph_input : Object
            The input to be added
        index : int
            The index at which the input is to be added
        """
        self._phony_init_inputs[index] = ph_input

    def add_phony_init_output(self, ph_output, index):
        """
        Adds a phony initialization output to the kernel

        Parameters
        ----------
        ph_output : Object
            The output to be added
        index : int
            The index at which the output is to be added
        """
        self._phony_init_outputs[index] = ph_output

    def _copy_init_labels_to_output(self, verification=False):
        """
        Copies the input labels from initialization to the output

        Parameters
        ----------
        verification: Bool
        """
        if not verification or self.phony_init_input_labels is None:
            src = self.init_input_labels
        else:
            src = self.phony_init_input_labels

        if not verification or self.phony_init_output_labels is None:
            dst = self.init_output_labels
        else:
            dst = self.phony_init_output_labels

        if src is None:  # nothing to copy
            return

        if src.mp_type != MPEnums.TENSOR:
            src = src.to_tensor()

        if dst is None:
            dst = Tensor.create_virtual(self.session)

        src.copy_to(dst)

    def update_parameters(self, parameter, value):
        """
        Update the kernel parameters

        Parameters
        ----------
        kwargs : dict
            Dictionary of parameters to be updated
        """
        try:
            setattr(self, str(parameter), value)

        except AttributeError:
            raise AttributeError(("Kernel {} does not " +
                                  "have parameter {}").format(self.name,
                                                              parameter))

    def verify(self):
        """
        Basic verification method that can be applied to all kernels.
        Generates phony inputs and attemps to execute the kernel.
        """
        print(f"Verifying kernel {self.name}...")
        if hasattr(self, '_verify'):
            # execute any kernel-specific verification
            self._verify()

        # extract verification inputs and outputs
        verif_inputs = []
        verif_outputs = []
        verif_io = (verif_inputs, verif_outputs)
        io = (self.inputs, self.outputs)
        phony_io = (self.phony_inputs, self.phony_outputs)

        for params, verif_params, phony_params in zip(io, verif_io, phony_io):
            for i_p, param in enumerate(params):
                if i_p in phony_params:
                    # use phony params if available
                    verif_params.append(phony_params[i_p])
                else:
                    # otherwise use real virtual params
                    verif_params.append(param)

        # if the kernel requires initialization, extract phony init
        # inputs and outputs
        if self._verif_inits_exist():
            verif_init_inputs = []
            verif_init_outputs = []
            verif_init_io = (verif_init_inputs, verif_init_outputs)
            init_io = (self.init_inputs, self.init_outputs)
            phony_init_io = (self.phony_init_inputs, self.phony_init_outputs)

            for params, verif_params, phony_params in zip(init_io,
                                                          verif_init_io,
                                                          phony_init_io):
                for i_p, param in enumerate(params):
                    if i_p in phony_params:
                        # use phony params if available
                        verif_params.append(phony_params[i_p])
                    else:
                        # otherwise use real virtual params
                        verif_params.append(param)

            if self.phony_init_input_labels is not None:
                verif_init_labels = self.phony_init_input_labels
            else:
                verif_init_labels = self.init_input_labels

            # adjust labels if needed
            if self._num_classes is not None:
                if verif_init_labels.mp_type != MPEnums.TENSOR:
                    verif_init_labels = verif_init_labels.to_tensor()

                if np.unique(verif_init_labels.data).shape[0] != self._num_classes:
                    verif_init_labels.assign_random_data(whole_numbers=True,
                                                         vmin=0,
                                                         vmax=(self._num_classes-1))

            # attempt initialization
            try:
                self._initialize(verif_init_inputs,
                                 verif_init_outputs,
                                 verif_init_labels)
                self._copy_init_labels_to_output(verification=True)
            except Exception as e:
                raise type(e)((f"{str(e)}\nTest initialization of " +
                               f"node {self.name} failed " +
                               "during verification. Please check" +
                               "parameters.")).with_traceback(
                                                    sys.exc_info()[2])

            # set init output shapes using phony init outputs as needed
            for init_output, verif_init_output in zip(self.init_outputs,
                                                      verif_init_outputs):
                if (init_output is not None and
                        init_output.shape != verif_init_output.shape):
                    raise ValueError("Test initialization of node " +
                                     f"{self.name} failed during " +
                                     "verification. Please check parameters.")

        # attempt kernel execution
        try:
            self._process_data(verif_inputs, verif_outputs)
        except Exception as e:
            raise type(e)((f"{str(e)}\nTest execution of node {self.name} " +
                           "failed during verification. Please check " +
                           "parameters.")).with_traceback(sys.exc_info()[2])

        # set output shapes using phony outputs as needed
        for output, verif_output in zip(self.outputs, verif_outputs):
            if (output is not None and
                    output.mp_type == MPEnums.TENSOR and
                    output.shape != verif_output.shape):
                raise ValueError(f"Test execution of node {self.name} " +
                                 "failed during verification. Output " +
                                 "shape does not match expected value." +
                                 "Please check parameters.")

    def initialize(self):
        """
        Initialize kernel
        """
        if hasattr(self, '_initialize'):
            self._initialized = False
            self._initialize(self.init_inputs,
                             self.init_outputs,
                             self.init_input_labels)
            self._initialized = True
            self._copy_init_labels_to_output()


    def update(self):
        """
        Update kernel
        """
        if hasattr(self, '_update'):
            self._initialized = False
            self.update(self.init_inputs,
                         self.init_outputs,
                         self.init_input_labels)
            self._initialized = True
            self._copy_init_labels_to_output()
            

    def execute(self):
        """
        Execute kernel to process data
        """
        self._process_data(self.inputs, self.outputs)

    def _verif_inits_exist(self):
        """
        Returns true if the kernel has initialization inputs for verification
        """
        exist = True
        for i_i, init_input in enumerate(self.init_inputs):
            if (i_i not in self.phony_init_inputs and
                (init_input is None or
                 init_input.shape == ())):
                exist = False
                break

        return exist

    def is_covariance_input(self, param):
        """
        Returns true if the parameter is a covariance input

        Parameters
        ----------
        param: data object
            data object to be check if is a covariance matrix

        Returns
        -------
        bool: True if the data object is a covariance matrix, False otherwise
        """
        if self._covariance_inputs is None:
            return False  # kernel does not have covariance inputs

        # search for the parameter in the inputs and init_inputs
        param_index = None
        input_groups = (self.inputs, self.init_inputs)
        i_ig = 0
        while param_index is None and i_ig < len(input_groups):
            input_group = input_groups[i_ig]
            for i_p, candidate_param in enumerate(input_group):
                if candidate_param.session_id == param.session_id:
                    param_index = i_p
                    break
            i_ig += 1

        if param_index is None:
            # search for the parameter in the phony inputs and
            # phony init_inputs
            input_groups = (self.phony_inputs, self.phony_init_inputs)
            i_ig = 0
            while param_index is None and i_ig < len(input_groups):
                input_group = input_groups[i_ig]
                for i_p in input_group:
                    if input_group[i_p].session_id == param.session_id:
                        param_index = i_p
                        break
                i_ig += 1

        return (param_index in self._covariance_inputs)

    def add_initialization_data(self, init_data, init_labels=None):
        """
        Add initialization data to the kernel

        Parameters
        ----------
        init_data : list or tuple of data objects
            MindPype container containing the initialization data
        init_labels : data object containing initialization
        labels, default = None
            MindPype container containing the initialization labels

        """
        if init_labels is not None:
            self.init_input_labels = init_labels

        self.init_inputs = list(init_data)

    def remove_initialization_data(self):
        """
        Remove initialization data from the kernel
        """
        self.init_inputs = []
        self.init_input_labels = None
