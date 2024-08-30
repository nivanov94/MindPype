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
    def num_inputs(self):
        """
        Returns the number of inputs to the kernel

        Returns
        -------
        num_inputs : int
            The number of inputs to the kernel
        """
        return len(self._inputs)
    
    @property
    def num_outputs(self):
        """
        Returns the number of outputs from the kernel

        Returns
        -------
        num_outputs : int
            The number of outputs from the kernel
        """
        return len(self._outputs)

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

        # update _init_inputs so the length matches inputs
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

        # update _init_outputs so the length matches outputs
        if len(self._init_outputs) != len(self._outputs):
            if len(self._init_outputs) == 0:
                self._init_outputs = [None] * len(self._outputs)
            else:
                self._init_outputs.extend(
                    [None] * (len(self._outputs) - len(self._init_outputs)))

    def set_parameter(self, param, index, ptype, direction, add_if_missing=False):
        """
        Set a parameter to the kernel

        Parameters
        ----------
        param : MindPype Container
            MindPype container to be added as a kernel parameter
        index : int
            The index at which the parameter is to be added. This should
            correspond to the index of the input or output to which the
            parameter is associated.
        ptype : str
            The type of parameter, either 'main', 'init', 'verif', 
            'verif_init', 'labels' or 'verif_labels',. This defines
            when and how the parameter is used. 'main' parameters are 
            used during
            normal execution, 'init' parameters are used during 
            initialization, 'verif' and 'verif_init' parameters 
            are used during verification. 'labels' parameters are used
            to store labels for initialization. 'verif_labels' are used
            to store labels for verification.
        direction : MPEnums Object
            The direction of the parameter, according to the MPEnums class
        add_if_missing : bool
            If True, the parameter will be added to the kernel if it does not
            already exist. If False, the parameter will not be added if it does
            not exist. Only applies to verification parameters.
        """


        if 'labels' in ptype:
            if ptype == 'labels':
                if direction == MPEnums.INPUT:
                    self._init_input_labels = param
                else:
                    self._init_output_labels = param
            else:
                if direction == MPEnums.INPUT:
                    self._phony_init_input_labels = param
                else:
                    self._phony_init_output_labels = param
        else: 
            dest = self._get_param_container(ptype, direction)
            if 'verif' not in ptype or add_if_missing or index in dest:
                dest[index] = param

    def get_parameter(self, index, ptype, direction):
        """
        Get a parameter from the kernel

        Parameters
        ----------
        index : int
            The index of the parameter to be returned
        ptype : str
            The type of parameter, either 'main', 'init', 'verif',
            'verif_init', 'labels', or 'verif_labels'. This defines
            when and how 
            the parameter is used. 'main' parameters are used during 
            normal execution, 'init' parameters are used during 
            initialization, 'verif' and 'verif_init' parameters are 
            used during verification. 'labels' parameters are used
            to store labels for initialization. 'verif_labels' are used
            to store labels for verification.
        direction : MPEnums Object
            The direction of the parameter, according to the MPEnums class

        Returns
        -------
        src[index] : MindPype Container
            The parameter at the specified index
        """
        src = self._get_param_container(ptype, direction)
        
        if 'verif' in ptype and index not in src:
            return None
        elif 'labels' in ptype:
            return src 
        else:
            return src[index]
        
    def _get_param_container(self, ptype, direction):
        """
        Find the appropriate container for the parameter
        """
        if ptype == 'main':
            if direction == MPEnums.INPUT:
                c = self._inputs
            else:
                c = self._outputs
        elif ptype == 'init':
            if direction == MPEnums.INPUT:
                c = self._init_inputs
            else:
                c = self._init_outputs
        elif ptype == 'verif':
            if direction == MPEnums.INPUT:
                c = self._phony_inputs
            else:
                c = self._phony_outputs
        elif ptype == 'verif_init':
            if direction == MPEnums.INPUT:
                c = self._phony_init_inputs
            else:
                c = self._phony_init_outputs
        elif ptype == 'labels':
            if direction == MPEnums.INPUT:
                c = self._init_input_labels
            else:
                c = self._init_output_labels
        elif ptype == 'verif_labels':
            if direction == MPEnums.INPUT:
                c = self._phony_init_input_labels
            else:
                c = self._phony_init_output_labels
        else:
            raise ValueError("Invalid parameter type. Must be 'main', " +
                             "'init', 'verif', 'verif_init', 'labels'," +
                             "or 'verif_labels'.")
        
        return c

    def _copy_init_labels_to_output(self, verification=False):
        """
        Copies the input labels from initialization to the output

        Parameters
        ----------
        verification: Bool
        """
        if not verification or self._phony_init_input_labels is None:
            src = self.get_parameter(0, 'labels', MPEnums.INPUT)
        else:
            src = self.get_parameter(0, 'verif_labels', MPEnums.INPUT)

        if not verification or self._phony_init_output_labels is None:
            dst = self.get_parameter(0, 'labels', MPEnums.OUTPUT)
        else:
            dst = self.get_parameter(0, 'verif_labels', MPEnums.OUTPUT)

        if src is None:  # nothing to copy
            return

        if src.mp_type != MPEnums.TENSOR:
            src = src.to_tensor()

        if dst is None:
            dst = Tensor.create_virtual(self.session)

        src.copy_to(dst)

    def find_param_index(self, target_param, dir):
        """
        Find the index of a main input or output parameter

        Parameters
        ----------
        target_param : MindPype Container
            The parameter to be found
        dir : MPEnums Object
            The direction of the parameter, according to the MPEnums class

        Returns
        -------
        index : int
            The index of the parameter
        """
        if dir == MPEnums.INPUT:
            param_lst = self._inputs
        else:
            param_lst = self._outputs

        for i, param in enumerate(param_lst):
            if (param is not None and 
                    param.session_id == target_param.session_id):
                return i

        return None

    def update_attribute(self, parameter, value):
        """
        Update the kernel attribute

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
        phony_io = (self._phony_inputs, self._phony_outputs)

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
            init_io = (self._init_inputs, self._init_outputs)
            phony_init_io = (self._phony_init_inputs, self._phony_init_outputs)

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

            if self._phony_init_input_labels is not None:
                verif_init_labels = self._phony_init_input_labels
            else:
                verif_init_labels = self._init_input_labels

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
                additional_msg = ("Test initialization of node " +
                                  f"{self.name} failed during verification. " +
                                  "See trace for further details.")
                if sys.version_info[:2] >= (3, 11):
                    e.add_note(additional_msg)
                else:
                    pretty_msg = f"{'*'*len(additional_msg)}\n{additional_msg}\n{'*'*len(additional_msg)}\n"
                    print(pretty_msg)
                raise

            # set init output shapes using phony init outputs as needed
            for init_output, verif_init_output in zip(self._init_outputs,
                                                      verif_init_outputs):
                if (init_output is not None and
                        init_output.shape != verif_init_output.shape):
                    raise ValueError("Test initialization of node " +
                                     f"{self.name} failed during " +
                                     "verification. Please check parameters.")
                
        elif self.init_style == MPEnums.INIT_FROM_DATA:
            raise ValueError(f"Node {self.name} requires initialization " +
                             "data but none was provided")

        # attempt kernel execution
        try:
            self._process_data(verif_inputs, verif_outputs)
        except Exception as e:
            additional_msg = ("Test execution of node " +
                              f"{self.name} failed during verification. " +
                              "See trace for further details.")
            if sys.version_info[:2] >= (3, 11):
                e.add_note(additional_msg)
            else:
                pretty_msg = f"{'*'*len(additional_msg)}\n{additional_msg}\n{'*'*len(additional_msg)}\n"
                print(pretty_msg)
            raise

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
            self._initialize(self._init_inputs,
                             self._init_outputs,
                             self._init_input_labels)
            self._initialized = True
            self._copy_init_labels_to_output()


    def update(self):
        """
        Update kernel
        """
        if hasattr(self, '_update'):
            self._initialized = False
            self.update(self._init_inputs,
                         self._init_outputs,
                         self._init_input_labels)
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
        for i_i, init_input in enumerate(self._init_inputs):
            if (i_i not in self._phony_init_inputs and
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
        input_groups = (self.inputs, self._init_inputs)
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
            input_groups = (self._phony_inputs, self._phony_init_inputs)
            i_ig = 0
            while param_index is None and i_ig < len(input_groups):
                input_group = input_groups[i_ig]
                for i_p in input_group:
                    if input_group[i_p].session_id == param.session_id:
                        param_index = i_p
                        break
                i_ig += 1

        return (param_index in self._covariance_inputs)

    def add_initialization_data(self, init_inputs, init_labels=None):
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
            self.set_parameter(init_labels, 0, 'labels', MPEnums.INPUT)

        for i, init_input in enumerate(init_inputs):
            self.set_parameter(init_input, i, 'init', MPEnums.INPUT)