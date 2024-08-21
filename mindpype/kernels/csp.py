from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from .kernel_utils import extract_init_inputs

import numpy as np
import mne


class CommonSpatialPatternKernel(Kernel):
    """
    Kernel to apply common spatial pattern (CSP) filters to trial data. CSP works by 
    finding spatial filters that maximize the variance for one condition while minimizing 
    it for the other, to distinguishing between different mental states.

    .. note::
        This kernel utilizes the mne class :class:`CSP <mne:mne.decoding.CSP>`

    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to
    inA : Tensor
        First input trial data
    outA : Tensor
        Output trial data
    n_components : int, default=4
        Number of components to decompose the input signals.
        See :class:`CSP <mne:mne.decoding.CSP>` for more information.
    cov_est : str, default='concat'
        Method to estimate the covariance matrix. Options are 'concat' or 'epoch'.
        See :class:`CSP <mne:mne.decoding.CSP>` for more information.
    reg : float, default=None
        Regularization parameter for covariance matrix estimation.
        See :class:`CSP <mne:mne.decoding.CSP>` for more information.
    init_data : Tensor or Array, default=None
        Initialization data to configure the filters (n_trials, n_channels, n_samples)
    labels : Tensor or Array, default=None
        Labels corresponding to initialization data class labels (n_trials,)

    See Also
    --------
    Kernel : Base class for all kernel objects
    """

    def __init__(self,graph, inA, outA, n_components=4,
                 cov_est='concat', reg=None,
                 init_data=None, labels=None):
        """ Init """
        super().__init__('CSP', MPEnums.INIT_FROM_DATA, graph)
        self.inputs = [inA]
        self.outputs = [outA]

        if init_data is not None:
            self.init_inputs = [init_data]

        if labels is not None:
            self.init_input_labels = labels

        self._n_components = n_components
        self._mdl = mne.decoding.CSP(n_components=n_components, cov_est=cov_est, 
                                     reg=reg, transform_into='csp_space')
        self._initialized = False


    def _initialize(self, init_inputs, init_outputs, labels):
        """
        Initialize the kernel and compute intialization data output.
        Sets the filter values based on the provided initialization data.

        Parameters
        ----------
        init_inputs : list of Tensors or Arrays 
            Initialization input data container, list of length 1
        init_outputs : list of Tensors or Arrays
            Initialization output data container, list of length 1
        labels : Tensor or Array
            Labels corresponding to initialization data class labels (n_trials,)
        """

        # check that the input init data is in the correct type
        init_in = init_inputs[0]
        accepted_inputs = (MPEnums.TENSOR, MPEnums.ARRAY, MPEnums.CIRCLE_BUFFER)

        for init_obj in (init_in,labels):
            if init_obj.mp_type not in accepted_inputs:
                raise TypeError('Initialization data must be a tensor, array, or circle buffer')

        if self.init_style == MPEnums.INIT_FROM_DATA:
            # extract the initialization data
            X = extract_init_inputs(init_in)
            y = extract_init_inputs(labels)
            self._initialized = False
            old_log_level = mne.set_log_level('WARNING', return_old_level=True)  # suppress CSP calculation output
            self._mdl.fit(X, y)
            mne.set_log_level(old_log_level)
            self._initialized = True

        # compute init output
        init_out = init_outputs[0]
        if init_out is not None:
            if init_in.mp_type != MPEnums.TENSOR:
                init_in = init_in.to_tensor()

            # adjust the shape of init output tensor
            if len(init_in.shape) == 3:
                init_out.shape = (init_in.shape[0], self._n_components, init_in.shape[2])

            self._process_data(init_inputs, init_outputs)


    def _process_data(self, inputs, outputs):
        """
        Apply the CSP filters to the input data and
        assign the filtered data to the output tensor

        Parameters
        ----------
        inputs : list of Tensors
            Input data container, list of length 1
        outputs : list of Tensors
            Output data container, list of length 1
        """
        if len(inputs[0].data.shape) == 2:
            d_in = np.expand_dims(inputs[0].data, axis=0)
        else:
            d_in = inputs[0].data
        
        outputs[0].data = self._mdl.transform(d_in)
    
    def _verify(self):
        """
        Verify the kernel parameters
        """

        # first ensure the input and output are tensors
        d_in = self.inputs[0]
        d_out = self.outputs[0]

        for param in (d_in, d_out):
            if param.mp_type != MPEnums.TENSOR:
                raise TypeError('Input and output parameters must be tensors')

        # input tensor should be two- or three-dimensional
        if len(d_in.shape) != 2 and len(d_in.shape) != 3:
            raise ValueError('Input tensor must be two- or three-dimensional')

        if len(d_in.shape) == 2:
            out_sz = (self._n_components, d_in.shape[1])
        else:
            out_sz =  (d_in.shape[0], self._n_components, d_in.shape[2])

        if d_out.virtual and len(d_out.shape) == 0:
            d_out.shape = out_sz

        # verify that output tensor can accept data of this size
        d_out.data = np.zeros(out_sz)


    @classmethod
    def add_to_graph(cls, graph, inA, outA,
                     initialization_data=None, labels=None,
                     n_components=4, cov_est='concat', reg=None):
        """
        Factory method to create a CSP filter kernel and 
        add it as a node to a graph

        Parameters
        ----------
        graph : Graph
            Graph that the kernel should be added to

        inA : Tensor
            Input trial data
        outA : Tensor
            Filtered trial data
        initialization_data : Tensor or Array, default=None
            Initialization data to configure the filters (n_trials, n_channels, n_samples)
        labels : Tensor or Array, default=None
            Labels corresponding to initialization data class labels (n_trials,)
        n_components : int, default=4
            Number of components to decompose the input signals.
            See :class:`CSP <mne:mne.decoding.CSP>` for more information.
        cov_est : str, default='concat'
            Method to estimate the covariance matrix. Options are 'concat' or 'epoch'.
            See :class:`CSP <mne:mne.decoding.CSP>` for more information.
        reg : float, default=None
            Regularization parameter for covariance matrix estimation.
            See :class:`CSP <mne:mne.decoding.CSP>` for more information.
        """

        # create the kernel object
        k = cls(graph, inA, outA,
                n_components=n_components, 
                cov_est=cov_est, reg=reg,
                init_data=initialization_data,
                labels=labels)

        # create parameter objects for the input and output
        params = (Parameter(inA, MPEnums.INPUT),
                  Parameter(outA, MPEnums.OUTPUT))

        # add the kernel to a generic node object
        node = Node(graph, k, params)

        # add the node to the graph
        graph.add_node(node)

        return node
