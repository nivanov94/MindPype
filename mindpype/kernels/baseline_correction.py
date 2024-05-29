from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter

import numpy as np


class BaselineCorrectionKernel(Kernel):
    """
    Kernel to perform baseline correction.
    Input data is baseline corrected by subtracting
    the mean of the baseline period from the input data
    along a specified axis.

    .. note::
        This kernel utilizes the numpy functions
        :func:`asarray <numpy:numpy.asarray>`,
        :func:`array <numpy:numpy.array>`,
        :func:`mean <numpy:numpy.mean>`.

    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to
    inA : MindPype Tensor object
        Input data container
    outA : MindPype Tensor object
        Output data container
    axis : int, default = -1
        Axis along which to perform baseline correction
    baseline_period : array-like (start, end)
        Baseline period where start and end are the start
        and end indices of the baseline period within the
        target axis.
    """

    def __init__(self, graph, inA, outA, axis=-1, baseline_period=(0,-1)):
        """ Init """
        super().__init__('BaselineCorrection', MPEnums.INIT_FROM_NONE, graph)
        self.inputs = [inA]
        self.outputs = [outA]
        self._baseline_period = np.asarray(baseline_period)
        self._axis = axis

    def _verify(self):
        """
        Verify the kernel parameters
        """

        d_in = self.inputs[0]
        d_out = self.outputs[0]

        # input and output must be a tensor
        if (d_in.mp_type != MPEnums.TENSOR or
            d_out.mp_type != MPEnums.TENSOR):
            raise TypeError("Input and output must be a tensor")

        # check the baseline period is valid
        if self._baseline_period.shape[0] != 2:
            raise ValueError("Baseline period must include start and end points")

        # if the axis or baseline indices are negative, convert them to positive
        if self._axis < 0:
            self._axis = self._axis + len(d_in.shape)

        for i, idx in enumerate(self._baseline_period):
            if idx < 0:
                self._baseline_period[i] = d_in.shape[self._axis] + idx

        # force the baseline period to be an integer
        self._baseline_period = np.array(self._baseline_period, dtype=int)

        # start index must be less than end index and both must be
        # within the range of the data
        if ((self._baseline_period[0] > self._baseline_period[1]) or
            (self._baseline_period[0] < 0) or
            (self._baseline_period[1] > d_in.shape[-1])):
            raise ValueError("Baseline period must be within the range of the data")

        # check the output dimensions are valid
        if d_out.virtual and len(d_out.shape) == 0:
            d_out.shape = d_in.shape

    def _initialize(self, init_inputs, init_outputs, labels=None):
        """
        Initialize the kernel and compute initialization data output

        Parameters
        ----------
        init_inputs : list of MindPype Tensor or Array data containers
            Initialization input data container, list of length 1
        init_outputs : list of MindPype Tensor or Array data containers
            Initialization output data container, list of length 1
        labels : Tensor 
            Class labels for initialization data
        """

        init_in = init_inputs[0]
        init_out = init_outputs[0]

        if init_out is not None and (init_in is not None and init_in.shape != ()):
            if init_in.mp_type != MPEnums.TENSOR:
                init_in = init_in.to_tensor()

            # Set the initialization output shape
            if init_out.virtual:
                output_shape = list(init_in.shape)
                init_out.shape = tuple(output_shape)

            self._process_data([init_in], init_outputs)

    def _process_data(self, inputs, outputs):
        """
        Perform baseline correction on the input data and
        store the result in the output data. Compute the
        baseline along the specified axis between the start
        and end indices. Then subtract the mean of the baseline
        from the input data.

        Parameters
        ----------
        inputs : list of MindPype Tensor or Array data containers
            Input data container, list of length 1
        outputs : list of MindPype Tensor or Array data containers
            Output data container, list of length 1
        """
        inA = inputs[0]
        outA = outputs[0]

        # compute baseline
        baseline = np.mean(inA.data[..., self._baseline_period[0]:self._baseline_period[1]],
                           axis=self._axis,
                           keepdims=True)

        # remove baseline and assign to output
        outA.data = inA.data - baseline

    @classmethod
    def add_to_graph(cls, graph, inputA, outputA,
                     baseline_period=(0,-1), axis=-1,
                     init_input=None, init_labels=None):
        """
        Factory method to create a baseline correction kernel
        and add it to a graph

        Parameters
        ----------
        graph : Graph
            Graph that the kernel should be added to
        inputA : MindPype Tensor
            Input data container
        outputA : MindPype Tensor
            Output data container
        baseline_period : array-like (start, end)
            Baseline period where start and end are the start
            and end indices of the baseline period within the
            target axis.
        axis : int, default = -1
            Axis along which to perform baseline correction
        init_input : MindPype Tensor or Scalar data container, default=None
            MindPype data container with initialization data to be
            transformed and passed to downstream nodes during graph
            initialization
        init_labels : MindPype Tensor or Array data container, default=None
            MindPype data container with initialization labels to be
            passed to downstream nodes during graph initialization

        Returns
        -------
        node : Node
            Node object containing the baseline correction kernel and parameters
        """

        # create the kernel object
        k = cls(graph, inputA, outputA, axis, baseline_period)

        # create parameter objects for the input and output
        params = (Parameter(inputA, MPEnums.INPUT),
                  Parameter(outputA, MPEnums.OUTPUT))

        # add the kernel to a generic node object
        node = Node(graph, k, params)

        # add the node to the graph
        graph.add_node(node)

        # if initialization data is provided, then add it to the node
        if init_input is not None:
            node.add_initialization_data([init_input], init_labels)

        return node
