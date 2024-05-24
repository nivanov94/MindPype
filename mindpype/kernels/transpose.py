from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter

import numpy as np

class TransposeKernel(Kernel):
    """
    Kernel to compute the tensor transpose

    .. note::
        This kernel utilizes the numpy function
        `transpose <numpy:numpy.transpose>`.

    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to

    inputA : Tensor
        Input data container

    outputA : Tensor
        Output data container

    axes : tuple or list of ints
        Specifies the axes that will be transposed. See 
        :func:`numpy.transpose <numpy:numpy.transpose>`.
    """

    def __init__(self, graph, inputA, outputA, axes):
        """Init"""
        super().__init__('Transpose', MPEnums.INIT_FROM_NONE, graph)
        self.inputs = [inputA]
        self.outputs = [outputA]
        self._axes = axes

    def _compute_output_shape(self, inA, axes):
        """
        Determine the shape of the transposed tensor

        Parameters
        ----------
        inA: Tensor
            Input data

        axes : tuple or list of ints
            Specifies the axes that will be transposed. See 
            :func:`numpy.transpose <numpy:numpy.transpose>`.

        """
        # check the shape
        input_shape = inA.shape
        input_rank = len(input_shape)

        # determine what the output shape should be
        if input_rank == 0:
            return ()

        if axes == None:
            output_shape = reversed(input_shape)
        else:
            output_shape = input_shape[self._axes]

        return tuple(output_shape)

    def _initialize(self, init_inputs, init_outputs, labels):
        """
        Initialize the kernel and compute initialization data output.

        Parameters
        ----------
        init_inputs : list of Tensor or Array
            Initialization input data container, list of length 1
        init_outputs : list of Tensor or Array
            Initialization output data container, list of length 1
        labels : None
            Not used, here for compatibility with other kernels
        """
        init_in = init_inputs[0]
        init_out = init_outputs[0]

        # if there is initialization data, process it
        if init_out is not None and (init_in is not None and init_in.shape != ()):
            # extract input data
            if init_in.mp_type != MPEnums.TENSOR:
                init_in = init_in.to_tensor()

            # set the initialization output shape, as needed
            if init_out.virtual:
                if self._axes == None:
                    init_axes = [_ for _ in range(len(init_in.shape))]
                    init_axes[-2:] = [init_axes[-1], init_axes[-2]]
                elif len(init_in.shape)+1 == len(self._axes):
                    init_axes = [0] + [a+1 for a in self._axes]
                else:
                    init_axes = self._axes

                init_out.shape = self._compute_output_shape(init_in, init_axes)

            # process the initialization data
            self._process_data([init_in], init_outputs)

    def _process_data(self, inputs, outputs):
        """
        Compute the transpose of the input data and store
        the result in the output data.

        Parameters
        ----------
        inputs: Tensor or Scalar
            Input trial data 
        outputs: Tensor or Scalar 
            Output trial data
        """
        outputs[0].data = np.transpose(inputs[0].data,axes=self._axes)

    @classmethod
    def add_to_graph(cls,graph,inputA,outputA,axes=None,init_input=None,init_labels=None):
        """
        Factory method to create a transpose kernel and add it to a graph.

        Parameters
        ----------
        graph : Graph
            Graph that the kernel should be added to
        inputA : Tensor or Scalar
            Input data
        outputA : Tensor or Scalar
            Output data
        axes : tuple or list of ints
            Specifies the axes that will be transposed. See
            :func:`numpy.transpose <numpy:numpy.transpose>`.
        
        Returns
        -------
        node : Node
            Node object that was added to the graph containing the kernel
        """

        # create the kernel object
        k = cls(graph, inputA, outputA, axes)

        # create parameter objects for the input and output
        params = (Parameter(inputA, MPEnums.INPUT),
                  Parameter(outputA, MPEnums.OUTPUT))

        # add the kernel to a generic node object
        node = Node(graph, k, params)

        # add the node to the graph
        graph.add_node(node)

        # if initialization data is provided, add it to the node
        if init_input is not None:
            node.add_initialization_data([init_input], init_labels)

        return node
