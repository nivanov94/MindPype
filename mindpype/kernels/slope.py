from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


class SlopeKernel(Kernel):
    """
    Estimates the slope of a time series

    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to

    inA : Tensor
        Input data

    outA : Tensor
        Output data

    Fs : int
        Sampling frequency of the input data

    axis : int
        Axis along which to compute the slope
    """

    def __init__(self, graph, inA, outA, Fs=1, axis=-1):
        """ Init """
        super().__init__('Slope',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA]
        self.outputs = [outA]
        self._Fs = Fs
        self._axis = axis

    def _initialize(self, init_inputs, init_outputs, labels):
        """
        This kernel has no internal state that must be initialized.
        Call initialization_execution if downstream nodes are missing training
        data

        Parameters
        ----------

        init_inputs: Tensor
            Input data

        init_outputs: Tensor
            Output data

        labels: None
        """

        init_in = init_inputs[0]
        init_out = init_outputs[0]

        if (init_out is not None and
            init_in is not None and
            init_in.shape != ()):
            if init_in is not None and init_in.mp_type != MPEnums.TENSOR:
                init_in = init_in.to_tensor()

            # adjust the axis if needed
            axis_adjusted = False
            if len(init_in.shape) == len(self.inputs[0].shape)+1 and self._axis >= 0:
                self._axis += 1
                axis_adjusted = True

            # update output size, as needed
            if init_out.virtual:
                output_sz = [d for i_d, d in enumerate(init_in.shape) if i_d != self._axis]
                output_sz.append(1)
                init_out.shape = tuple(output_sz)

            self._process_data([init_in], init_outputs)

            # reset the axis, if needed
            if axis_adjusted:
                self._axis -= 1

    def _verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """

        inA = self.inputs[0]
        outA = self.outputs[0]

        if inA.mp_type != MPEnums.TENSOR:
            raise TypeError("Input must be a tensor")

        if outA.mp_type != MPEnums.TENSOR:
            raise TypeError("Output must be a tensor")

        if self._Fs <= 0:
            raise ValueError("Sampling frequency must be greater than 0")
        
        if self._axis >= len(inA.shape) or self._axis < -len(inA.shape):
            raise ValueError("Axis out of range")

        if self._axis < 0:
            self._axis += len(inA.shape)

        out_shape = [d for i_d, d in enumerate(inA.shape) if i_d != self._axis]
        out_shape.append(1)
        out_shape = tuple(out_shape)
        if outA.virtual and len(outA.shape) == 0:
            outA.shape = out_shape

        if outA.shape != out_shape:
            raise ValueError(f"Output shape must be {out_shape}")

    def _process_data(self, inputs, outputs):
        """
        Estimate slope of time series.

        Parameters
        ----------

        inputs: list of Tensors
            Input data container, list of length 1

        outputs: list of Tensors
            Output data container, list of length 1

        """

        inA = inputs[0]
        outA = outputs[0]

        Ns = inA.shape[self._axis]
        X = np.linspace(0, Ns/self._Fs, Ns)

        # move the axis of interest to the end
        Y = np.moveaxis(inA.data, self._axis, -1)

        X -= np.mean(X)
        Y = Y - np.mean(Y, axis=-1, keepdims=True)

        outA.data = np.expand_dims(Y.dot(X) / X.dot(X), axis=-1)

    @classmethod
    def add_to_graph(cls, graph, inA, outA, Fs=1, axis=-1,
                     init_input=None, init_labels=None):
        """
        Factory method to create a slope estimation kernel

        Parameters
        ----------
        graph : Graph
            Graph that the kernel should be added to

        inA : Tensor
            Input data

        outA : Tensor
            Output data

        Fs : int
            Sampling frequency of the input data

        axis : int
            Axis along which to compute the slope

        init_inputs : Tensor or Array, default = None
            Initialization data for the graph

        init_labels : Tensor or Array, default = None
            Initialization labels for the graph
        """

        # create the kernel object
        k = cls(graph, inA, outA, Fs, axis)

        # create parameter objects for the input and output
        params = (Parameter(inA, MPEnums.INPUT),
                  Parameter(outA, MPEnums.OUTPUT))

        # add the kernel to a generic node object
        node = Node(graph, k, params)

        # add the node to the graph
        graph.add_node(node)

        # if initialization data is provided, add it to the graph
        if init_input is not None:
            node.add_initialization_data([init_input], init_labels)

        return node
