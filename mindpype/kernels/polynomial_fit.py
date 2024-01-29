from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


class PolynomialFitKernel(Kernel):
    """
    Fits a polynomial to time series data and outputs the fit values

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

    order : int
        Order of the polynomial to fit
    """

    def __init__(self, graph, inA, outA, Fs=1, order=2):
        """
        Kernel fits a polynomial to time series data and outputs the fit values
        """
        super().__init__('PolynomialFit',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA]
        self.outputs = [outA]
        self._Fs = Fs
        self._order = order
        self._axis = -1 # for now, only support fitting along the last axis

    def _initialize(self, init_inputs, init_outputs, labels):
        """
        This kernel has no internal state that must be initialized.
        Call initialization_execution if downstream nodes are missing training
        data
        """

        init_in = init_inputs[0]
        init_out = init_outputs[0]

        if (init_out is not None and
            init_in is not None and
            init_in.shape != ()):
            if init_in is not None and init_in.mp_type != MPEnums.TENSOR:
                init_in = init_in.to_tensor()

            # update output size, as needed
            if init_out.virtual:
                output_sz = init_in.shape[:-1] + (self._order,)
                init_out.shape = output_sz

            self._process_data([init_in], init_outputs)

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

        if self._order < 0:
            raise ValueError("Order must be greater than 0")

        if self._Fs <= 0:
            raise ValueError("Sampling frequency must be greater than 0")

        out_shape = inA.shape[:-1] + (self._order,)
        if outA.virtual and len(outA.shape) == 0:
            outA.shape = out_shape

        if outA.shape != out_shape:
            raise ValueError(f"Output shape must be {out_shape}")

    def _process_data(self, inputs, outputs):
        """
        Execute the kernel and compute the polynomial fit
        """

        inA = inputs[0]
        outA = outputs[0]

        Ns = inA.shape[self._axis]
        x = np.linspace(0, Ns/self._Fs, Ns)

        # reshape the input data to 2D
        if len(inA.shape) == 1:
            y = inA.data[np.newaxis,:]
        else:
            # move the target axis to the end
            y = np.moveaxis(inA.data, self._axis, -1)
            y = y.reshape(-1, y.shape[-1])

        # fit polynomials for each time series
        coefs = np.zeros((y.shape[0], self._order))
        for i_t, ts in enumerate(y):
            model = LinearRegression() #make_pipeline(PolynomialFeatures(self._order), LinearRegression())
            model.fit(x[:, np.newaxis], ts[:, np.newaxis])
            coefs[i_t,:] = model.coef_.squeeze() #model.steps[1][1].coef_.squeeze()[1:]

        # reshape the coefs to match the output shape
        outA.data = np.reshape(coefs, outA.shape)

    @classmethod
    def add_polynomial_fit_node(cls, graph, inA, outA, Fs=1, order=2,
                                init_inputs=None, init_labels=None):
        """
        Factory method to create a polynomial fit kernel

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

        order : int
            Order of the polynomial to fit

        init_inputs : Tensor or Array, default = None
            Initialization data for the graph

        init_labels : Tensor or Array, default = None
            Initialization labels for the graph
        """

        # create the kernel object
        k = cls(graph, inA, outA, Fs, order)

        # create parameter objects for the input and output
        params = (Parameter(inA, MPEnums.INPUT),
                  Parameter(outA, MPEnums.OUTPUT))

        # add the kernel to a generic node object
        node = Node(graph, k, params)

        # add the node to the graph
        graph.add_node(node)

        # if initialization data is provided, add it to the graph
        if init_inputs is not None:
            node.add_initialization_data(init_inputs, init_labels)

        return node
