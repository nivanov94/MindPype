from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Tensor
from .kernel_utils import extract_init_inputs

import numpy as np


class Unary:
    """
    Base class for unary arithmetic operator kernels.

    Kernel to perform element-wise unary arithmetic operation on
    one MindPype data container (e.g., tensor or scalar).

    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to
    inA : Tensor or Scalar 
        Input data container
    outA : Tensor or Scalar 
        Output data container

    See Also
    --------
    Kernel : Base class for all kernels
    AbsoluteKernel : Kernel to calculate the absolute value of a MindPype data container
    LogKernel : Kernel to calculate the natural logarithm of a MindPype data container
    """

    def _initialize(self, init_inputs, init_outputs, labels=None):
        """
        Initialize the kernel and compute initialization data output.

        Parameters
        ----------
        init_inputs : List of Tensors or Arrays
            Initialization input data container, list of length 1
        init_outputs : List of Tensors or Arrays
            Initialization output data container, list of length 1
        labels : None
            Not used, here for compatability with other kernels
        """

        # get the initialization params
        init_in = init_inputs[0]
        init_out = init_outputs[0]

        # check if initialization is needed
        if init_in is None or init_out is None:
            # init not needed
            return

        # check the init inputs are in valid data objects
        accepted_inputs = (MPEnums.TENSOR,MPEnums.ARRAY,MPEnums.CIRCLE_BUFFER)
        if init_in.mp_type not in accepted_inputs:
            raise TypeError("Invalid initialization input type")

        if init_in.mp_type != MPEnums.TENSOR:
            init_in = init_in.to_tensor()

        # set the output size, as needed
        if init_out.virtual:
            init_out.shape = init_in.shape

        self._process_data([init_in], init_outputs)


class AbsoluteKernel(Unary, Kernel):
    """
    Kernel to calculate the element-wise absolute value of
    one MindPype data container (i.e. tensor or scalar)

    .. note::
        This kernel utilizes the numpy function
        :func:`absolute <numpy:numpy.absolute>`.

    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to
    inA : Tensor or Scalar
        Input data container
    outA : Tensor or Scalar
        Output data container

    See Also
    --------
    Kernel : Base class for all kernels
    Unary : Base class for all unary arithmetic operator kernels
    """

    def __init__(self, graph, inA, outA):
        """Init"""
        super().__init__("Absolute", MPEnums.INIT_FROM_NONE, graph)
        self.inputs = [inA]
        self.outputs = [outA]

    def _process_data(self, inputs, outputs):
        """
        Calculate the absolute value of the input data
        and assign it to the output container

        Parameters
        ----------
        inputs: List of Tensors or Scalars
            Input data container, list of length 1
        outputs: List of Tensors or Scalars
            Output data container, list of length 1
        """
        outputs[0].data = np.absolute(inputs[0].data)

    @classmethod
    def add_to_graph(cls, graph, inA, outA, init_input=None, init_labels=None):
        """
        Factory method to create an absolute value kernel node
        and add it to a graph.

        Parameters
        ----------
        graph : Graph
            Graph that the kernel should be added to
        inA : Tensor or Scalar 
            Input data container
        outA : Tensor or Scalar data 
            Output data container
        init_input : Tensor or Scalar, default=None
            MindPype data container with initialization data to be
            transformed and passed to downstream nodes during graph
            initialization

        Return
        ------
        node : Node
            Node object containing the absolute kernel and parameters

        """

        # create the kernel object
        k = cls(graph, inA, outA)

        # create parameter objects for the input and output
        params = (Parameter(inA, MPEnums.INPUT), Parameter(outA, MPEnums.OUTPUT))

        # add the kernel to a generic node object
        node = Node(graph, k, params)

        # add the node to the graph
        graph.add_node(node)

        # if initialization data is provided, add it to the node
        if init_input is not None:
            node.add_initialization_data([init_input], init_labels)

        return node


class LogKernel(Unary, Kernel):
    """
    Kernel to perform element-wise natural logarithm operation on
    one MindPype data container (i.e. tensor or scalar)

    .. note::
        This kernel utilizes the numpy function
        :func:`log <numpy:numpy.log>`.

    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to
    inA : Tensor or Scalar
        Input data
    outA : Tensor or Scalar
        Output data
    """

    def __init__(self, graph, inA, outA):
        """Init"""
        super().__init__("Log", MPEnums.INIT_FROM_NONE, graph)
        self.inputs = [inA]
        self.outputs = [outA]

    def _process_data(self, inputs, outputs):
        """
        Calculate the natural logarithm of the input data
        and assign it to the output container

        Parameters
        ----------
        input_data : list of Tensors or Scalars
            Input data container, list of length 1
        output_data : Tensor or Scalar
            Output data container, list of length 1

        """
        outputs[0].data = np.log(inputs[0].data)

    @classmethod
    def add_to_graph(cls, graph, inA, outA, init_input=None, init_labels=None):
        """
        Factory method to create a log kernel node
        and add it to a graph.

        Parameters
        ----------
        graph : Graph
            Graph that the kernel should be added to
        inA : Tensor or Scalar
            Input data container
        outA : Tensor or Scalar 
            Output data container
        init_input : Tensor or Scalar, default=None
            MindPype data container with initialization data to be
            transformed and passed to downstream nodes during graph
            initialization
        init_labels : Tensor or Array, default=None
            MindPype data container with initialization labels to be
            passed to downstream nodes during graph initialization

        Return
        ------
        node : Node
            Node object containing the log kernel and parameters
        """

        # create the kernel object
        k = cls(graph, inA, outA)

        # create parameter objects for the input and output
        params = (Parameter(inA, MPEnums.INPUT), Parameter(outA, MPEnums.OUTPUT))

        # add the kernel to a generic node object
        node = Node(graph, k, params)

        # add the node to the graph
        graph.add_node(node)

        # if initialization data is provided, add it to the node
        if init_input is not None:
            node.add_initialization_data([init_input], init_labels)

        return node


class Binary:
    """
    Base class for binary arithmetic operator kernels.

    Kernel to perform binary arithmetic operation on
    one MindPype data container (e.g., tensor or scalar).
    Numpy broadcasting rules apply.

    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to
    inA : Tensor or Scalar 
        Input data container
    inB : Tensor or Scalar 
        Input data container
    outA : Tensor or Scalar
        Output data container

    See Also
    --------
    Kernel : Base class for all kernels
    AdditionKernel : Kernel to compute sum of two MindPype data containers
    DivisionKernel : Kernel to compute quotient of two MindPype data containers
    MultiplicationKernel : Kernel to compute product of two MindPype data containers
    SubtractionKernel : Kernel to compute difference of two MindPype data containers
    """

    def _initialize(self, init_inputs, init_outputs, labels=None):
        """
        Initialize the kernel and compute initialization data output.

        Parameters
        ----------
        init_inputs : List of Tensors or Arrays 
            Initialization input data container, list of length 2
        init_outputs : List of Tensors or Arrays
            Initialization output data container, list of length 1
        labels : None
            Not used, here for compatability with other kernels
        """

        # get the initialization params
        init_inA = init_inputs[0]
        init_inB = init_inputs[1]
        init_out = init_outputs[0]

        # check if initialization is needed
        if init_inA is None or init_inB is None or init_out is None:
            # init not needed
            return

        accepted_data_inputs = (MPEnums.TENSOR, MPEnums.ARRAY,
                                MPEnums.CIRCLE_BUFFER, MPEnums.SCALAR)

        # check the init inputs are in valid data objects
        for init_obj in (init_inA, init_inB):
            if init_obj.mp_type not in accepted_data_inputs:
                raise TypeError("Invalid initialization input type")

        # extract the data from the input
        X = [None] * 2
        for i, i_in in enumerate((init_inA, init_inB)):
            X[i] = extract_init_inputs(i_in)

        # determine output dimensions and adjust init_out shape
        phony_out = X[0] + X[1]
        init_out.shape = phony_out.shape
        tmp_inA = Tensor.create_from_data(self.session, X[0])
        tmp_inB = Tensor.create_from_data(self.session, X[1])
        self._process_data([tmp_inA, tmp_inB], init_outputs)


class AdditionKernel(Binary, Kernel):
    """
    Kernel to sum two MindPype data containers together

    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to
    inA : Tensor or Scalar 
        Input data container
    inB : Tensor or Scalar
        Input data container
    outA : Tensor or Scalar 
        Output data container

    See Also
    --------
    Kernel : Base class for all kernels
    Binary : Base class for all binary arithmetic operator kernels
    add_addition_node : Factory method to create an addition kernel node and add it to a graph
    """

    def __init__(self, graph, inA, inB, outA):
        """Init"""
        super().__init__("Addition", MPEnums.INIT_FROM_NONE, graph)
        self.inputs = [inA, inB]
        self.outputs = [outA]

    def _process_data(self, inputs, outputs):
        """
        Calculate the sum of the input data
        and assign it to the output container

        Parameters
        ----------
        input_data : List of data containers
            Input data containers, list of length 2
        output_data : List of data containers
            Output data containers, list of length 1
        """
        outputs[0].data = inputs[0].data + inputs[1].data

    @classmethod
    def add_to_graph(cls, graph, inA, inB, outA, init_inputs=None, init_labels=None):
        """
        Factory method to create an addition kernel node
        and add it to a graph

        Parameters
        ----------
        graph : Graph
            Graph that the kernel should be added to
        inA : Tensor or Scalar
            Input data container
        inB : Tensor or Scalar 
            Input data container
        outA : Tensor or Scalar 
            Output data container
        init_inputs : List of two Tensors or Scalars, default=None
            MindPype data containers with initialization data to be
            transformed and passed to downstream nodes during graph
            initialization
        init_labels : Tensor or Array, default=None
            MindPype data container with initialization labels to be
            passed to downstream nodes during graph initialization

        See Also
        --------
        AdditionKernel : Kernel to sum two MindPype data containers together
        """

        # create the kernel object
        k = cls(graph, inA, inB, outA)

        # create parameter objects for the input and output
        params = (
            Parameter(inA, MPEnums.INPUT),
            Parameter(inB, MPEnums.INPUT),
            Parameter(outA, MPEnums.OUTPUT),
        )

        # add the kernel to a generic node object
        node = Node(graph, k, params)

        # add the node to the graph
        graph.add_node(node)

        # if initialization data is provided, add it to the node
        if init_inputs is not None:
            node.add_initialization_data(init_inputs, init_labels)

        return node


class DivisionKernel(Binary, Kernel):
    """
    Kernel to compute the quotient of two MindPype data containers

    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to
    inA : Tensor or Scalar 
        Input containing the dividend
    inB : Tensor or Scalar 
        Input containing the divisor
    outA : Tensor or Scalar 
        Output data container

    See Also
    --------
    Kernel : Base class for all kernels
    Binary : Base class for all binary arithmetic operator kernels
    add_division_node : Factory method to create a division kernel node and add it to a graph
    """

    def __init__(self, graph, inA, inB, outA):
        """Init"""
        super().__init__("Division", MPEnums.INIT_FROM_NONE, graph)
        self.inputs = [inA, inB]
        self.outputs = [outA]

    def _process_data(self, inputs, outputs):
        """
        Calculate the quotient of the input tensors
        and assign it to the output container

        Parameters
        ----------
        input_data : List of data containers
            Input data containers, list of length 2
        output_data : List of data containers
            Output data containers, list of length 1
        """
        outputs[0].data = inputs[0].data / inputs[1].data

    @classmethod
    def add_to_graph(cls, graph, inA, inB, outA, init_inputs=None, init_labels=None):
        """
        Factory method to create a division kernel node
        and add it to a graph.

        Parameters
        ----------
        graph : Graph
            Graph that the kernel should be added to
        inA : Tensor or Scalar 
            Input containing the dividend
        inB : Tensor or Scalar 
            Input containing the divisor
        outA : Tensor or Scalar 
            Output data container
        init_inputs : List of two Tensors or Scalars, default=None
            MindPype data containers with initialization data to be
            transformed and passed to downstream nodes during graph
            initialization
        init_labels : Tensor or Array, default=None
            MindPype data container with initialization labels to be
            passed to downstream nodes during graph initialization

        See Also
        --------
        DivisionKernel : Kernel to compute the quotient of two MindPype data containers
        """

        # create the kernel object
        k = cls(graph, inA, inB, outA)

        # create parameter objects for the input and output
        params = (
            Parameter(inA, MPEnums.INPUT),
            Parameter(inB, MPEnums.INPUT),
            Parameter(outA, MPEnums.OUTPUT),
        )

        # add the kernel to a generic node object
        node = Node(graph, k, params)

        # add the node to the graph
        graph.add_node(node)

        # if initialization data is provided, add it to the node
        if init_inputs is not None:
            node.add_initialization_data(init_inputs, init_labels)

        return node


class MultiplicationKernel(Binary, Kernel):
    """
    Kernel to compute the product of two MindPype data containers

    .. note:: This is an element-wise multiplication operation

    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to
    inA : Tensor or Scalar
        Input data container
    inB : Tensor or Scalar
        Input data container
    outA : Tensor or Scalar 
        Output data container

    See Also
    --------
    Kernel : Base class for all kernels
    Binary : Base class for all binary arithmetic operator kernels
    add_multiplication_node : Factory method to create a multiplication kernel node and add it to a graph
    """

    def __init__(self, graph, inA, inB, outA):
        """Init"""
        super().__init__("Multiplication", MPEnums.INIT_FROM_NONE, graph)
        self.inputs = [inA, inB]
        self.outputs = [outA]

    def _process_data(self, inputs, outputs):
        """
        Calculate the product of the inputs
        and assign it to the output container

        Parameters
        ----------
        input_data : List of data containers
            Input data containers, list of length 2
        output_data : List of data containers
            Output data containers, list of length 1
        """
        outputs[0].data = inputs[0].data * inputs[1].data

    @classmethod
    def add_to_graph(cls, graph, inA, inB, outA, init_inputs=None, init_labels=None):
        """
        Factory method to create a multiplication kernel node
        and add it to a graph

        Parameters
        ----------
        graph : Graph
            Graph that the kernel should be added to
        inA : Tensor or Scalar 
            Input data container
        inB : Tensor or Scalar 
            Input data container
        outA : Tensor or Scalar 
            Output data container
        init_inputs : List of two data containers, default=None
            MindPype data containers with initialization data to be
            transformed and passed to downstream nodes during graph
            initialization
        init_labels : Tensor or Array, default=None
            MindPype data container with initialization labels to be
            passed to downstream nodes during graph initialization

        Returns
        -------
        node : Node
            Node object that has kernel and parameter stored in it
        """

        # create the kernel object
        k = cls(graph, inA, inB, outA)

        # create parameter objects for the input and output
        params = (
            Parameter(inA, MPEnums.INPUT),
            Parameter(inB, MPEnums.INPUT),
            Parameter(outA, MPEnums.OUTPUT),
        )

        # add the kernel to a generic node object
        node = Node(graph, k, params)

        # add the node to the graph
        graph.add_node(node)

        # if initialization data is provided, add it to the node
        if init_inputs is not None:
            node.add_initialization_data(init_inputs, init_labels)

        return node


class SubtractionKernel(Binary, Kernel):
    """
    Kernel to calculate the difference between two MindPype data containers

    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to
    inA : Tensor or Scalar
        Input containing the minuend
    inB : Tensor or Scalar 
        Input containing the subtrahend
    outA : Tensor or Scalar
        Output data container

    See Also
    --------
    Kernel : Base class for all kernels
    Binary : Base class for all binary arithmetic operator kernels
    add_subtraction_node : Factory method to create a subtraction kernel node and add it to a graph
    """

    def __init__(self, graph, inA, inB, outA):
        """Init"""
        super().__init__("Subtraction", MPEnums.INIT_FROM_NONE, graph)
        self.inputs = [inA, inB]
        self.outputs = [outA]

    def _process_data(self, inputs, outputs):
        """
        Calculate the difference between the inputs
        and assign it to the output container

        Parameters
        ----------
        input_data : List of data containers
            Input data containers, list of length 2
        output_data : List of data containers
            Output data containers, list of length 1
        """
        outputs[0].data = inputs[0].data - inputs[1].data

    @classmethod
    def add_to_graph(cls, graph, inA, inB, outA, init_inputs=None, init_labels=None):
        """
        Factory method to create a subtraction kernel node
        and add it to a graph

        Parameters
        ----------
        graph : Graph
            Graph that the kernel should be added to
        inA : Tensor or Scalar
            Input containing the minuend
        inB : Tensor or Scalar 
            Input containing the subtrahend
        outA : Tensor or Scalar
            Output data container
        init_inputs : List of two data containers, default=None
            MindPype data containers with initialization data to be
            transformed and passed to downstream nodes during graph
            initialization
        init_labels : Tensor or Array, default=None
            MindPype data container with initialization labels to be
            passed to downstream nodes during graph initialization

        Returns
        -------
        node : Node
            Node object that has kernel and parameter stored in it

        See Also
        --------
        SubtractionKernel : Kernel to calculate the difference between two MindPype data containers
        """

        # create the kernel object
        k = cls(graph, inA, inB, outA)

        # create parameter objects for the input and output
        params = (
            Parameter(inA, MPEnums.INPUT),
            Parameter(inB, MPEnums.INPUT),
            Parameter(outA, MPEnums.OUTPUT),
        )

        # add the kernel to a generic node object
        node = Node(graph, k, params)

        # add the node to the graph
        graph.add_node(node)

        # if initialization data is provided, add it to the node
        if init_inputs is not None:
            node.add_initialization_data(init_inputs, init_labels)

        return node
