from ..core import BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Scalar, Tensor
from .kernel_utils import extract_init_inputs

import numpy as np


class Unary:
    def _initialize(self,init_inputs, init_outputs, labels):
        """
        Initialize the kernel if there is an internal state to initialize, including downstream initialization data
        """
        # get the initialization params
        init_in = init_inputs[0]
        init_out = init_outputs[0]

        # check if initialization is needed
        if init_in is None or init_out is None:
            # init not needed
            return

        # check the init inputs are in valid data objects
        accepted_inputs = (BcipEnums.TENSOR,BcipEnums.ARRAY,BcipEnums.CIRCLE_BUFFER)
        if init_in.bcip_type not in accepted_inputs:
            raise TypeError("Invalid initialization input type")
    
        # set the output size, as needed
        if init_out.virtual:
            init_out.shape = init_in.shape

        self._process_data(init_inputs, init_outputs)


class AbsoluteKernel(Unary, Kernel):
    """
    Calculate the element-wise absolute value of Tensor elements

    Parameters
    ----------

    graph : Graph
        Graph that the kernel should be added to
    inA : Tensor or Scalar
        Input trial data
    outA : Tensor or Scalar
        Output trial data

    """

    def __init__(self, graph, inA, outA):
        """
        Constructor for the absolute value kernel
        """
        super().__init__("Absolute", BcipEnums.INIT_FROM_NONE, graph)
        self.inputs = [inA]
        self.outputs = [outA]

    def _process_data(self, inputs, outputs):
        """
        Calculate the absolute value of the input data, and assign it to the output data
        """
        outputs[0].data = np.absolute(inputs[0].data)

    @classmethod
    def add_absolute_node(cls, graph, inA, outA):
        """
        Factory method to create an absolute value kernel
        and add it to a graph as a generic node object.

        Parameters
        ----------

        graph : Graph 
            Graph that the kernel should be added to
        inA : Tensor or Scalar 
            Input trial data
        outA : Tensor or Scalar 
            Output trial data

        Return
        ------
        node : Node 
            Node object containing the absolute kernel and parameters

        """

        # create the kernel object
        k = cls(graph, inA, outA)

        # create parameter objects for the input and output
        params = (Parameter(inA, BcipEnums.INPUT), Parameter(outA, BcipEnums.OUTPUT))

        # add the kernel to a generic node object
        node = Node(graph, k, params)

        # add the node to the graph
        graph.add_node(node)

        return node


class LogKernel(Unary, Kernel):
    """
    Kernel to perform element-wise natural logarithm operation on
    one BCIP data container (i.e. tensor or scalar)

    Numpy broadcasting rules apply.

    Parameters
    ----------
    graph : Graph 
        Graph that the kernel should be added to
    inA : Tensor or Scalar 
        Input trial data
    outA : Tensor or Scalar 
        Output trial data

    """

    def __init__(self, graph, inA, outA):
        super().__init__("Log", BcipEnums.INIT_FROM_NONE, graph)
        self.inputs = [inA]
        self.outputs = [outA]

    def _process_data(self, inputs, outputs):
        """
        Calculate the natural logarithm of the input data, and assign it to the output data

        Parameters
        ----------
        input_data : Tensor or Scalar 
            Input trial data
        output_data : Tensor or Scalar 
            Output trial data

        """
        outputs[0].data = np.log(inputs[0].data)

    @classmethod
    def add_log_node(cls, graph, inA, outA):
        """
        Factory method to create a log kernel
        and add it to a graph as a generic node object.

        Parameters
        ----------
        graph : Graph 
            Graph that the node should be added to

        inA : Tensor or Scalar 
            Input trial data

        outA : Tensor or Scalar 
            Output trial data

        Return
        ------
        node : Node 
            Node object containing the log kernel and parameters

        Return Type
        -----------
        BCIPy Node object
        """

        # create the kernel object
        k = cls(graph, inA, outA)

        # create parameter objects for the input and output
        params = (Parameter(inA, BcipEnums.INPUT), Parameter(outA, BcipEnums.OUTPUT))

        # add the kernel to a generic node object
        node = Node(graph, k, params)

        # add the node to the graph
        graph.add_node(node)

        return node


class Binary:
    def _initialize(self, init_inputs, init_outputs, labels):
        """
        This kernel has no internal state that must be initialized
        """
        # get the initialization params
        init_inA = init_inputs[0]
        init_inB = init_inputs[1]
        init_out = init_outputs[0]

        # check if initialization is needed
        if init_inA is None or init_inB is None or init_out is None:
            # init not needed
            return

        accepted_data_inputs = (BcipEnums.TENSOR, BcipEnums.ARRAY,
                                BcipEnums.CIRCLE_BUFFER, BcipEnums.SCALAR)
        
        # check the init inputs are in valid data objects
        for init_obj in (init_inA, init_inB):
            if init_obj.bcip_type not in accepted_data_inputs:
                raise TypeError("Invalid initialization input type")
            
        # extract the data from the input
        X = [None] * 2
        for i, i_in in enumerate((init_inA, init_inB)):
            X[i] = extract_init_inputs(i_in)

        # determine output dimensions and adjust init_out shape
        phony_out = X[0] + X[1]
        init_out.shape = phony_out.shape
        tmp_inA = Tensor.create_from_data(self.session, X[0].shape, X[0])
        tmp_inB = Tensor.create_from_data(self.session, X[1].shape, X[1])
        self._process_data([tmp_inA, tmp_inB], init_outputs)


class AdditionKernel(Binary, Kernel):
    """
    Kernel to add two BCIPP data containers (i.e. tensor or scalar) together

    Parameters
    ----------
    graph : Graph 
        Graph that the kernel should be added to
    inA : Tensor or Scalar 
        First input trial data
    inB : Tensor or Scalar 
        Second input trial data
    outA : Tensor or Scalar 
        Output trial data

    """

    def __init__(self, graph, inA, inB, outA):
        super().__init__("Addition", BcipEnums.INIT_FROM_NONE, graph)
        self.inputs = [inA, inB]
        self.outputs = [outA]

    def _process_data(self, inputs, outputs):
        outputs[0].data = inputs[0].data + inputs[1].data

    @classmethod
    def add_addition_node(cls, graph, inA, inB, outA):
        """
        Factory method to create an addition kernel and add it to a graph
        as a generic node object.

        Parameters
        ----------
        graph : Graph 
        Graph that the kernel should be added to
        inA : Tensor or Scalar 
            First input trial data
        inB : Tensor or Scalar 
            Second input trial data
        outA : Tensor or Scalar 
            Output trial data

        Returns
        -------
        node : Node 
            Node object that has kernel and parameter stored in it

        Return type
        -----------
        BCIPy Node 
        """

        # create the kernel object
        k = cls(graph, inA, inB, outA)

        # create parameter objects for the input and output
        params = (
            Parameter(inA, BcipEnums.INPUT),
            Parameter(inB, BcipEnums.INPUT),
            Parameter(outA, BcipEnums.OUTPUT),
        )

        # add the kernel to a generic node object
        node = Node(graph, k, params)

        # add the node to the graph
        graph.add_node(node)

        return node


class DivisionKernel(Binary, Kernel):
    """
    Kernel to divide two BCIP data containers (i.e. tensor or scalar)
    together

    .. note:: This is element-wise division (ie. _inA ./ _inB)

    Parameters
    ----------
    graph : Graph 
        Graph that the kernel should be added to
    inA : Tensor or Scalar 
        First input trial data
    inB : Tensor or Scalar 
        Second input trial data
    outA : Tensor or Scalar 
        Output trial data

    """

    def __init__(self, graph, inA, inB, outA):
        """
        Constructor for the division kernel
        """
        super().__init__("Division", BcipEnums.INIT_FROM_NONE, graph)
        self.inputs = [inA, inB]
        self.outputs = [outA]

    def _process_data(self, inputs, outputs):
        """
        Calculate the quotient of the input tensors, and assign it to the output data
        """
        outputs[0].data = inputs[0].data / inputs[1].data

    @classmethod
    def add_division_node(cls, graph, inA, inB, outA):
        """
        Factory method to create a element-wise divsion kernel and add it to a graph
        as a generic node object.

        Parameters
        ----------
        graph : Graph 
            Graph that the kernel should be added to
        inA : Tensor or Scalar 
            First input trial data
        inB : Tensor or Scalar 
            Second input trial data
        outA : Tensor or Scalar 
            Output trial data

        Returns
        -------
        node : Node 
            Node object that has kernel and parameter stored in it

        Return type
        -----------
        BCIPy Node object
        """

        # create the kernel object
        k = cls(graph, inA, inB, outA)

        # create parameter objects for the input and output
        params = (
            Parameter(inA, BcipEnums.INPUT),
            Parameter(inB, BcipEnums.INPUT),
            Parameter(outA, BcipEnums.OUTPUT),
        )

        # add the kernel to a generic node object
        node = Node(graph, k, params)

        # add the node to the graph
        graph.add_node(node)

        return node


class MultiplicationKernel(Binary, Kernel):
    """
    Kernel to multiply two BCIPP data containers (i.e. tensor or scalar)
    together

    .. note:: This is element-wise multiplication (ie. _inA .* _inB)

    Parameters
    ----------
    graph : Graph 
        Graph that the kernel should be added to
    inA : Tensor or Scalar 
        First input trial data
    inB : Tensor or Scalar 
        Second input trial data
    outA : Tensor or Scalar 
        Output trial data

    """

    def __init__(self, graph, inA, inB, outA):
        super().__init__("Multiplication", BcipEnums.INIT_FROM_NONE, graph)
        self.inputs = [inA, inB]
        self.outputs = [outA]

    def _process_data(self, inputs, outputs):
        """
        Calculate the product of the input tensors, and assign it to the output data
        """
        outputs[0].data = inputs[0].data * inputs[1].data

    @classmethod
    def add_multiplication_node(cls, graph, inA, inB, outA):
        """
        Factory method to create a multiplication kernel and add it to a graph
        as a generic node object.

        Parameters
        ----------
        graph : Graph 
            Graph that the node should be added to
        inA : Tensor or Scalar 
            First input trial data
        inB : Tensor or Scalar 
            Second input trial data
        outA : Tensor or Scalar 
            Output trial data

        Returns
        -------
        node : Node 
            Node object that has kernel and parameter stored in it

        Return type
        -----------
        BCIPy Node object
        """

        # create the kernel object
        k = cls(graph, inA, inB, outA)

        # create parameter objects for the input and output
        params = (
            Parameter(inA, BcipEnums.INPUT),
            Parameter(inB, BcipEnums.INPUT),
            Parameter(outA, BcipEnums.OUTPUT),
        )

        # add the kernel to a generic node object
        node = Node(graph, k, params)

        # add the node to the graph
        graph.add_node(node)

        return node


class SubtractionKernel(Binary, Kernel):
    """
    Kernel to calculate the difference between two BCIP data containers
    (i.e. tensor or scalar)

    .. note:: This is element-wise subtraction (ie. _inA - _inB)

    Parameters
    ----------
    graph : Graph 
        Graph that the kernel should be added to
    inA : Tensor or Scalar 
        First input trial data
    inB : Tensor or Scalar 
        Second input trial data
    outA : Tensor or Scalar 
        Output trial data

    """

    def __init__(self, graph, inA, inB, outA):
        super().__init__("Subtraction", BcipEnums.INIT_FROM_NONE, graph)
        self.inputs = [inA, inB]
        self.outputs = [outA]

    def _process_data(self, inputs, outputs):
        """
        Process data according to outlined kernel function
        """
        outputs[0].data = inputs[0].data - inputs[1].data

    @classmethod
    def add_subtraction_node(cls, graph, inA, inB, outA):
        """
        Factory method to create a kernel and add it to a graph
        as a generic node object.

        Parameters
        ----------
        graph : Graph 
            Graph that the kernel should be added to
        inA : Tensor or Scalar 
            First input trial data
        inB : Tensor or Scalar 
            Second input trial data
        outA : Tensor or Scalar 
            Output trial data

        Returns
        -------
        node : Node 
            Node object that has kernel and parameter stored in it

        Return type
        -----------
        BCIPy Node object
        """

        # create the kernel object
        k = cls(graph, inA, inB, outA)

        # create parameter objects for the input and output
        params = (
            Parameter(inA, BcipEnums.INPUT),
            Parameter(inB, BcipEnums.INPUT),
            Parameter(outA, BcipEnums.OUTPUT),
        )

        # add the kernel to a generic node object
        node = Node(graph, k, params)

        # add the node to the graph
        graph.add_node(node)

        return node
