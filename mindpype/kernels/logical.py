from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Tensor
from .kernel_utils import extract_init_inputs

import numpy as np

class Unary:
    def _initialize(self, init_inputs, init_outputs, labels):
        """
        This kernel has no internal state that must be initialized
        """
        # get the input and output initialization parameters
        init_in = init_inputs[0]
        init_out = init_outputs[0]

        if init_in is not None and init_in.mp_type != MPEnums.TENSOR:
            init_in = init_in.to_tensor()

        if init_out is not None and (init_in is not None and init_in.shape != ()):
            # set the output size, as needed
            if init_out.virtual:
                init_out.shape = init_in.shape

            self._process_data([init_in], init_outputs)


class NotKernel(Unary, Kernel):
    """
    Kernel to perform logical NOT operation elementwise on
    one MindPype data container (i.e. tensor or scalar)

    Numpy broadcasting rules apply.

    .. note::
        This kernel utilizes the numpy function
        :data:`logical_not <numpy:numpy.logical_not>`.

    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to

    inA : Tensor or Scalar
        Input data

    outA : Tensor or Scalar
        Output data
    """

    def __init__(self,graph,inA,outA):
        """ Init """
        super().__init__('NOT',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA]
        self.outputs = [outA]

    def _process_data(self, inputs, outputs):
        """
        Perform logical NOT operation elementwise

        Parameters
        ----------

        inputs: list of Tensors
            Input data container, list of length 1

        outputs: list pf Tensors
            Output data container, list of length 1
        """
        outputs[0].data = np.logical_not(inputs[0].data)

    @classmethod
    def add_to_graph(cls,graph,inA,outA,init_input=None,init_labels=None):
        """
        Factory method to create a logical NOT kernel
        and add it to a graph as a generic node object.

        Parameters
        ----------
        graph : Graph
            Graph that the node should be added to

        inA : Tensor or Scalar
            Input data

        outA : Tensor or Scalar
            Output data

        Returns
        -------
        node : Node
            Node object that contains the kernel
        """

        # create the kernel object
        k = cls(graph,inA,outA)

        # create parameter objects for the input and output
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(outA,MPEnums.OUTPUT))

        # add the kernel to a generic node object
        node = Node(graph,k,params)

        # add the node to the graph
        graph.add_node(node)

        # if initialization data is provided, then add it to the node
        if init_input is not None:
            node.add_initialization_data([init_input], init_labels)

        return node

class Binary:
    """ 
    Base class for binary logical operator kernels.

    Parameters
    ----------

    graph: Graph
        Graph that the kernel should be added to
        
    inA: Tensor or Scalar
        Input data

    inB: Tensor or Scalar
        Input data

    outA: Tensor or Scalar
        Output data
    """
    def _initialize(self, init_inputs, init_outputs, labels):
        """
        This kernel has no internal state that must be initialized
        """
        init_inA, init_inB = init_inputs
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
        phony_out = np.logical_and(X[0],X[1])
        init_out.shape = phony_out.shape
        tmp_inA = Tensor.create_from_data(self.session, X[0])
        tmp_inB = Tensor.create_from_data(self.session, X[1])
        self._process_data([tmp_inA, tmp_inB], init_outputs)


class AndKernel(Binary,Kernel):
    """
    Kernel to perform logical AND operation elementwise on
    two MindPype data containers (i.e. tensor or scalar)

    Numpy broadcasting rules apply.

    .. note::
        This kernel utilizes the numpy function
        :data:`logical_and <numpy:numpy.logical_and>`.

    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to

    inA : Tensor or Scalar
        Input 1 data

    inB : Tensor or Scalar
        Input 2 data

    outA : Tensor or Scalar
        Output data
    """

    def __init__(self,graph,inA,inB,outA):
        """ Init """
        super().__init__('AND',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA,inB]
        self.outputs = [outA]

    def _process_data(self, inputs, outputs):
        """
        Perform logical AND operation elementwise

        Parameters
        ----------

        inputs: list of Tensors or Scalars
            Input data container, list of length 2

        outputs: list of Tensors or Scalars
            Output data container, list of length 1

        """
        outputs[0].data = np.logical_and(inputs[0].data, inputs[1].data)

    @classmethod
    def add_to_graph(cls,graph,inA,inB,outA,init_inputs=None,init_labels=None):
        """
        Factory method to create a logical AND kernel
        and add it to a graph as a generic node object.

        Parameters
        ----------
        graph : Graph
            Graph that the node should be added to

        inA : Tensor or Scalar
            Input 1 data

        inB : Tensor or Scalar
            Input 2 data

        outA : Tensor or Scalar
            Output data

        """

        # create the kernel object
        k = cls(graph,inA,inB,outA)

        # create parameter objects for the input and output
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(inB,MPEnums.INPUT),
                  Parameter(outA,MPEnums.OUTPUT))

        # add the kernel to a generic node object
        node = Node(graph,k,params)

        # add the node to the graph
        graph.add_node(node)

        # if initialization data is provided, then add it to the node
        if init_inputs is not None:
            node.add_initialization_data(init_inputs, init_labels)

        return node

class OrKernel(Binary,Kernel):
    """
    Kernel to perform logical OR operation elementwise on
    two MindPype data containers (i.e. tensor or scalar)

    Numpy broadcasting rules apply.

    .. note::
        This kernel utilizes the numpy function
        :data:`logical_or <numpy:numpy.logical_or>`.

    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to

    inA : Tensor or Scalar
        Input 1 data

    inB : Tensor or Scalar
        Input 2 data

    outA : Tensor or Scalar
        Output data
    """

    def __init__(self,graph,inA,inB,outA):
        """ Init """
        super().__init__('OR',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA,inB]
        self.outputs = [outA]

    def _process_data(self, inputs, outputs):
        """
        Perform logical OR operation elementwise

        Parameters
        ----------
        inputs: list of Tensors or Scalars
            Input data container, list of length 2

        outputs: list of Tensors or Scalars
            Output data container, list of length 1
        """
        outputs[0].data = np.logical_or(inputs[0].data, inputs[1].data)

    @classmethod
    def add_to_graph(cls,graph,inA,inB,outA,init_inputs=None,init_labels=None):
        """
        Factory method to create a logical OR kernel
        and add it to a graph as a generic node object.

        Parameters
        ----------
        graph : Graph
            Graph that the node should be added to

        inA : Tensor or Scalar
            Input 1 data

        inB : Tensor or Scalar
            Input 2 data

        outA : Tensor or Scalar
            Output data
        """

        # create the kernel object
        k = cls(graph,inA,inB,outA)

        # create parameter objects for the input and output
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(inB,MPEnums.INPUT),
                  Parameter(outA,MPEnums.OUTPUT))

        # add the kernel to a generic node object
        node = Node(graph,k,params)

        # add the node to the graph
        graph.add_node(node)

        # if initialization data is provided, then add it to the node
        if init_inputs is not None:
            node.add_initialization_data(init_inputs, init_labels)

        return node

class XorKernel(Binary,Kernel):
    """
    Kernel to perform logical XOR operation elementwise on
    two MindPype data containers (i.e. tensor or scalar)

    Numpy broadcasting rules apply.

    .. note::
        This kernel utilizes the numpy function
        :data:`logical_xor <numpy:numpy.logical_xor>`.

    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to

    inA : Tensor or Scalar
        Input 1 data

    inB : Tensor or Scalar
        Input 2 data

    outA : Tensor or Scalar
        Output data
    """

    def __init__(self,graph,inA,inB,outA):
        """ Init """
        super().__init__('XOR',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA,inB]
        self.outputs = [outA]

    def _process_data(self, inputs, outputs):
        """
        Perform logical XOR operation elementwise

        Parameters
        ----------
        inputs: list of Tensors or Scalars
            Input data container, list of length 2

        outputs: list of Tensors or Scalars
            Output data container, list of length 1
        """
        outputs[0].data = np.logical_xor(inputs[0].data, inputs[1].data)

    @classmethod
    def add_to_graph(cls,graph,inA,inB,outA,init_inputs=None,init_labels=None):
        """
        Factory method to create a logical XOR kernel
        and add it to a graph as a generic node object.

        Parameters
        ----------
        graph : Graph
            Graph that the node should be added to

        inA : Tensor or Scalar
            Input 1 data

        inB : Tensor or Scalar
            Input 2 data

        outA : Tensor or Scalar
            Output data
        """

        # create the kernel object
        k = cls(graph,inA,inB,outA)

        # create parameter objects for the input and output
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(inB,MPEnums.INPUT),
                  Parameter(outA,MPEnums.OUTPUT))

        # add the kernel to a generic node object
        node = Node(graph,k,params)

        # add the node to the graph
        graph.add_node(node)

        # if initialization data is provided, then add it to the node
        if init_inputs is not None:
            node.add_initialization_data(init_inputs, init_labels)

        return node


class GreaterKernel(Binary,Kernel):
    """
    Kernel to perform greater than logical operation elementwise on
    two MindPype data containers (i.e. tensor or scalar)

    Numpy broadcasting rules apply.

    Parameters
    ----------

    graph : Graph
        Graph that the kernel should be added to

    inA : Tensor or Scalar
        Input 1 data

    inB : Tensor or Scalar
        Input 2 data

    outA : Tensor or Scalar
        Output data

    Note: The calculation is _inA .> _inB
    """

    def __init__(self,graph,inA,inB,outA):
        """ Init """
        super().__init__('Greater',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA,inB]
        self.outputs = [outA]


    def _process_data(self, inputs, outputs):
        """
        Perform greater than logical operation elementwise

        Parameters
        ----------
        inputs: list of Tensors or Scalars
            Input data container, list of length 2

        outputs: list of Tensors or Scalars
            Output data container, list of length 1
        """
        outputs[0].data = inputs[0].data > inputs[1].data

    @classmethod
    def add_to_graph(cls,graph,inA,inB,outA,init_inputs=None,init_labels=None):
        """
        Factory method to create a greater than comparison kernel
        and add it to a graph as a generic node object.

        Parameters
        ----------
        graph : Graph
            Graph that the kernel should be added to

        inA : Tensor or Scalar
            Input 1 data

        inB : Tensor or Scalar
            Input 2 data

        outA : Tensor or Scalar
            Output data

        Note: The calculation is _inA .> _inB
        """

        # create the kernel object
        k = cls(graph,inA,inB,outA)

        # create parameter objects for the input and output
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(inB,MPEnums.INPUT),
                  Parameter(outA,MPEnums.OUTPUT))

        # add the kernel to a generic node object
        node = Node(graph,k,params)

        # add the node to the graph
        graph.add_node(node)

        # if initialization data is provided, then add it to the node
        if init_inputs is not None:
            node.add_initialization_data(init_inputs, init_labels)

        return node

class LessKernel(Binary,Kernel):
    """
    Kernel to perform less than logical operation elementwise on
    two MindPype data containers (i.e. tensor or scalar)

    Numpy broadcasting rules apply.

    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to

    inA : Tensor or Scalar
        Input 1 data

    inB : Tensor or Scalar
        Input 2 data

    outA : Tensor or Scalar
        Output data

    Note: The calculation is _inA .< _inB
    """

    def __init__(self,graph,inA,inB,outA):
        """ Init """
        super().__init__('Less',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA,inB]
        self.outputs = [outA]

    def _process_data(self, inputs, outputs):
        """
        Perform less than logical operation elementwise

        Parameters
        ----------
        inputs: list of Tensors or Scalars
            Input data container, list of length 2

        outputs: list of Tensors or Scalars
            Output data container, list of length 1
        """
        outputs[0].data = inputs[0].data < inputs[1].data

    @classmethod
    def add_to_graph(cls,graph,inA,inB,outA,init_inputs=None,init_labels=None):
        """
        Factory method to create a less than comparison kernel
        and add it to a graph as a generic node object.

        Parameters
        ----------

        graph : Graph
            Graph that the node should be added to

        inA : Tensor or Scalar
            Input 1 data

        inB : Tensor or Scalar
            Input 2 data

        outA : Tensor or Scalar
            Output data

        Note: The calculation is _inA .< _inB
        """

        # create the kernel object
        k = cls(graph,inA,inB,outA)

        # create parameter objects for the input and output
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(inB,MPEnums.INPUT),
                  Parameter(outA,MPEnums.OUTPUT))

        # add the kernel to a generic node object
        node = Node(graph,k,params)

        # add the node to the graph
        graph.add_node(node)

        # if initialization data is provided, then add it to the node
        if init_inputs is not None:
            node.add_initialization_data(init_inputs, init_labels)

        return node

class EqualKernel(Binary,Kernel):
    """
    Kernel to perform equal to logical operation elementwise on
    two MindPype data containers (i.e. tensor or scalar)

    Numpy broadcasting rules apply.

    Parameters
    ----------

    graph : Graph
        Graph that the kernel should be added to

    inA : Tensor or Scalar
        Input 1 data

    inB : Tensor or Scalar
        Input 2 data

    outA : Tensor or Scalar
        Output data
    """

    def __init__(self,graph,inA,inB,outA):
        """ Init """
        super().__init__('Equal',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA,inB]
        self.outputs = [outA]

    def _process_data(self, inputs, outputs):
        """
        Perform equal to logical operation elementwise

        Parameters
        ----------
        inputs: list of Tensors or Scalars
            Input data container, list of length 2

        outputs: list of Tensors or Scalars
            Output data container, list of length 1
        """
        outputs[0].data = inputs[0].data == inputs[1].data

    @classmethod
    def add_to_graph(cls,graph,inA,inB,outA,init_inputs=None,init_labels=None):
        """
        Factory method to create a equality comparison kernel
        and add it to a graph as a generic node object.

        Parameters
        ----------

        graph : Graph
            Graph that the node should be added to

        inA : Tensor or Scalar
            Input 1 data

        inB : Tensor or Scalar
            Input 2 data

        outA : Tensor or Scalar
            Output data
        """

        # create the kernel object
        k = cls(graph,inA,inB,outA)

        # create parameter objects for the input and output
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(inB,MPEnums.INPUT),
                  Parameter(outA,MPEnums.OUTPUT))

        # add the kernel to a generic node object
        node = Node(graph,k,params)

        # add the node to the graph
        graph.add_node(node)

        # if initialization data is provided, then add it to the node
        if init_inputs is not None:
            node.add_initialization_data(init_inputs, init_labels)

        return node
