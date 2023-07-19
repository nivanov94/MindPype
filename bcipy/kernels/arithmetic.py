from ..core import BCIP, BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Scalar, Tensor
from .kernel_utils import extract_nested_data

import numpy as np


class Unary:
    def initialize(self):
        """
        Initialize the kernel if there is an internal state to initialize, including downstream initialization data
        """
        sts = BcipEnums.SUCCESS

        # get the initialization params
        init_in = self.init_inputs[0]
        labels = self.init_input_labels
        init_out = self.init_outputs[0]

        # check if initialization is needed
        if init_in is None or init_out is None:
            # init not needed, return success
            return sts

        accepted_inputs = (BcipEnums.TENSOR,BcipEnums.ARRAY,BcipEnums.CIRCLE_BUFFER)
        
        # check the init inputs are in valid data objects
        for init_obj in (init_in,labels):
            if init_obj.bcip_type not in accepted_inputs:
                return BcipEnums.INITIALIZATION_FAILURE
    
        # extract the data from the input
        if init_in.bcip_type == BcipEnums.TENSOR: 
            X = init_in.data
        else:
            try:
                # extract the data from a potentially nested array of tensors
                X = extract_nested_data(init_in)
            except:
                return BcipEnums.INITIALIZATION_FAILURE    
    
        if labels.bcip_type == BcipEnums.TENSOR:    
            y = labels.data
        else:
            try:
                y = extract_nested_data(labels)
            except:
                return BcipEnums.INITIALIZATION_FAILURE

        
        
        # set the output size, as needed
        if init_out.virtual:
            init_out.shape = init_in.shape

        sts = self._process_data(init_in, init_out)

        # pass on labels
        if labels.bcip_type != BcipEnums.TENSOR:
            labels = labels.to_tensor()
        labels.copy_to(self.init_output_labels)

        return sts

    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """

        # input/output must be a tensor or scalar
        d_in = self.inputs[0]
        d_out = self.outputs[0]

        # ensure that the parameters are valid types
        for param in (d_in, d_out):
            if param.bcip_type not in (BcipEnums.TENSOR, BcipEnums.SCALAR):
                return BcipEnums.INVALID_PARAMETERS
        
        # ensure that the input and output are the same type
        if d_in.bcip_type != d_out.bcip_type:
            return BcipEnums.INVALID_PARAMETERS
        

        if d_in.bcip_type == BcipEnums.TENSOR:
            # input tensor must contain some values
            if len(d_in.shape) == 0:
                return BcipEnums.INVALID_PARAMETERS

        # check output shape
        if d_out.bcip_type == BcipEnums.TENSOR:
            if d_out.virtual and len(d_out.shape) == 0:
                d_out.shape = d_in.shape

            if d_out.shape != d_in.shape:
                return BcipEnums.INVALID_PARAMETERS

        else:
            if not (d_in.data_type in Scalar.valid_numeric_types()):
                return BcipEnums.INVALID_PARAMETERS

            if d_out.data_type != d_in.data_type:
                return BcipEnums.INVALID_PARAMETERS

        return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute the kernel function with the input trial data
        """
        return self._process_data(self.inputs[0], self.outputs[0])


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

    def _process_data(self, input_data, output_data):
        """
        Calculate the absolute value of the input data, and assign it to the output data
        """
        try:
            if input_data.bcip_type == BcipEnums.TENSOR:
                output_data.data = np.absolute(input_data.data)
            else:
                output_data.data = abs(input_data.data)
        except:
            return BcipEnums.EXE_FAILURE

        return BcipEnums.SUCCESS

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

    def _process_data(self, input_data, output_data):
        """
        Calculate the natural logarithm of the input data, and assign it to the output data

        Parameters
        ----------
        input_data : Tensor or Scalar 
            Input trial data
        output_data : Tensor or Scalar 
            Output trial data

        Return
        ------
        BcipEnums.SUCCESS or BcipEnums.EXE_FAILURE

        Return Type
        -----------
        BcipEnums
        """

        try:
            data = np.log(input_data.data)
            if output_data.bcip_type == BcipEnums.SCALAR:
                output_data.data = data.item()
            else:
                output_data.data = data
        except:
            return BcipEnums.EXE_FAILURE

        return BcipEnums.SUCCESS

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
    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        sts = BcipEnums.SUCCESS

        # get the initialization params
        init_inA = self.init_inputs[0]
        init_inB = self.init_inputs[1]
        labels = self.init_input_labels
        init_out = self.init_outputs[0]

        # check if initialization is needed
        if init_inA is None or init_inB is None or init_out is None:
            # init not needed, return success
            return sts

        accepted_data_inputs = (BcipEnums.TENSOR, BcipEnums.ARRAY,
                                BcipEnums.CIRCLE_BUFFER, BcipEnums.SCALAR)
        
        # check the init inputs are in valid data objects
        for init_obj in (init_inA, init_inB):
            if init_obj.bcip_type not in accepted_data_inputs:
                return BcipEnums.INITIALIZATION_FAILURE
            
        if (labels.bcip_type != BcipEnums.TENSOR and
            labels.bcip_type != BcipEnums.ARRAY and
            labels.bcip_type != BcipEnums.CIRCLE_BUFFER):
            return BcipEnums.INITIALIZATION_FAILURE
    
        # extract the data from the input
        X = [None] * 2
        for i, i_in in enumerate((init_inA, init_inB)):
            if i_in.bcip_type == BcipEnums.TENSOR: 
                X[i] = i_in.data
            else:
                try:
                    # extract the data from a potentially nested array of tensors
                    X[i] = extract_nested_data(i_in)
                except:
                    return BcipEnums.INITIALIZATION_FAILURE

        if labels.bcip_type == BcipEnums.TENSOR:    
            y = labels.data
        else:
            try:
                y = extract_nested_data(labels)
            except:
                return BcipEnums.INITIALIZATION_FAILURE

        # determine output dimensions and adjust init_out shape
        try:
            phony_out = X[0] + X[1]
            init_out.shape = phony_out.shape
            tmp_inA = Tensor.create_from_data(self.session, X[0].shape, X[0])
            tmp_inB = Tensor.create_from_data(self.session, X[1].shape, X[1])
            sts = self._process_data(tmp_inA, tmp_inB, init_out)
        except:
            sts = BcipEnums.INIT_FAILURE

        # pass on labels
        if labels.bcip_type != BcipEnums.TENSOR:
            labels = labels.to_tensor()
        labels.copy_to(self.init_output_labels)

        return sts

    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        # get the input and output parameters
        d_inA = self.inputs[0]
        d_inB = self.inputs[1]
        d_out = self.outputs[0]

        # first ensure the inputs and outputs are the appropriate type
        accepted_types = (BcipEnums.TENSOR, BcipEnums.SCALAR)
        for operand in (d_inA, d_inB):
            if operand.bcip_type not in accepted_types:
                return BcipEnums.INVALID_PARAMETERS

        if (d_inA.bcip_type == BcipEnums.TENSOR
            or d_inB.bcip_type == BcipEnums.TENSOR):
            if d_out.bcip_type != BcipEnums.TENSOR:
                # if one of the inputs is a tensor, the output will be a tensor
                return BcipEnums.INVALID_PARAMETERS
        elif d_out.bcip_type != BcipEnums.SCALAR:
            # o.w. the output should be a scalar
            return BcipEnums.INVALID_PARAMETERS

        # if the inputs are scalars, ensure they are numeric
        for param in (d_inA, d_inB, d_out):
            if (param.bcip_type == BcipEnums.SCALAR
                and param.data_type not in Scalar.valid_numeric_types()):
                return BcipEnums.INVALID_PARAMETERS

        # check the shapes
        if d_inA.bcip_type == BcipEnums.TENSOR:
            inA_shape = d_inA.shape
        else:
            inA_shape = (1,)

        if d_inB.bcip_type == BcipEnums.TENSOR:
            inB_shape = d_inB.shape
        else:
            inB_shape = (1,)

        # determine what the output shape should be
        try:
            phony_a = np.zeros(inA_shape)
            phony_b = np.zeros(inB_shape)

            phony_out = phony_a + phony_b

        except ValueError:
            # these dimensions cannot be broadbast together
            return BcipEnums.INVALID_PARAMETERS

        out_shape = phony_out.shape

        # if the output is a virtual tensor and has no defined shape, set the shape now
        if (
            d_out.bcip_type == BcipEnums.TENSOR
            and d_out.virtual
            and len(d_out.shape) == 0
        ):
            d_out.shape = out_shape

        # ensure the output shape equals the expected output shape
        if d_out.bcip_type == BcipEnums.TENSOR and d_out.shape != out_shape:
            return BcipEnums.INVALID_PARAMETERS
        elif d_out.bcip_type == BcipEnums.SCALAR and out_shape != (1,):
            return BcipEnums.INVALID_PARAMETERS
        else:
            return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute the kernel function using numpy function
        """
        return self._process_data(self.inputs[0], self.inputs[1], self.outputs[0])


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

    def _process_data(self, input_data1, input_data2, output_data):
        """
        Calculate the absolute value of the input data, and assign it to the output data

        Parameters
        ----------
        input_data1 : Tensor or Scalar 
            First input trial data
        input_data2 : Tensor or Scalar 
            Second input trial data
        output_data : Tensor or Scalar 
            Output trial data

        Returns
        -------
        sts : BcipEnums.SUCCESS or BcipEnums.EXE_FAILURE
            Status of the execution

        Return type
        -----------
        BcipEnums
        """
        try:
            output_data.data = input_data1.data + input_data2.data

        except ValueError:
            return BcipEnums.EXE_FAILURE

        return BcipEnums.SUCCESS

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

    def _process_data(self, input_data1, input_data2, output_data):
        """
        Calculate the quotient of the input tensors, and assign it to the output data

        Parameters
        ----------
        input_data1 : Tensor or Scalar 
            First input trial data
        input_data2 : Tensor or Scalar 
            Second input trial data
        output_data : Tensor or Scalar 
            Output trial data

        Returns
        -------
        sts : BcipEnums.SUCCESS or BcipEnums.EXE_FAILURE
            Status of the execution

        Return type
        -----------
        BcipEnums
        """
        try:
            output_data.data = input_data1.data / input_data2.data

        except ValueError:
            return BcipEnums.EXE_FAILURE

        return BcipEnums.SUCCESS

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

    def _process_data(self, input_data1, input_data2, output_data):
        """
        Calculate the product of the input tensors, and assign it to the output data

        Parameters
        ----------
        input_data1 : Tensor or Scalar 
            First input trial data
        input_data2 : Tensor or Scalar 
            Second input trial data
        output_data : Tensor or Scalar 
            Output trial data

        Returns
        -------
        sts : BcipEnums.SUCCESS or BcipEnums.EXE_FAILURE
            Status of the execution

        Return type
        -----------
        BcipEnums
        """
        try:
            output_data.data = input_data1.data * input_data2.data

        except ValueError:
            return BcipEnums.EXE_FAILURE

        return BcipEnums.SUCCESS

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

    def _process_data(self, input_data1, input_data2, output_data):
        """
        Process data according to outlined kernel function

        Parameters
        ----------
        input_data1 : Tensor or Scalar 
            First input trial data
        input_data2 : Tensor or Scalar 
            Second input trial data
        output_data : Tensor or Scalar 
            Output trial data

        Returns
        -------
        sts : BcipEnums.SUCCESS or BcipEnums.EXE_FAILURE
            Status of the execution

        Return type
        -----------
        BcipEnums
        """
        try:
            output_data.data = input_data1.data - input_data2.data

        except ValueError:
            return BcipEnums.EXE_FAILURE

        return BcipEnums.SUCCESS

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
