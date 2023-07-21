from ..core import BCIP, BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Scalar

import numpy as np

class Unary:
    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        sts = BcipEnums.SUCCESS

        # get the input and output initialization parameters
        init_in = self.init_inputs[0]
        init_out = self.init_outputs[0]

        if init_out is not None and (init_in is not None and init_in.shape != ()):
            # set the output size, as needed
            if init_out.virtual:
                init_out.shape = init_in.shape

            sts = self._process_data(init_in, init_out)

            # pass on the labels
            self.copy_init_labels_to_output()
        
        return sts

    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        d_in = self.inputs[0]
        d_out = self.outputs[0]

        # first ensure the inputs and outputs are the appropriate type
        if d_in.bcip_type != BcipEnums.TENSOR and d_in.bcip_type != BcipEnums.SCALAR:
            return BcipEnums.INVALID_PARAMETERS
        
        if d_in.bcip_type == BcipEnums.TENSOR and d_out.bcip_type != BcipEnums.TENSOR:
            # if  the input is a tensor, the output will be a tensor
            return BcipEnums.INVALID_PARAMETERS
        elif d_out.bcip_type != BcipEnums.SCALAR:
            # o.w. the output should be a scalar
            return BcipEnums.INVALID_PARAMETERS
        
        # if the params are scalars, ensure they are logical
        for param in (d_in, d_out):
            if (param.bcip_type == BcipEnums.SCALAR and 
                param.data_type != bool):
                return BcipEnums.INVALID_PARAMETERS

        # check the shapes
        if d_in.bcip_type == BcipEnums.TENSOR:
            param_shape = d_in.shape
        else:
            param_shape = (1,)
        
        # if the output is a virtual tensor and has no defined shape, set the shape now
        if (d_out.bcip_type == BcipEnums.TENSOR and d_out.virtual 
            and len(d_out.shape) == 0):
            d_out.shape = param_shape
        
        # ensure the output shape equals the expected output shape
        if d_out.bcip_type == BcipEnums.TENSOR and d_out.shape != param_shape:
            return BcipEnums.INVALID_PARAMETERS
        elif d_out.bcip_type == BcipEnums.SCALAR and param_shape != (1,):
            return BcipEnums.INVALID_PARAMETERS
        else:
            return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute the kernel function using numpy function
        """
        return self._process_data(self.inputs[0], self.outputs[0])
 

class NotKernel(Unary, Kernel):
    """
    Kernel to perform logical NOT operation elementwise on
    one BCIPP data container (i.e. tensor or scalar)
    
    Numpy broadcasting rules apply.

    Parameters
    ----------
    graph : Graph 
        Graph that the kernel should be added to

    inA : Tensor or Scalar 
        First Input trial data

    outA : Tensor or Scalar 
        Output trial data
    """
    
    def __init__(self,graph,inA,outA):
        super().__init__('NOT',BcipEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA]
        self.outputs = [outA]
 
    def _process_data(self, input_data, output_data):
        try:
            data = np.logical_not(input_data.data)
            if isinstance(output_data,Scalar):
               output_data.data = data.item()
            else:
                output_data.data = data

        except:
            return BcipEnums.EXE_FAILURE
            
        return BcipEnums.SUCCESS

    @classmethod
    def add_not_node(cls,graph,inA,outA):
        """
        Factory method to create a logical NOT kernel 
        and add it to a graph as a generic node object.
        """
        
        # create the kernel object
        k = cls(graph,inA,outA)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node

class Binary:
    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        sts = BcipEnums.SUCCESS
        init_inA, init_inB = self.init_inputs
        init_out = self.init_outputs[0]

        if init_out is not None and (init_inA is not None and init_inA.shape != ()):
            # determine output dimensions and adjust init_outA shape
            try:
                phony_out = np.logical_and(init_inA.data, init_inB.data)
                init_out.shape = phony_out.shape
                sts = self._process_data(init_inA, init_inB, init_out)
            except:
                sts = BcipEnums.INIT_FAILURE

            # pass on the labels
            self.copy_init_labels_to_output()
        
        return sts

    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        inA, inB = self.inputs
        outA = self.outputs[0]

        # first ensure the inputs and outputs are the appropriate type
        for operand in (inA, inB):
            if (operand.bcip_type != BcipEnums.TENSOR and
                operand.bcip_type != BcipEnums.SCALAR):
                return BcipEnums.INVALID_PARAMETERS
        
        if (inA.bcip_type == BcipEnums.TENSOR or 
            inB.bcip_type == BcipEnums.TENSOR): 
            if outA.bcip_type != BcipEnums.TENSOR:
                # if one of the inputs is a tensor, the output will be a tensor
                return BcipEnums.INVALID_PARAMETERS
        elif outA.bcip_type != BcipEnums.SCALAR:
            # o.w. the output should be a scalar
            return BcipEnums.INVALID_PARAMETERS
        
        # if the inputs are scalars, ensure they are logical
        for param in (inA, inB, outA):
            if (param.bcip_type == BcipEnums.SCALAR and 
                param.data_type != bool):
                return BcipEnums.INVALID_PARAMETERS
        
        # check the shapes
        if inA.bcip_type == BcipEnums.TENSOR:
            inA_shape = inA.shape
        else:
            inA_shape = (1,)
        
        if inB.bcip_type == BcipEnums.TENSOR:
            inB_shape = inB.shape
        else:
            inB_shape = (1,)
        
        # determine what the output shape should be
        try:
            phony_a = np.zeros(inA_shape)
            phony_b = np.zeros(inB_shape)
            
            phony_out = np.logical_and(phony_a,phony_b)
        
        except ValueError:
            # these dimensions cannot be broadbast together
            return BcipEnums.INVALID_PARAMETERS
        
        out_shape = phony_out.shape
        
        # if the output is a virtual tensor and has no defined shape, set the shape now
        if (outA.bcip_type == BcipEnums.TENSOR and outA.virtual 
            and len(outA.shape) == 0):
            outA.shape = out_shape
        
        # ensure the output shape equals the expected output shape
        if outA.bcip_type == BcipEnums.TENSOR and outA.shape != out_shape:
            return BcipEnums.INVALID_PARAMETERS
        elif outA.bcip_type == BcipEnums.SCALAR and out_shape != (1,):
            return BcipEnums.INVALID_PARAMETERS
        else:
            return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute the kernel function using numpy function
        """
        return self._process_data(self.inputs[0], self.inputs[1], self.outputs[0])


class AndKernel(Binary,Kernel):
    """
    Kernel to perform logical AND operation elementwise on
    two BCIP data containers (i.e. tensor or scalar)
    
    Numpy broadcasting rules apply.

    Parameters
    ----------
    graph : Graph 
        Graph that the kernel should be added to

    inA : Tensor or Scalar 
        First Input trial data

    inB : Tensor or Scalar 
        Second Input trial data

    outA : Tensor or Scalar 
        Output trial data
    """
    
    def __init__(self,graph,inA,inB,outA):
        super().__init__('AND',BcipEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA,inB]
        self.outputs = [outA]

    def _process_data(self, input_data1, input_data2, output_data):
        """
        Process data according to the outlined kernel function
        """
        try:
            data = np.logical_and(input_data1.data,input_data2.data)
            if isinstance(output_data,Scalar):
                output_data.data = data.item()
            else:
                output_data.data = data

        except:
            return BcipEnums.EXE_FAILURE
            
        return BcipEnums.SUCCESS

    @classmethod
    def add_and_node(cls,graph,inA,inB,outA):
        """
        Factory method to create a logical AND kernel 
        and add it to a graph as a generic node object.

        Parameters
        ----------
        graph : Graph 
            Graph that the node should be added to

        inA : Tensor or Scalar 
            First Input trial data

        inB : Tensor or Scalar 
            Second Input trial data

        outA : Tensor or Scalar 
            Output trial data
        
        """
        
        # create the kernel object
        k = cls(graph,inA,inB,outA)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(inB,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node

class OrKernel(Binary,Kernel):
    """
    Kernel to perform logical OR operation elementwise on
    two BCIPP data containers (i.e. tensor or scalar)
    
    Numpy broadcasting rules apply.

    Parameters
    ----------
    graph : Graph 
        Graph that the kernel should be added to

    inA : Tensor or Scalar 
        First Input trial data

    inB : Tensor or Scalar 
        Second Input trial data

    outA : Tensor or Scalar 
        Output trial data
    """
    
    def __init__(self,graph,inA,inB,outA):
        super().__init__('OR',BcipEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA,inB]
        self.outputs = [outA]

    def _process_data(self, input_data1, input_data2, output_data):
        try:
            data = np.logical_or(input_data1.data, input_data2.data)
            if isinstance(output_data,Scalar):
               output_data.data = data.item()
            else:
                output_data.data = data

        except:
            return BcipEnums.EXE_FAILURE
            
        return BcipEnums.SUCCESS

    @classmethod
    def add_or_node(cls,graph,inA,inB,outA):
        """
        Factory method to create a logical OR kernel 
        and add it to a graph as a generic node object.

        Parameters
        ----------
        graph : Graph 
            Graph that the node should be added to

        inA : Tensor or Scalar 
            First Input trial data

        inB : Tensor or Scalar 
            Second Input trial data

        outA : Tensor or Scalar 
            Output trial data
        """
        
        # create the kernel object
        k = cls(graph,inA,inB,outA)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(inB,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node

class XorKernel(Binary,Kernel):
    """
    Kernel to perform logical XOR operation elementwise on
    two BCIPP data containers (i.e. tensor or scalar)
    
    Numpy broadcasting rules apply.

    Parameters
    ----------
    graph : Graph 
        Graph that the kernel should be added to

    inA : Tensor or Scalar 
        First Input trial data

    inB : Tensor or Scalar 
        Second Input trial data

    outA : Tensor or Scalar 
        Output trial data
    """
    
    def __init__(self,graph,inA,inB,outA):
        super().__init__('XOR',BcipEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA,inB]
        self.outputs = [outA]
 
    def process_data(self, input_data1, input_data2, output_data):
        try:
            data = np.logical_xor(input_data1.data, input_data2.data)
            if isinstance(output_data,Scalar):
               output_data.data = data.item()
            else:
                output_data.data = data

        except:
            return BcipEnums.EXE_FAILURE
            
        return BcipEnums.SUCCESS


    @classmethod
    def add_xor_node(cls,graph,inA,inB,outA):
        """
        Factory method to create a logical XOR kernel 
        and add it to a graph as a generic node object.

        Parameters
        ----------
        graph : Graph 
            Graph that the node should be added to

        inA : Tensor or Scalar 
            First Input trial data

        inB : Tensor or Scalar 
            Second Input trial data

        outA : Tensor or Scalar 
            Output trial data
        """
        
        # create the kernel object
        k = cls(graph,inA,inB,outA)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(inB,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node


class GreaterKernel(Binary,Kernel):
    """
    Kernel to perform greater than logical operation elementwise on
    two BCIPP data containers (i.e. tensor or scalar)
    
    Numpy broadcasting rules apply.

    graph : Graph 
        Graph that the kernel should be added to

    inA : Tensor or Scalar 
        First input trial data

    inB : Tensor or Scalar 
        Second input trial data

    outA : Tensor or Scalar 
        Output trial data

    Note: The calculation is _inA .> _inB
    """
    
    def __init__(self,graph,inA,inB,outA):
        super().__init__('Greater',BcipEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA,inB]
        self.outputs = [outA]

    def _process_data(self, input_data1, input_data2, output_data):
        try:
            output_data.data = input_data1.data > input_data2.data

        except:
            return BcipEnums.EXE_FAILURE
            
        return BcipEnums.SUCCESS

    @classmethod
    def add_greater_node(cls,graph,inA,inB,outA):
        """
        Factory method to create a greater than comparison kernel 
        and add it to a graph as a generic node object.

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

        Note: The calculation is _inA .> _inB
        """
        
        # create the kernel object
        k = cls(graph,inA,inB,outA)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(inB,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node

class LessKernel(Binary,Kernel):
    """
    Kernel to perform less than logical operation elementwise on
    two BCIPP data containers (i.e. tensor or scalar)
    
    Numpy broadcasting rules apply.


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

    Note: The calculation is _inA .< _inB
    """
    
    def __init__(self,graph,inA,inB,outA):
        super().__init__('Less',BcipEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA,inB]
        self.outputs = [outA]

    def _process_data(self, input_data1, input_data2, output_data):
        """Process data according to the kernel function using numpy data"""
        try:
            output_data.data = input_data1.data < input_data2.data

        except:
            return BcipEnums.EXE_FAILURE
            
        return BcipEnums.SUCCESS

    @classmethod
    def add_less_node(cls,graph,inA,inB,outA):
        """
        Factory method to create a less than comparison kernel 
        and add it to a graph as a generic node object.

        graph : Graph 
            Graph that the node should be added to

        inA : Tensor or Scalar 
            First input trial data

        inB : Tensor or Scalar 
            Second input trial data

        outA : Tensor or Scalar 
            Output trial data

        Note: The calculation is _inA .> _inB
        """
        
        # create the kernel object
        k = cls(graph,inA,inB,outA)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(inB,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node

class EqualKernel(Binary,Kernel):
    """
    Kernel to perform greater than logical operation elementwise on
    two BCIPP data containers (i.e. tensor or scalar)
    
    Numpy broadcasting rules apply.

    graph : Graph 
        Graph that the kernel should be added to

    inA : Tensor or Scalar 
        First input trial data

    inB : Tensor or Scalar 
        Second input trial data

    outA : Tensor or Scalar 
        Output trial data
    """
    
    def __init__(self,graph,inA,inB,outA):
        super().__init__('Equal',BcipEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA,inB]
        self.outputs = [outA]

    def _process_data(self, input_data1, input_data2, output_data):
        try:
            output_data.data = input_data1.data == input_data2.data

        except:
            return BcipEnums.EXE_FAILURE
            
        return BcipEnums.SUCCESS

    @classmethod
    def add_equal_node(cls,graph,inA,inB,outA):
        """
        Factory method to create a equality comparison kernel 
        and add it to a graph as a generic node object.
        """
        
        # create the kernel object
        k = cls(graph,inA,inB,outA)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(inB,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
