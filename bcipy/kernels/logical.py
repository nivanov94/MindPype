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

        if self._init_outA is not None and (self._init_inA is not None and self._init_inA.shape != ()):
            # set the output size, as needed
            if self._init_outA.virtual:
                self._init_outA.shape = self._init_inA.shape

            sts = self._process_data(self._init_inA, self._init_outA)

            # pass on the labels
            if self._init_labels_in._bcip_type != BcipEnums.TENSOR:
                input_labels = self._init_labels_in.to_tensor()
            else:
                input_labels = self._init_labels_in
            input_labels.copy_to(self._init_labels_out) 
        
        return sts

    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        # first ensure the inputs and outputs are the appropriate type
        if self._inA._bcip_type != BcipEnums.TENSOR and self._inA._bcip_type != BcipEnums.SCALAR:
            return BcipEnums.INVALID_PARAMETERS
        
        if self._inA._bcip_type == BcipEnums.TENSOR and self._outA._bcip_type != BcipEnums.TENSOR:
            # if  the input is a tensor, the output will be a tensor
            return BcipEnums.INVALID_PARAMETERS
        elif self._outA._bcip_type != BcipEnums.SCALAR:
            # o.w. the output should be a scalar
            return BcipEnums.INVALID_PARAMETERS
        
        # if the params are scalars, ensure they are logical
        for param in (self._inA, self._outA):
            if (param._bcip_type == BcipEnums.SCALAR and 
                param.data_type != bool):
                return BcipEnums.INVALID_PARAMETERS

        # check the shapes
        if self._inA._bcip_type == BcipEnums.TENSOR:
            inA_shape = self._inA.shape
        else:
            inA_shape = (1,)
        
        out_shape = inA_shape
        
        # if the output is a virtual tensor and has no defined shape, set the shape now
        if (self._outA._bcip_type == BcipEnums.TENSOR and self._outA.virtual 
            and len(self._outA.shape) == 0):
            self._outA.shape = out_shape
        
        # ensure the output shape equals the expected output shape
        if self._outA._bcip_type == BcipEnums.TENSOR and self._outA.shape != out_shape:
            return BcipEnums.INVALID_PARAMETERS
        elif self._outA._bcip_type == BcipEnums.SCALAR and out_shape != (1,):
            return BcipEnums.INVALID_PARAMETERS
        else:
            return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute the kernel function using numpy function
        """
        return self._process_data(self._inA, self._outA)
 

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
        self._inA  = inA
        self._outA = outA

        self._init_inA = None
        self._init_outA = None

        self._init_labels_in = None
        self._init_labels_out = None
 
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

        if self._init_outA is not None and (self._init_inA is not None and self._init_inA.shape != ()):
            # determine output dimensions and adjust init_outA shape
            inA = self._init_inA.data
            inB = self._init_inB.data
            try:
                phony_out = np.logical_and(inA, inB)
                self._init_outA.shape = phony_out.shape
                sts = self._process_data(self._init_inA,self._init_inB,self._init_outA)
            except:
                sts = BcipEnums.INIT_FAILURE

            # pass on the labels
            if self._init_labels_in._bcip_type != BcipEnums.TENSOR:
                input_labels = self._init_labels_in.to_tensor()
            else:
                input_labels = self._init_labels_in
            input_labels.copy_to(self._init_labels_out)
        
        return sts

    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        # first ensure the inputs and outputs are the appropriate type
        for operand in (self._inA, self._inB):
            if (operand._bcip_type != BcipEnums.TENSOR and
                operand._bcip_type != BcipEnums.SCALAR):
                return BcipEnums.INVALID_PARAMETERS
        
        if (self._inA._bcip_type == BcipEnums.TENSOR or 
            self._inB._bcip_type == BcipEnums.TENSOR): 
            if self._outA._bcip_type != BcipEnums.TENSOR:
                # if one of the inputs is a tensor, the output will be a tensor
                return BcipEnums.INVALID_PARAMETERS
        elif self._outA._bcip_type != BcipEnums.SCALAR:
            # o.w. the output should be a scalar
            return BcipEnums.INVALID_PARAMETERS
        
        # if the inputs are scalars, ensure they are logical
        for param in (self._inA, self._inB, self._outA):
            if (param._bcip_type == BcipEnums.SCALAR and 
                param.data_type != bool):
                return BcipEnums.INVALID_PARAMETERS
        
        # check the shapes
        if self._inA._bcip_type == BcipEnums.TENSOR:
            inA_shape = self._inA.shape
        else:
            inA_shape = (1,)
        
        if self._inB._bcip_type == BcipEnums.TENSOR:
            inB_shape = self._inB.shape
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
        if (self._outA._bcip_type == BcipEnums.TENSOR and self._outA.virtual 
            and len(self._outA.shape) == 0):
            self._outA.shape = out_shape
        
        # ensure the output shape equals the expected output shape
        if self._outA._bcip_type == BcipEnums.TENSOR and self._outA.shape != out_shape:
            return BcipEnums.INVALID_PARAMETERS
        elif self._outA._bcip_type == BcipEnums.SCALAR and out_shape != (1,):
            return BcipEnums.INVALID_PARAMETERS
        else:
            return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute the kernel function using numpy function
        """
        return self._process_data(self._inA, self._inB, self._outA)


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
        self._inA  = inA
        self._inB  = inB
        self._outA = outA

        self._init_inA = None
        self._init_inB = None
        self._init_outA = None

        self._init_labels_in = None
        self._init_labels_out = None

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
        self._inA  = inA
        self._inB  = inB
        self._outA = outA

        self._init_inA = None
        self._init_inB = None
        self._init_outA = None

        self._init_labels_in = None
        self._init_labels_out = None


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
        self._inA  = inA
        self._inB  = inB
        self._outA = outA

        self._init_inA = None
        self._init_inB = None
        self._init_outA = None

        self._init_labels_in = None
        self._init_labels_out = None

 
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
        self._inA  = inA
        self._inB  = inB
        self._outA = outA

        self._init_inA = None
        self._init_inB = None
        self._init_outA = None

        self._init_labels_in = None
        self._init_labels_out = None

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
        self._inA  = inA
        self._inB  = inB
        self._outA = outA

        self._init_inA = None
        self._init_inB = None
        self._init_outA = None
        
        self._init_labels_in = None
        self._init_labels_out = None

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
        self._inA  = inA
        self._inB  = inB
        self._outA = outA

        self._init_inA = None
        self._init_inB = None
        self._init_outA = None

        self._init_labels_in = None
        self._init_labels_out = None

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
