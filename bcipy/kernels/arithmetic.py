from ..core import BCIP, BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Scalar

import numpy as np

class Unary:
    def initialize(self):
        """
        Initialize the kernel if there is an internal state to initialize, including downstream initialization data
        """
        sts = BcipEnums.SUCCESS

        if self._init_outA != None:
            # set the output size, as needed
            if len(self._init_outA.shape) == 0:
                self._init_outA.shape = self._init_inA.shape

            sts = self._process_data(self._init_inA, self._init_outA)

            # pass on labels
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
        
        # input/output must be a tensor or scalar
        if not ((self._inA._bcip_type == BcipEnums.TENSOR and self._outA._bcip_type == BcipEnums.TENSOR) or 
                (self._inA._bcip_type == BcipEnums.SCALAR and self._outA._bcip_type == BcipEnums.SCALAR)):
            return BcipEnums.INVALID_PARAMETERS

        if self._inA._bcip_type == BcipEnums.TENSOR:
            # input tensor must contain some values
            if len(self._inA.shape) == 0:
                return BcipEnums.INVALID_PARAMETERS

        if self._outA._bcip_type == BcipEnums.TENSOR:
            if self._outA.virtual and len(self._outA.shape) == 0:
                self._outA.shape = self._inA.shape

            if self._outA.shape != self._inA.shape:
                return BcipEnums.INVALID_PARAMETERS

        else:
            if not (self._inA.data_type in Scalar.valid_numeric_types()):
                return BcipEnums.INVALID_PARAMETERS

            if self._outA.data_type != self._inA.data_type:
                return BcipEnums.INVALID_PARAMETERS

        return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute the kernel function with the input trial data
        """
        return self._process_data(self._inA, self._outA)
     

class AbsoluteKernel(Unary, Kernel):
    """
    Calculate the element-wise absolute value of Tensor elements

    Parameters
    ----------

    graph : Graph Object
        - Graph that the kernel should be added to

    inA : Tensor or Scalar object
        - Input trial data

    outA : Tensor or Scalar object
        - Output trial data
    """
    
    def __init__(self,graph,inA,outA):
        super().__init__('Absolute',BcipEnums.INIT_FROM_NONE,graph)
        self._inA   = inA
        self._outA  = outA

        self._init_labels_in = None
        self._init_labels_out = None

        self._init_inA = None
        self._init_outA = None

    
    def _process_data(self, input_data, output_data):
        """
        Calculate the absolute value of the input data, and assign it to the output data
        """
        try:
            if input_data._bcip_type == BcipEnums.TENSOR:
                output_data.data = np.absolute(input_data.data)
            else:
                output_data.data = abs(input_data.data)
        except:
            return BcipEnums.EXE_FAILURE
        
        return BcipEnums.SUCCESS

    
    @classmethod
    def add_absolute_node(cls,graph,inA,outA):
        """
        Factory method to create an absolute value kernel 
        and add it to a graph as a generic node object.

        Parameters
        ----------
        
        graph : Graph Object
            - Graph that the kernel should be added to

        inA : Tensor or Scalar object
            - Input trial data

        outA : Tensor or Scalar object
            - Output trial data
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


class LogKernel(Unary,Kernel):
    """
    Kernel to perform element-wise natural logarithm operation on
    one BCIP data container (i.e. tensor or scalar)
    
    Numpy broadcasting rules apply.

    Parameters
    ----------
    graph : Graph Object
        - Graph that the kernel should be added to

    inA : Tensor or Scalar object
        - Input trial data

    outA : Tensor or Scalar object
        - Output trial data

    """
    
    def __init__(self,graph,inA,outA):
        super().__init__('Log',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._outA = outA

        self._init_inA = None
        self._init_outA = None

        self._init_labels_in = None
        self._init_labels_out = None


    def _process_data(self, input_data,output_data):
        try:
            data = np.log(input_data.data)
            if output_data._bcip_type == BcipEnums.SCALAR:
                output_data.data = data.item()
            else:
                output_data.data = data
        except:
            return BcipEnums.EXE_FAILURE

        return BcipEnums.SUCCESS

    @classmethod
    def add_log_node(cls,graph,inA,outA):
        """
        Factory method to create a log kernel 
        and add it to a graph as a generic node object.

        Parameters
        ----------
        graph : Graph Object
            - Graph that the node should be added to

        inA : Tensor or Scalar object
            - Input trial data

        outA : Tensor or Scalar object
            - Output trial data
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

        if self._init_outA != None:
            # determine output dimensions and adjust init_outA shape
            inA = self._init_inA.data
            inB = self._init_inB.data
            try:
                phony_out = inA + inB
                self._init_outA.shape = phony_out.shape
                sts = self._process_data(self._init_inA,self._init_inB,self._init_outA)
            except:
                sts = BcipEnums.INIT_FAILURE    
        
        return sts

    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        # first ensure the inputs and outputs are the appropriate type
        for operand in (self._inA, self._inB):
            if not (operand._bcip_type == BcipEnums.TENSOR or operand._bcip_type == SCALAR):
                return BcipEnums.INVALID_PARAMETERS
        
        if (self._inA._bcip_type == BcipEnums.TENSOR or 
            self._inB._bcip_type == BcipEnums.TENSOR):
            if self._outA._bcip_type != BcipEnums.TENSOR:
                # if one of the inputs is a tensor, the output will be a tensor
                return BcipEnums.INVALID_PARAMETERS
        elif self._outA._bcip_type != BcipEnums.SCALAR:
            # o.w. the output should be a scalar
            return BcipEnums.INVALID_PARAMETERS
        
        # if the inputs are scalars, ensure they are numeric
        for param in (self._inA,self._inB,self._outA):
            if (param._bcip_type == BcipEnums.SCALAR and 
                param.data_type not in Scalar.valid_numeric_types()):
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
            
            phony_out = phony_a + phony_b
        
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
 

class AdditionKernel(Binary, Kernel):
    """
    Kernel to add two BCIPP data containers (i.e. tensor or scalar) together

    graph : Graph Object
        - Graph that the kernel should be added to

    inA : Tensor or Scalar object
        - First input trial data

    inB : Tensor or Scalar object
        - Second input trial data

    outA : Tensor or Scalar object
        - Output trial data
    """
    
    def __init__(self,graph,inA,inB,outA):
        super().__init__('Addition',BcipEnums.INIT_FROM_NONE,graph)
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
        Calculate the absolute value of the input data, and assign it to the output data
        """
        try:
            output_data.data = input_data1.data + input_data2.data

        except ValueError:
            return BcipEnums.EXE_FAILURE
            
        return BcipEnums.SUCCESS

    @classmethod
    def add_addition_node(cls,graph,inA,inB,outA):
        """
        Factory method to create an addition kernel and add it to a graph
        as a generic node object.

        graph : Graph Object
        - Graph that the kernel should be added to

        inA : Tensor or Scalar object
            - First input trial data

        inB : Tensor or Scalar object
            - Second input trial data

        outA : Tensor or Scalar object
            - Output trial data
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

class DivisionKernel(Binary,Kernel):
    """
    Kernel to divide two BCIP data containers (i.e. tensor or scalar)
    together
    
    Note: This is element-wise division

    graph : Graph Object
        - Graph that the kernel should be added to

    inA : Tensor or Scalar object
        - First input trial data

    inB : Tensor or Scalar object
        - Second input trial data

    outA : Tensor or Scalar object
        - Output trial data

    Note: The calculation is _inA ./ _inB
    """
    
    def __init__(self,graph,inA,inB,outA):
        super().__init__('Division',BcipEnums.INIT_FROM_NONE,graph)
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
            output_data.data = input_data1.data / input_data2.data

        except ValueError:
            return BcipEnums.EXE_FAILURE
            
        return BcipEnums.SUCCESS

    @classmethod
    def add_division_node(cls,graph,inA,inB,outA):
        """
        Factory method to create a element-wise divsion kernel and add it to a graph
        as a generic node object.

        graph : Graph Object
            - Graph that the kernel should be added to

        inA : Tensor or Scalar object
            - First input trial data

        inB : Tensor or Scalar object
            - Second input trial data

        outA : Tensor or Scalar object
            - Output trial data

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

class MultiplicationKernel(Binary,Kernel):
    """
    Kernel to multiply two BCIPP data containers (i.e. tensor or scalar)
    together
    
    Note: This is element-wise multiplication

    Parameters
    ----------
    graph : Graph Object
        - Graph that the kernel should be added to

    inA : Tensor or Scalar object
        - First input trial data

    inB : Tensor or Scalar object
        - Second input trial data

    outA : Tensor or Scalar object
        - Output trial data

    Note: The calculation is _inA .* _inB
    """
    
    def __init__(self,graph,inA,inB,outA):
        super().__init__('Multiplication',BcipEnums.INIT_FROM_NONE,graph)
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
           output_data.data = input_data1.data * input_data2.data

        except ValueError:
            return BcipEnums.EXE_FAILURE
            
        return BcipEnums.SUCCESS
 
    @classmethod
    def add_multiplication_node(cls,graph,inA,inB,outA):
        """
        Factory method to create a multiplication kernel and add it to a graph
        as a generic node object.

        graph : Graph Object
            - Graph that the node should be added to

        inA : Tensor or Scalar object
            - First input trial data

        inB : Tensor or Scalar object
            - Second input trial data

        outA : Tensor or Scalar object
            - Output trial data

        Note: The calculation is _inA .* _inB
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

class SubtractionKernel(Binary,Kernel):
    """
    Kernel to calculate the difference between two BCIP data containers 
    (i.e. tensor or scalar)

    graph : Graph Object
        - Graph that the kernel should be added to

    inA : Tensor or Scalar object
        - First input trial data

    inB : Tensor or Scalar object
        - Second input trial data

    outA : Tensor or Scalar object
        - Output trial data
    """
    
    def __init__(self,graph,inA,inB,outA):
        super().__init__('Subtraction',BcipEnums.INIT_FROM_NONE,graph)
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
        Process data according to outlined kernel function
        """
        try:
            output_data.data = input_data1.data - input_data2.data

        except ValueError:
            return BcipEnums.EXE_FAILURE
            
        return BcipEnums.SUCCESS

    @classmethod
    def add_subtraction_node(cls,graph,inA,inB,outA):
        """
        Factory method to create a kernel and add it to a graph
        as a generic node object.

        graph : Graph Object
            - Graph that the kernel should be added to

        inA : Tensor or Scalar object
            - First input trial data

        inB : Tensor or Scalar object
            - Second input trial data

        outA : Tensor or Scalar object
            - Output trial data
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

