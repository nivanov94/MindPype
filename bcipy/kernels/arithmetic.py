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

            sts = _process_data(self._init_inA, self._init_outA)
        
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

        self._labels = None

        self._init_inA = None
        self._init_outA = None
    
    def initialize(self):
        """
        Initialize the kernel if there is an internal state to initialize, including downstream initialization data
        """
        sts = BcipEnums.SUCCESS

        if self._init_outA != None:
            # set the output size, as needed
            if len(self._init_outA.shape) == 0:
                self._init_outA.shape = self._init_inA.shape

            sts = _process_data(self._init_inA, self._init_outA)
        
        return sts
        
    
#    def verify(self):
#        """
#        Verify the inputs and outputs are appropriately sized
#        """
#        
#        # input/output must be a tensor or scalar
#        if not ((self._inA._bcip_type == BcipEnums.TENSOR and self._outA._bcip_type == BcipEnums.TENSOR) or 
#                (self._inA._bcip_type == BcipEnums.SCALAR and self._outA._bcip_type == BcipEnums.SCALAR)):
#            return BcipEnums.INVALID_PARAMETERS
#
#        if self._inA._bcip_type == BcipEnums.TENSOR:
#            # input tensor must contain some values
#            if len(self._inA.shape) == 0:
#                return BcipEnums.INVALID_PARAMETERS
#
#        if self._outA._bcip_type == BcipEnums.TENSOR:
#            if self._outA.virtual and len(self._outA.shape) == 0:
#                self._outA.shape = self._inA.shape
#
#            if self._outA.shape != self._inA.shape:
#                return BcipEnums.INVALID_PARAMETERS
#
#        else:
#            if not (self._inA.data_type in Scalar.valid_numeric_types()):
#                return BcipEnums.INVALID_PARAMETERS
#
#            if self._outA.data_type != self._inA.data_type:
#                return BcipEnums.INVALID_PARAMETERS
#
#        return BcipEnums.SUCCESS
    
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

#    def execute(self):
#        """
#        Execute the kernel function with the input trial data
#        """
#        return self._process_data(self._inA, self._outA)
        
    
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
