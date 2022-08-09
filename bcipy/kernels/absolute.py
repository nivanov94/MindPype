from concurrent.futures import process
from types import NoneType
from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.bcip_enums import BcipEnums
from classes.scalar import Scalar
from classes.tensor import Tensor
from classes.bcip import BCIP

import numpy as np

class AbsoluteKernel(Kernel):
    """
    Calculate the element-wise absolute value of Tensor elements
    """
    
    def __init__(self,graph,inA,outA):
        super().__init__('Absolute',BcipEnums.INIT_FROM_NONE,graph)
        self._in   = inA
        self._out  = outA

        self.graph = graph

        self._labels = None

        self._init_inA = None
        self._init_outA = None
    
    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        if self._init_outA.__class__ != NoneType:
            return self.initialization_execution()
        
        return BcipEnums.SUCCESS
        
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        # input/output must be a tensor or scalar
        if not ((isinstance(self._in,Tensor) and isinstance(self._out,Tensor)) or \
                (isinstance(self._in,Scalar) and isinstance(self._out,Scalar))):
            return BcipEnums.INVALID_PARAMETERS

        if isinstance(self._in,Tensor):
            # input tensor must contain some values
            if len(self._in.shape) == 0:
                return BcipEnums.INVALID_PARAMETERS

        if isinstance(self._out,Tensor):
            if self._out.virtual() and len(self._out.shape) == 0:
                self._out.shape = self._in.shape

            if self._out.shape != self._in.shape:
                return BcipEnums.INVALID_PARAMETERS

        else:
            if not (self._in.data_type in Scalar.valid_numeric_types()):
                return BcipEnums.INVALID_PARAMETERS

            if self._out.data_type != self._in.data_type:
                return BcipEnums.INVALID_PARAMETERS

        return BcipEnums.SUCCESS
    
    def initialization_execution(self):
        """Process initialization data"""
        sts = self.process_data(self._init_inA, self._init_outA)
        if sts != BcipEnums.SUCCESS:
            return BcipEnums.INITIALIZATION_FAILURE
        
        return sts

    def process_data(self, input_data, output_data):
        try:
            if isinstance(input_data, Tensor):
                output_data.data = np.absolute(input_data.data)
            else:
                output_data.data = abs(input_data.data)
        except:
            return BcipEnums.EXE_FAILURE
        
        return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute the kernel function using numpy function
        """
        return self.process_data(self._in, self._out)
        
    
    @classmethod
    def add_absolute_node(cls,graph,inA,outA):
        """
        Factory method to create an absolute value kernel 
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
