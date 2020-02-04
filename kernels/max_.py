from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.bcip_enums import BcipEnums
from classes.scalar import Scalar
from classes.tensor import Tensor
from classes.bcip import BCIP

import numpy as np


class MaxKernel(Kernel):
    """
    Kernel to extract maximum value within a Tensor
    """
    
    def __init__(self,graph,inA,outA):
        super().__init__('Max',BcipEnums.INIT_FROM_NONE,graph)
        self._in   = inA
        self._out  = outA
    
    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        return BcipEnums.SUCCESS
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        # input must be a tensor
        if not isinstance(self._in,Tensor):
            return BcipEnums.INVALID_PARAMETERS

        # output must be a tensor or scalar
        if not (isinstance(self._out,Tensor) or isinstance(self._out,Scalar)):
            return BcipEnums.INVALID_PARAMETERS

        # input tensor must contain some values
        if len(self._in.shape) == 0:
            return BcipEnums.INVALID_PARAMETERS

        if isinstance(self._out,Tensor):
            if self._out.virtual() and len(self._out.shape) == 0:
                self._out.shape = (1,)

            if self._out.shape != (1,):
                return BcipEnums.INVALID_PARAMETERS

        else:
            if self._out.data_type != float:
                return BcipEnums.INVALID_PARAMETERS

        return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel function using numpy function
        """
        
        try:
            if isinstance(self._out,Scalar):
                self._out.data = np.amax(self._in.data).item()
            else:
                self._out.data = np.asarray([np.amax(self._in.data)])
        except:
            return BcipEnums.EXE_FAILURE
        
        return BcipEnums.SUCCESS
    
    @classmethod
    def add_max_node(cls,graph,inA,outA):
        """
        Factory method to create a maximum value kernel 
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
