from types import NoneType
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

    Parameters
    ----------
    graph : Graph Object
        - Graph that the kernel should be added to

    inA : Tensor object
        - Input data (max value will be extracted from here)

    outA : Tensor or Scalar object
        - Output trial data
    """
    
    def __init__(self,graph,inA,outA):
        super().__init__('Max',BcipEnums.INIT_FROM_NONE,graph)
        self._inA   = inA
        self._outA  = outA

        self._init_inA = None
        self._init_outA = None
        

        self._labels = None
    
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
        
        # input must be a tensor
        if not isinstance(self._inA,Tensor):
            return BcipEnums.INVALID_PARAMETERS

        # output must be a tensor or scalar
        if not ((isinstance(self._outA,Tensor) or isinstance(self._outA,Scalar))):
            return BcipEnums.INVALID_PARAMETERS

        # input tensor must contain some values
        if len(self._inA.shape) == 0:
            return BcipEnums.INVALID_PARAMETERS

        if isinstance(self._outA,Tensor):
            if self._outA.virtual() and len(self._outA.shape) == 0:
                self._outA.shape = (1,)

            if self._outA.shape != (1,):
                return BcipEnums.INVALID_PARAMETERS

        else:
            if self._outA.data_type != float:
                return BcipEnums.INVALID_PARAMETERS

        return BcipEnums.SUCCESS

    def initialization_execution(self):


        sts = self.process_data(self._init_inA, self._init_outA)
        
        if sts != BcipEnums.SUCCESS:
            return BcipEnums.INITIALIZATION_FAILURE
        
        return sts

    def process_data(self, input_data, output_data):
        try:
            if isinstance(self._outA,Scalar):
                output_data.data = np.amax(input_data.data).item()
            else:
                output_data.data = np.asarray([np.amax(input_data.data)])
        except:
            return BcipEnums.EXE_FAILURE
        
        return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute the kernel function using numpy function
        """
        
        return self.process_data(self._inA, self._outA)
    
    @classmethod
    def add_max_node(cls,graph,inA,outA):
        """
        Factory method to create a maximum value kernel 
        and add it to a graph as a generic node object.

        Parameters
        ----------
        graph : Graph Object
            - Graph that the node should be added to

        inA : Tensor object
            - Input data (max value will be extracted from here)

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
