"""
Created on Fri Mar  6 11:14:30 2020

@author: ivanovn
"""

from types import NoneType
from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.bcip_enums import BcipEnums

import numpy as np

class StdKernel(Kernel):
    """
    Calculates the standard deviation of values in a tensor

    graph : Graph Object
        - Graph that the kernel should be added to

    inA : Tensor object
        - First input trial data

    outA : Tensor object
        - Output trial data

    axis : None or int or tuple of ints, optional
        - Axis or axes along which the standard deviation is computed. The default is to compute the standard deviation of the flattened array.

    ddof : int, optional
        - Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements. By default ddof is zero.
    """
    
    def __init__(self,graph,inA,outA,axis,ddof):
        """
        Kernal calculates arithmetic standard deviation of values in tensor
        """
        super().__init__('Std',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._outA = outA
        self._axis = axis
        self._ddof = ddof
        
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
        Verify the inputs and outputs are appropriately sized and typed
        """
        
        # first ensure the input and output are tensors
        if (not isinstance(self._inA,Tensor)) or \
            (not isinstance(self._outA,Tensor)):
                return BcipEnums.INVALID_PARAMETERS
        
        input_shape = self._inA.shape
        
        if self._axis == None:
            output_shape = (1,)
        else:
            output_shape = tuple([x for i,x in enumerate(input_shape) if i != self._axis])

        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if (self._outA.virtual and len(self._outA.shape) == 0):
            self._outA.shape = output_shape
        
        # check output shape
        if self._outA.shape != output_shape:
            return BcipEnums.INVALID_PARAMETERS
  
        return BcipEnums.SUCCESS

    def initialization_execution(self):
        """
        Process initialization data if downstream nodes are missing training data
        """
        sts = self.process_data(self._init_inA, self._init_outA)
        
        if sts != BcipEnums.SUCCESS:
            return BcipEnums.INITIALIZATION_FAILURE
        
        return sts

    def process_data(self, input_data, output_data):
        """
        Process data according to outlined kernel function
        """
        try:
            output_data.data = np.std(input_data.data,
                                      axis=self._axis,
                                      ddof=self._ddof)
        except:
            return BcipEnums.EXE_FAILURE
        
        return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute the kernel and calculate the mean
        """
        return self.process_data(self._inA, self._outA)
    
    @classmethod
    def add_std_node(cls,graph,inA,outA,axis=None,ddof=0):
        """
        Factory method to create a Riemann mean calculating kernel
        """
        
        # create the kernel object
        k = cls(graph,inA,outA,axis,ddof)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT), \
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node