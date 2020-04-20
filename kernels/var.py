"""
Created on Sun Apr  5 17:13:07 2020

@author: Nick
"""

from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.bcip_enums import BcipEnums

import numpy as np

class VarKernel(Kernel):
    """
    Calculates the variance of values in a tensor
    """
    
    def __init__(self,graph,inA,outA,axis,ddof,keep_dims):
        """
        Kernal calculates arithmetic variance of values in tensor
        """
        super().__init__('Var',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._outA = outA
        self._axis = axis
        self._ddof = ddof
        self._keep_dims = keep_dims
        
    
    def initialize(self):
        """
        No internal state to setup
        """
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
            if self._keep_dims:
                output_rank = len(input_shape)
            else:
                output_rank = 1
            output_shape = (1,) * output_rank
        else:
            if self._keep_dims:
                output_shape = input_shape[:self._axis] + (1,) + input_shape[self._axis+1:]
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
        
    def execute(self):
        """
        Execute the kernel and calculate the mean
        """
        
        try:
            self._outA.data = np.var(self._inA.data,
                                      axis=self._axis,
                                      ddof=self._ddof,
                                      keepdims=self._keep_dims)
        except:
            return BcipEnums.EXE_FAILURE
        
        return BcipEnums.SUCCESS
    
    @classmethod
    def add_var_node(cls,graph,inA,outA,axis=None,ddof=0,keep_dims=False):
        """
        Factory method to create a variance kernel
        """
        
        # create the kernel object
        k = cls(graph,inA,outA,axis,ddof,keep_dims)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT), \
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node