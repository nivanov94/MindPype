# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:46:23 2019

@author: ivanovn
"""

from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.scalar import Scalar
from classes.bcip_enums import BcipEnums

import numpy as np

class AndKernel(Kernel):
    """
    Kernel to perform logical AND operation elementwise on
    two BCIPP data containers (i.e. tensor or scalar)
    
    Numpy broadcasting rules apply.
    """
    
    def __init__(self,graph,inA,inB,outA):
        super().__init__('AND',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._inB  = inB
        self._outA = outA
    
    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        return BcipEnums.SUCCESS
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        # first ensure the inputs and outputs are the appropriate type
        if not (isinstance(self._inA,Tensor) or isinstance(self._inA,Scalar)):
            return BcipEnums.INVALID_PARAMETERS
        
        if not (isinstance(self._inB,Tensor) or isinstance(self._inAB,Scalar)):
            return BcipEnums.INVALID_PARAMETERS
        
        if (isinstance(self._inA,Tensor) or isinstance(self._inB,Tensor)) and \
           (not isinstance(self._outA,Tensor)):
            # if one of the inputs is a tensor, the output will be a tensor
            return BcipEnums.INVALID_PARAMETERS
        elif not isinstance(self._outA,Scalar):
            # o.w. the output should be a scalar
            return BcipEnums.INVALID_PARAMETERS
        
        # if the inputs are scalars, ensure they are logical
        if isinstance(self._inA,Scalar) and \
           not self._inA.data_type == bool:
            return BcipEnums.INVALID_PARAMETERS
        
        if isinstance(self._inB,Scalar) and \
           not self._inB.data_type == bool:
            return BcipEnums.INVALID_PARAMETERS

        if isinstance(self._outA,Scalar) and \
           self._outA.data_type != bool:
            return BcipEnums.INVALID_PARAMETERS
        
        
        # check the shapes
        if isinstance(self._inA,Tensor):
            inA_shape = self._inA.shape
        else:
            inA_shape = (1,)
        
        if isinstance(self._inB,Tensor):
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
        if isinstance(self._outA,Tensor) and self._outA.virtual \
           and len(self._outA.shape) == 0:
            self._outA.shape = out_shape
        
        # ensure the output shape equals the expected output shape
        if isinstance(self._outA,Tensor) and self._outA.shape != out_shape:
            return BcipEnums.INVALID_PARAMETERS
        elif isinstance(self._outA,Scalar) and out_shape != (1,):
            return BcipEnums.INVALID_PARAMETERS
        else:
            return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel function using numpy function
        """
        
        try:
            self._outA.data = np.logical_and(self._inA.data, self._inB.data)

        except ValueError:
            return BcipEnums.EXE_ERROR
            
        return BcipEnums.SUCCESS
    
    @classmethod
    def add_and_node(cls,graph,inA,inB,outA):
        """
        Factory method to create a logical AND kernel 
        and add it to a graph as a generic node object.
        """
        
        # create the kernel object
        k = cls(graph,inA,inB,outA)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT), \
                  Parameter(inB,BcipEnums.INPUT), \
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node