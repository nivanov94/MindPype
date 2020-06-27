# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 19:50:46 2020

@author: Nick
"""

from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.scalar import Scalar
from classes.bcip_enums import BcipEnums

import numpy as np

class LogKernel(Kernel):
    """
    Kernel to perform element-wise natural logarithm operation on
    one BCIP data container (i.e. tensor or scalar)
    
    Numpy broadcasting rules apply.
    """
    
    def __init__(self,graph,inA,outA):
        super().__init__('Log',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
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
        
        if isinstance(self._inA,Tensor):
            if not isinstance(self._outA,Tensor):
                # if  the input is a tensor, the output will be a tensor
                return BcipEnums.INVALID_PARAMETERS
        elif not isinstance(self._outA,Scalar):
            # o.w. the output should be a scalar
            return BcipEnums.INVALID_PARAMETERS
        
        # if the inputs are scalars, ensure they are logical
        if (isinstance(self._inA,Scalar) and 
           not self._inA.data_type == bool):
            return BcipEnums.INVALID_PARAMETERS

        if (isinstance(self._outA,Scalar) and 
           self._outA.data_type != bool):
            return BcipEnums.INVALID_PARAMETERS
        
        # check the shapes
        if isinstance(self._inA,Tensor):
            inA_shape = self._inA.shape
        else:
            inA_shape = (1,)
    
        
        out_shape = inA_shape
        
        # if the output is a virtual tensor and has no defined shape, set the shape now
        if (isinstance(self._outA,Tensor) and self._outA.virtual 
           and len(self._outA.shape) == 0):
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
            data = np.log(self._inA.data)
            if isinstance(self._outA,Scalar):
                self._outA.data = data.item()
            else:
                self._outA.data = data

        except:
            return BcipEnums.EXE_ERROR
            
        return BcipEnums.SUCCESS
    
    @classmethod
    def add_log_node(cls,graph,inA,outA):
        """
        Factory method to create a log kernel 
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